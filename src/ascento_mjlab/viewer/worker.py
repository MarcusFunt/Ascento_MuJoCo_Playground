"""Run an isolated one-environment Ascento policy in mjlab's Viser browser viewer."""

from __future__ import annotations

import argparse
import html
import itertools
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.checkpoint_contract import require_current_checkpoint_contracts
from ascento_mjlab.evaluation.policy import RslRlPolicyAdapter
from ascento_mjlab.geometry import projected_gravity_tilt
from ascento_mjlab.mdp.jump import PHASE_FLIGHT
from ascento_mjlab.physics import PHYSICS_PROFILE
from ascento_mjlab.structured_action import StructuredTargetAction

from .attribution import explain_integrated_gradients, normalizer_mean_baseline
from .checkpoints import CheckpointInfo, discover_checkpoints, resolve_run_checkpoint
from .diagnostics import compute_balance_confidence, smooth_value, sparkline
from .introspection import (
    PolicyIntrospector,
    StructuredActionObserver,
    build_action_pipeline_snapshot,
)
from .introspection_contract import build_policy_branch_schema, resolve_runtime_handles
from .ipc import IntrospectionIPC
from .observation_schema import build_observation_schema
from .replay import (
    EventTriggerDetector,
    PolicyReplayBuffer,
    PolicyReplayRecorder,
    TransitionSnapshot,
    checkpoint_iteration,
)


def _write_json(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _preflight_checkpoint(checkpoint_path: Path, canonical_env_cfg: Any) -> dict[str, Any]:
    """Validate checkpoint provenance before mutating the live viewer runner."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint is not a supported RSL-RL checkpoint")
    infos = payload.get("infos")
    if not isinstance(infos, dict):
        raise ValueError("checkpoint lacks provenance metadata")
    require_current_checkpoint_contracts(infos, canonical_env_cfg)
    return infos


def _load_actor_transactionally(
    runner: Any,
    checkpoint_path: Path,
    canonical_env_cfg: Any,
    *,
    device: str,
) -> dict[str, Any]:
    """Load actor weights without leaving a rejected candidate partially applied."""
    _preflight_checkpoint(checkpoint_path, canonical_env_cfg)
    actor = runner.alg.get_policy()
    previous_state = {
        name: value.detach().clone() for name, value in actor.state_dict().items()
    }
    try:
        infos = runner.load(
            str(checkpoint_path),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        require_current_checkpoint_contracts(infos, canonical_env_cfg)
    except Exception:
        runner.alg.get_policy().load_state_dict(previous_state, strict=True)
        runner.alg.eval_mode()
        raise
    return infos


class _ViewerPolicy:
    """Adapt the evaluation policy interface to mjlab's callable viewer protocol."""

    def __init__(
        self,
        adapter: RslRlPolicyAdapter,
        introspector: PolicyIntrospector,
        action_observer: StructuredActionObserver,
        next_sequence,
        *,
        wrapper_clip: float | None,
    ):
        self.adapter = adapter
        self.introspector = introspector
        self.action_observer = action_observer
        self.next_sequence = next_sequence
        self.wrapper_clip = wrapper_clip
        self.latest_frame = None

    def __call__(self, observations):
        sequence_id = self.next_sequence()
        capture = self.introspector.capture_action(
            observations,
            lambda: self.adapter.act(observations),
            sequence_id=sequence_id,
        )
        self.latest_frame = self.introspector.maybe_calculate_jacobian(capture.frame)
        return capture.action

    def complete_transition(self, transition: TransitionSnapshot | None = None):
        if self.latest_frame is None or self.action_observer.latest is None:
            return self.latest_frame
        pipeline = build_action_pipeline_snapshot(
            self.latest_frame.actor_output,
            self.action_observer.latest,
            wrapper_clip=self.wrapper_clip,
        )
        self.latest_frame = replace(
            self.latest_frame,
            action_pipeline=pipeline,
            transition=transition,
        )
        return self.latest_frame

    def close(self) -> None:
        self.introspector.close()

    def reset(self) -> None:
        self.adapter.reset()


class _FollowViserPlayViewer(ViserPlayViewer):
    """Viser viewer with checkpoint following and an always-visible diagnostic HUD."""

    def __init__(
        self,
        *args,
        follow: bool = False,
        follow_poll_seconds: float = 2.0,
        introspection_ipc: IntrospectionIPC | None = None,
        transition_observer: "ViewerTransitionObserver | None" = None,
        replay_buffer: PolicyReplayBuffer | None = None,
        replay_recorder: PolicyReplayRecorder | None = None,
        checkpoint_schema_provider=None,
        event_detector: EventTriggerDetector | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._follow = bool(follow)
        self._follow_poll_seconds = max(0.5, float(follow_poll_seconds))
        self._next_follow_poll = 0.0
        self._follow_rejected_checkpoint: str | None = None
        self._introspection_ipc = introspection_ipc
        self._transition_observer = transition_observer
        self._replay_buffer = replay_buffer
        self._replay_recorder = replay_recorder
        self._checkpoint_schema_provider = checkpoint_schema_provider
        self._event_detector = event_detector or EventTriggerDetector()
        self._episode_start_observations: dict[int, list[float]] = {}

        self._diagnostic_html = None
        self._diagnostic_freeze = None
        self._diagnostic_reward_breakdown = None
        self._diagnostic_lookahead = None
        self._diagnostic_smoothing = None
        self._diagnostic_history_seconds = None
        self._diagnostic_top_terms = None
        self._diagnostic_next_update = 0.0
        self._diagnostic_prev_tilt: float | None = None
        self._diagnostic_prev_episode_step: int | None = None
        self._diagnostic_smoothed_confidence: float | None = None
        self._diagnostic_reset_count = 0
        self._diagnostic_history: dict[str, deque[float]] = {
            "reward_rate": deque(maxlen=600),
            "tilt_deg": deque(maxlen=600),
            "confidence": deque(maxlen=600),
        }

    def setup(self) -> None:
        super().setup()
        self._setup_diagnostic_hud()
        self._update_diagnostic_hud(force=True)

    def _setup_diagnostic_hud(self) -> None:
        """Create a root-level panel that stays visible beside all mjlab tabs."""
        with self._server.gui.add_folder("Live diagnostic HUD"):
            self._diagnostic_html = self._server.gui.add_html(
                "<div style='padding:0.25em'>Waiting for the first simulation sample…</div>"
            )
            self._diagnostic_freeze = self._server.gui.add_checkbox(
                "Freeze HUD",
                initial_value=False,
                hint="Freeze the displayed values/history without pausing simulation.",
            )
            self._diagnostic_reward_breakdown = self._server.gui.add_checkbox(
                "Show reward terms",
                initial_value=True,
                hint="Show the largest weighted per-second reward contributions.",
            )
            self._diagnostic_lookahead = self._server.gui.add_slider(
                "Fall-risk look-ahead (s)",
                min=0.0,
                max=0.75,
                step=0.05,
                initial_value=0.25,
                hint=(
                    "Project only worsening measured tilt forward by this duration. "
                    "This changes the diagnostic confidence only, never the policy."
                ),
            )
            self._diagnostic_smoothing = self._server.gui.add_slider(
                "Confidence smoothing",
                min=0.0,
                max=0.95,
                step=0.05,
                initial_value=0.65,
                hint="0 = immediate; larger values make the displayed confidence steadier.",
            )
            self._diagnostic_history_seconds = self._server.gui.add_slider(
                "HUD history (s)",
                min=2,
                max=60,
                step=1,
                initial_value=12,
            )
            self._diagnostic_top_terms = self._server.gui.add_slider(
                "Reward terms shown",
                min=3,
                max=12,
                step=1,
                initial_value=6,
            )
            clear_button = self._server.gui.add_button("Clear HUD history")

            @clear_button.on_click
            def _(_) -> None:
                for history in self._diagnostic_history.values():
                    history.clear()
                self._diagnostic_prev_tilt = None
                self._diagnostic_prev_episode_step = None
                self._diagnostic_smoothed_confidence = None
                self._diagnostic_reset_count = 0
                self._diagnostic_next_update = 0.0

    def _process_actions(self) -> None:
        now = time.monotonic()
        manager = getattr(self, "_ckpt_mgr", None)
        if self._follow and manager is not None and now >= self._next_follow_poll:
            self._next_follow_poll = now + self._follow_poll_seconds
            entries = manager.fetch_available()
            latest = entries[-1][0] if entries else None
            if (
                latest is not None
                and latest != manager.current_name
                and latest != self._follow_rejected_checkpoint
            ):
                self._actions.append((ViewerAction.FETCH_CHECKPOINT, "latest"))
        super()._process_actions()

    def _handle_custom_action(self, action: ViewerAction, payload: Any) -> bool:
        manager = getattr(self, "_ckpt_mgr", None)
        previous = manager.current_name if manager is not None else None
        try:
            handled = super()._handle_custom_action(action, payload)
        except Exception as exc:
            if action != ViewerAction.FETCH_CHECKPOINT:
                raise
            attempted: str | None = None
            if manager is not None:
                if payload == "latest":
                    entries = manager.fetch_available()
                    attempted = entries[-1][0] if entries else None
                elif payload == "selected":
                    attempted = self._ckpt_dropdown.value.split("  (")[0]
            self._follow_rejected_checkpoint = attempted
            self._last_error = f"Checkpoint load rejected: {exc}"
            print(f"[WARN]: {self._last_error}")
            if manager is not None:
                current = next(
                    (
                        label
                        for label in self._ckpt_dropdown.options
                        if label.startswith(manager.current_name)
                    ),
                    manager.current_name,
                )
                self._ckpt_user_event.clear()
                self._ckpt_dropdown.value = current
                self._ckpt_user_event.set()
            return True
        if manager is not None and manager.current_name != previous:
            self._follow_rejected_checkpoint = None
            self._last_error = None
        return handled

    def sync_env_to_viewer(self) -> None:
        super().sync_env_to_viewer()
        self._update_diagnostic_hud()

    def _execute_step(self) -> bool:
        succeeded = super()._execute_step()
        if succeeded and isinstance(self.policy, _ViewerPolicy):
            transition = (
                self._transition_observer.latest
                if self._transition_observer is not None
                else None
            )
            frame = self.policy.complete_transition(transition)
            if frame is not None and self._replay_buffer is not None:
                self._replay_buffer.append(frame)
                self._capture_triggered_events(frame)
                if transition is not None and transition.reset:
                    self._replay_buffer.clear()
                    self._event_detector.reset()
            if frame is not None and self._introspection_ipc is not None:
                self._introspection_ipc.publish_frame(frame)
        return succeeded

    def _capture_triggered_events(self, frame) -> None:
        if self._replay_buffer is None or self._replay_recorder is None:
            return
        transition = getattr(frame, "transition", None)
        if transition is None:
            return
        if transition.episode_step <= 1:
            self._episode_start_observations[transition.episode_id] = frame.raw_observation.tolist()
            for episode_id in sorted(self._episode_start_observations)[:-32]:
                self._episode_start_observations.pop(episode_id, None)
        fall_boundary_rad, _ = self._fall_boundaries(self.env.unwrapped)
        event_type = self._event_detector.update(
            frame,
            fall_boundary_rad=fall_boundary_rad,
        )
        if event_type is not None:
            reason = "fallen termination term" if event_type == "fall" else (
                "stability margin recovered and remained above 70% for 0.5 seconds"
            )
            self._persist_capture(frame, event_type, reason, markers=self._event_detector.last_markers)
        if self._introspection_ipc is not None:
            if self._introspection_ipc.consume_manual_capture_request() is not None:
                self._persist_capture(frame, "manual", "manual dashboard capture")
            request = self._introspection_ipc.consume_explanation_request()
            if request is not None:
                self._process_explanation_request(frame, request)

    def _process_explanation_request(self, frame, request: dict[str, Any]) -> None:
        if self._introspection_ipc is None or not isinstance(self.policy, _ViewerPolicy):
            return
        explanation_id = str(request.get("explanation_id") or "")
        result: dict[str, Any]
        try:
            if request.get("checkpoint") != frame.checkpoint:
                raise ValueError("requested frame checkpoint is not the checkpoint currently loaded in the viewer")
            baseline_kind = request.get("baseline_kind")
            if baseline_kind == "normalizer_mean":
                baseline = normalizer_mean_baseline(self.policy.introspector.handles)
                baseline_description = (
                    "Raw input corresponding to normalized zero (the running mean when a normalizer exists; raw zero otherwise)."
                )
            elif baseline_kind == "episode_start":
                episode_id = request.get("episode_id")
                baseline_values = self._episode_start_observations.get(int(episode_id))
                if baseline_values is None:
                    raise ValueError("this viewer no longer has the first observation for that episode")
                baseline = baseline_values
                baseline_description = "First actor observation captured in the selected episode."
            elif baseline_kind == "selected_frame":
                baseline = request.get("baseline_raw")
                baseline_description = str(request.get("baseline_description") or "Selected replay frame.")
            else:
                raise ValueError("unsupported Integrated Gradients baseline")
            explained = explain_integrated_gradients(
                self.policy.introspector.handles,
                explanation_id=explanation_id,
                input_raw=request.get("input_raw", []),
                baseline_raw=baseline,
                action_index=int(request.get("action_index", -1)),
                action_name=str(request.get("action_name") or "selected action"),
                baseline_kind=str(baseline_kind),
                baseline_description=baseline_description,
                n_steps=int(request.get("n_steps", 32)),
            )
            result = explained.to_dict()
            result.update(
                {
                    "event_id": request.get("event_id"),
                    "live_sequence": request.get("live_sequence"),
                    "checkpoint": frame.checkpoint,
                }
            )
        except Exception as exc:
            result = {
                "explanation_id": explanation_id,
                "status": "error",
                "method": "integrated_gradients",
                "action_index": request.get("action_index"),
                "action_name": request.get("action_name"),
                "baseline_kind": request.get("baseline_kind"),
                "event_id": request.get("event_id"),
                "live_sequence": request.get("live_sequence"),
                "checkpoint": frame.checkpoint,
                "error": str(exc),
            }
        self._introspection_ipc.publish_explanation(explanation_id, result)

    def _persist_capture(
        self,
        frame,
        event_type: str,
        reason: str,
        *,
        markers: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._replay_buffer is None or self._replay_recorder is None:
            return
        manager = getattr(self, "_ckpt_mgr", None)
        checkpoint = str(getattr(manager, "current_name", frame.checkpoint))
        schema = (
            self._checkpoint_schema_provider()
            if callable(self._checkpoint_schema_provider)
            else {}
        )
        try:
            self._replay_recorder.persist(
                event_type=event_type,
                frames=self._replay_buffer.snapshot(),
                trigger_sequence=frame.sequence_id,
                trigger_reason=reason,
                checkpoint=checkpoint,
                checkpoint_iteration=checkpoint_iteration(checkpoint),
                schema=schema,
                policy_generation=frame.policy_generation,
                markers=markers,
            )
        except (OSError, ValueError) as exc:
            print(f"[WARN]: Could not persist {event_type} viewer capture: {exc}")

    def close(self) -> None:
        policy = getattr(self, "policy", None)
        if isinstance(policy, _ViewerPolicy):
            policy.close()
        super().close()

    def _update_diagnostic_hud(self, *, force: bool = False) -> None:
        if self._diagnostic_html is None:
            return
        if self._diagnostic_freeze is not None and self._diagnostic_freeze.value and not force:
            return

        now = time.monotonic()
        if not force and now < self._diagnostic_next_update:
            return
        self._diagnostic_next_update = now + 0.10

        env = self.env.unwrapped
        env_idx = int(getattr(getattr(self, "_scene", None), "env_idx", 0))
        asset = env.scene["robot"]

        gravity = asset.data.projected_gravity_b[env_idx : env_idx + 1]
        tilt_rad = float(projected_gravity_tilt(gravity).item())
        tilt_deg = math.degrees(tilt_rad)
        height_m = float(asset.data.root_link_pos_w[env_idx, 2].item())
        angular_rate_rad_s = float(
            torch.linalg.vector_norm(asset.data.root_link_ang_vel_b[env_idx, :2]).item()
        )
        angular_rate_deg_s = math.degrees(angular_rate_rad_s)

        episode_step = int(env.episode_length_buf[env_idx].item())
        previous_episode_step = self._diagnostic_prev_episode_step
        reset_detected = (
            previous_episode_step is not None and episode_step < previous_episode_step
        )

        if (
            reset_detected
            or self._diagnostic_prev_tilt is None
            or previous_episode_step is None
            or episode_step <= previous_episode_step
        ):
            tilt_rate_rad_s = 0.0
            if reset_detected:
                self._diagnostic_reset_count += 1
        else:
            simulation_dt = max(
                1.0e-9,
                (episode_step - previous_episode_step) * float(env.step_dt),
            )
            tilt_rate_rad_s = (tilt_rad - self._diagnostic_prev_tilt) / simulation_dt

        self._diagnostic_prev_tilt = tilt_rad
        self._diagnostic_prev_episode_step = episode_step

        left_contact = self._wheel_contact(env, "left_wheel_contact", env_idx)
        right_contact = self._wheel_contact(env, "right_wheel_contact", env_idx)
        support_expected = self._support_expected(env, env_idx)
        fall_tilt_rad, min_height_m = self._fall_boundaries(env)

        lookahead_s = (
            float(self._diagnostic_lookahead.value)
            if self._diagnostic_lookahead is not None
            else 0.25
        )
        confidence = compute_balance_confidence(
            tilt_rad=tilt_rad,
            tilt_rate_rad_s=tilt_rate_rad_s,
            height_m=height_m,
            left_contact=left_contact,
            right_contact=right_contact,
            fall_tilt_rad=fall_tilt_rad,
            min_height_m=min_height_m,
            nominal_height_m=PHYSICS_PROFILE.default_root_height_m,
            lookahead_s=lookahead_s,
            support_expected=support_expected,
        )
        smoothing = (
            float(self._diagnostic_smoothing.value)
            if self._diagnostic_smoothing is not None
            else 0.65
        )
        self._diagnostic_smoothed_confidence = smooth_value(
            self._diagnostic_smoothed_confidence,
            confidence.confidence,
            smoothing,
        )

        reward_manager = env.reward_manager
        step_terms = getattr(reward_manager, "_step_reward", None)
        if step_terms is not None:
            term_values = step_terms[env_idx].detach().cpu().tolist()
            reward_terms = dict(zip(reward_manager.active_terms, term_values, strict=False))
        else:
            reward_terms = {
                name: float(values[0])
                for name, values in reward_manager.get_active_iterable_terms(env_idx)
            }
        reward_rate = sum(reward_terms.values())
        reward_buf = getattr(reward_manager, "_reward_buf", None)
        if reward_buf is not None:
            step_reward = float(reward_buf[env_idx].item())
        else:
            step_reward = reward_rate * float(env.step_dt)

        self._diagnostic_history["reward_rate"].append(reward_rate)
        self._diagnostic_history["tilt_deg"].append(tilt_deg)
        self._diagnostic_history["confidence"].append(
            self._diagnostic_smoothed_confidence * 100.0
        )

        fallen = False
        try:
            fallen = bool(env.termination_manager.get_term("fallen")[env_idx].item())
        except (KeyError, ValueError):
            pass

        self._diagnostic_html.content = self._render_diagnostic_html(
            reward_rate=reward_rate,
            step_reward=step_reward,
            reward_terms=reward_terms,
            tilt_deg=tilt_deg,
            tilt_rate_deg_s=math.degrees(tilt_rate_rad_s),
            angular_rate_deg_s=angular_rate_deg_s,
            height_m=height_m,
            left_contact=left_contact,
            right_contact=right_contact,
            support_expected=support_expected,
            fall_tilt_deg=math.degrees(fall_tilt_rad),
            min_height_m=min_height_m,
            predicted_tilt_deg=math.degrees(confidence.predicted_tilt_rad),
            confidence_percent=self._diagnostic_smoothed_confidence * 100.0,
            fallen=fallen,
            episode_step=episode_step,
        )

    @staticmethod
    def _wheel_contact(env, sensor_name: str, env_idx: int) -> bool:
        try:
            found = env.scene[sensor_name].data.found
        except KeyError:
            return False
        if found is None:
            return False
        return bool(found[env_idx].flatten().any().item())

    @staticmethod
    def _support_expected(env, env_idx: int) -> bool:
        """Treat intentional jump flight as unsupported-by-design, not instability."""
        state = getattr(env, "ascento_jump_state", None)
        if not isinstance(state, dict):
            return True
        phase = state.get("phase")
        if phase is None:
            return True
        return int(phase[env_idx].item()) != PHASE_FLIGHT

    @staticmethod
    def _fall_boundaries(env) -> tuple[float, float]:
        min_height_m = 0.35
        max_gravity_z = -0.5
        try:
            cfg = env.termination_manager.get_term_cfg("fallen")
            params = cfg.params or {}
            min_height_m = float(params.get("min_height", min_height_m))
            max_gravity_z = float(params.get("max_gravity_z", max_gravity_z))
        except ValueError:
            pass
        cosine_threshold = max(-1.0, min(1.0, -max_gravity_z))
        return math.acos(cosine_threshold), min_height_m

    def _render_diagnostic_html(
        self,
        *,
        reward_rate: float,
        step_reward: float,
        reward_terms: dict[str, float],
        tilt_deg: float,
        tilt_rate_deg_s: float,
        angular_rate_deg_s: float,
        height_m: float,
        left_contact: bool,
        right_contact: bool,
        support_expected: bool,
        fall_tilt_deg: float,
        min_height_m: float,
        predicted_tilt_deg: float,
        confidence_percent: float,
        fallen: bool,
        episode_step: int,
    ) -> str:
        history_seconds = (
            int(self._diagnostic_history_seconds.value)
            if self._diagnostic_history_seconds is not None
            else 12
        )
        history_points = max(1, min(600, history_seconds * 10))
        reward_history = list(self._diagnostic_history["reward_rate"])[-history_points:]
        tilt_history = list(self._diagnostic_history["tilt_deg"])[-history_points:]
        confidence_history = list(self._diagnostic_history["confidence"])[-history_points:]

        confidence_color = (
            "#22c55e"
            if confidence_percent >= 70.0
            else "#f59e0b"
            if confidence_percent >= 35.0
            else "#ef4444"
        )
        support = (
            "airborne (expected)"
            if not support_expected
            else "both"
            if left_contact and right_contact
            else "left only"
            if left_contact
            else "right only"
            if right_contact
            else "none"
        )
        fall_state = (
            '<span style="color:#ef4444;font-weight:700;">FALLEN / RESET</span>'
            if fallen
            else '<span style="color:#22c55e;">running</span>'
        )

        breakdown = ""
        if (
            self._diagnostic_reward_breakdown is not None
            and self._diagnostic_reward_breakdown.value
        ):
            top_n = (
                int(self._diagnostic_top_terms.value)
                if self._diagnostic_top_terms is not None
                else 6
            )
            terms = sorted(
                reward_terms.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:top_n]
            rows = "".join(
                (
                    "<tr>"
                    f"<td style='padding-right:0.8em'>{html.escape(name)}</td>"
                    f"<td style='text-align:right;font-family:monospace'>{value:+.4f}/s</td>"
                    "</tr>"
                )
                for name, value in terms
            )
            breakdown = (
                "<div style='margin-top:0.7em'>"
                "<strong>Largest reward contributions</strong>"
                "<table style='width:100%;font-size:0.82em;margin-top:0.25em'>"
                f"{rows}</table></div>"
            )

        return f"""
        <div style="font-size:0.86em;line-height:1.25;padding:0.2em 0.45em 0.5em 0.45em">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.45em">
            <div style="padding:0.45em;border:1px solid rgba(127,127,127,.25);border-radius:6px">
              <div style="opacity:.65">Reward rate</div>
              <div style="font-size:1.35em;font-weight:700">{reward_rate:+.4f}/s</div>
              <div style="opacity:.7">actual step: {step_reward:+.6f}</div>
            </div>
            <div style="padding:0.45em;border:1px solid rgba(127,127,127,.25);border-radius:6px">
              <div style="opacity:.65">Tilt / fall boundary</div>
              <div style="font-size:1.35em;font-weight:700">
                {tilt_deg:.2f}° / {fall_tilt_deg:.1f}°
              </div>
              <div style="opacity:.7">predicted: {predicted_tilt_deg:.2f}°</div>
            </div>
            <div style="padding:0.45em;border:1px solid rgba(127,127,127,.25);border-radius:6px">
              <div style="opacity:.65">Stability margin*</div>
              <div style="font-size:1.35em;font-weight:700;color:{confidence_color}">
                {confidence_percent:.1f}%
              </div>
              <div style="height:6px;background:rgba(127,127,127,.2);border-radius:4px">
                <div style="height:6px;width:{max(0.0, min(100.0, confidence_percent)):.1f}%;
                  background:{confidence_color};border-radius:4px"></div>
              </div>
            </div>
            <div style="padding:0.45em;border:1px solid rgba(127,127,127,.25);border-radius:6px">
              <div style="opacity:.65">Motion / support</div>
              <div><strong>tilt rate:</strong> {tilt_rate_deg_s:+.1f}°/s</div>
              <div><strong>roll/pitch ω:</strong> {angular_rate_deg_s:.1f}°/s</div>
              <div><strong>wheels:</strong> {support}</div>
            </div>
          </div>

          <div style="margin-top:0.65em;font-family:monospace;font-size:0.78em;
            white-space:nowrap;overflow:hidden">
            reward&nbsp; {sparkline(reward_history)}<br/>
            tilt&nbsp;&nbsp;&nbsp;
            {sparkline(tilt_history, minimum=0.0, maximum=fall_tilt_deg)}<br/>
            margin&nbsp;&nbsp; {sparkline(confidence_history, minimum=0.0, maximum=100.0)}
          </div>

          <div style="margin-top:0.55em">
            <strong>height:</strong> {height_m:.3f} m
            (fall &lt; {min_height_m:.2f} m) ·
            <strong>episode step:</strong> {episode_step} ·
            <strong>observed resets:</strong> {self._diagnostic_reset_count} ·
            {fall_state}
          </div>
          {breakdown}
          <div style="margin-top:0.65em;opacity:.62;font-size:0.78em">
            *Diagnostic state-derived margin, not a probability or confidence emitted by the policy.
            Reward terms are weighted reward rates before dt scaling.
          </div>
        </div>
        """


class ViewerTransitionObserver:
    """Viewer-local tap pairing policy actions with the exact environment transition."""

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        self.env = env
        self.latest: TransitionSnapshot | None = None
        self._terminal_state: dict[str, Any] | None = None
        self._episode_id = 0
        self._viewer_step_count = 0
        self._previous_tilt: float | None = None
        self._previous_episode_id: int | None = None
        self._termination_manager = env.termination_manager
        self._original_compute = self._termination_manager.compute
        self._compute_wrapper = self._observe_termination
        self._termination_manager.compute = self._compute_wrapper
        self._reward_manager = env.reward_manager
        self._original_reward_compute = self._reward_manager.compute
        self._reward_compute_wrapper = self._observe_rewards
        self._captured_reward_terms: dict[str, float] = {}
        self._captured_step_reward: float | None = None
        self._reward_manager.compute = self._reward_compute_wrapper
        self._original_step = env.step
        self._step_wrapper = self._observe_step
        env.step = self._step_wrapper

    def close(self) -> None:
        if getattr(self.env, "step", None) is self._step_wrapper:
            del self.env.step
        if getattr(self._termination_manager, "compute", None) is self._compute_wrapper:
            del self._termination_manager.compute
        if getattr(self._reward_manager, "compute", None) is self._reward_compute_wrapper:
            del self._reward_manager.compute

    def _observe_termination(self, *args, **kwargs):
        result = self._original_compute(*args, **kwargs)
        self._terminal_state = self._capture_state()
        return result

    def _observe_rewards(self, *args, **kwargs):
        result = self._original_reward_compute(*args, **kwargs)
        try:
            self._captured_step_reward = float(result[0].detach().cpu().item())
        except (AttributeError, IndexError, TypeError):
            self._captured_step_reward = None
        manager = self._reward_manager
        values = getattr(manager, "_step_reward", None)
        names = getattr(manager, "active_terms", ())
        self._captured_reward_terms = {}
        if isinstance(values, torch.Tensor) and values.ndim >= 2:
            self._captured_reward_terms = {
                str(name): float(value)
                for name, value in zip(names, values[0].detach().cpu().tolist(), strict=False)
            }
        return result

    def _observe_step(self, actions: torch.Tensor):
        self._terminal_state = None
        self._viewer_step_count += 1
        result = self._original_step(actions)
        state = self._terminal_state or self._capture_state()
        self.latest = self._build_transition(state, result)
        if self.latest.reset:
            self._episode_id += 1
            self._previous_tilt = None
            self._previous_episode_id = None
        else:
            self._previous_tilt = float(state["tilt_rad"])
            self._previous_episode_id = self._episode_id
        return result

    def _capture_state(self) -> dict[str, Any]:
        env = self.env
        index = 0
        asset = env.scene["robot"]
        gravity = asset.data.projected_gravity_b[index : index + 1]
        tilt_rad = float(projected_gravity_tilt(gravity).item())
        height_m = float(asset.data.root_link_pos_w[index, 2].item())
        angular_rate = float(
            torch.linalg.vector_norm(asset.data.root_link_ang_vel_b[index, :2]).item()
        )
        contacts = {
            "left": _read_contact(env, "left_wheel_contact", index),
            "right": _read_contact(env, "right_wheel_contact", index),
        }
        fallen = False
        try:
            term = env.termination_manager.get_term("fallen")
            fallen = bool(term[index].item())
        except (KeyError, ValueError, AttributeError, IndexError):
            pass
        episode_step = int(env.episode_length_buf[index].item())
        simulation_step = self._viewer_step_count
        fall_boundary, min_height = _runtime_fall_boundaries(env)
        if (
            self._previous_tilt is None
            or self._previous_episode_id != self._episode_id
            or episode_step <= 1
        ):
            tilt_rate = 0.0
        else:
            tilt_rate = (tilt_rad - self._previous_tilt) / max(float(env.step_dt), 1.0e-9)
        state = {
            "episode_step": episode_step,
            "simulation_step": simulation_step,
            "tilt_rad": tilt_rad,
            "tilt_rate_rad_s": tilt_rate,
            "height_m": height_m,
            "angular_rate_rad_s": angular_rate,
            "contacts": contacts,
            "fallen": fallen,
            "fall_boundary_rad": fall_boundary,
            "min_height_m": min_height,
            "support_expected": _runtime_support_expected(env, index),
        }
        return state

    def _build_transition(self, state: dict[str, Any], result: Any) -> TransitionSnapshot:
        env = self.env
        index = 0
        reward_terms = dict(self._captured_reward_terms)
        if not reward_terms:
            try:
                reward_terms = {
                    str(name): float(values[0])
                    for name, values in env.reward_manager.get_active_iterable_terms(index)
                }
            except (AttributeError, TypeError, ValueError, IndexError):
                reward_terms = {}
        step_reward = self._captured_step_reward
        if step_reward is None:
            try:
                step_reward = float(result[1][index].item())
            except (TypeError, IndexError, AttributeError):
                step_reward = 0.0
        terminated = _tensor_flag(result[2] if len(result) > 2 else False, index)
        timeout = _tensor_flag(result[3] if len(result) > 3 else False, index)
        reset = terminated or timeout
        fall_tilt = float(state["fall_boundary_rad"])
        margin = compute_balance_confidence(
            tilt_rad=float(state["tilt_rad"]),
            tilt_rate_rad_s=float(state["tilt_rate_rad_s"]),
            height_m=float(state["height_m"]),
            left_contact=bool(state["contacts"]["left"]),
            right_contact=bool(state["contacts"]["right"]),
            fall_tilt_rad=fall_tilt,
            min_height_m=float(state["min_height_m"]),
            nominal_height_m=PHYSICS_PROFILE.default_root_height_m,
            lookahead_s=0.25,
            support_expected=bool(state["support_expected"]),
        ).confidence
        return TransitionSnapshot(
            episode_id=self._episode_id,
            episode_step=int(state["episode_step"]),
            sim_time_s=int(state["simulation_step"]) * float(env.step_dt),
            reward_rate=sum(reward_terms.values()),
            step_reward=step_reward,
            reward_terms=reward_terms,
            tilt_rad=float(state["tilt_rad"]),
            tilt_rate_rad_s=float(state["tilt_rate_rad_s"]),
            height_m=float(state["height_m"]),
            contacts=dict(state["contacts"]),
            balance_margin=float(margin),
            fallen=bool(state["fallen"]),
            reset=reset,
        )


def _read_contact(env: Any, sensor_name: str, index: int) -> bool:
    try:
        found = env.scene[sensor_name].data.found
    except (KeyError, AttributeError, TypeError):
        return False
    return bool(found is not None and found[index].flatten().any().item())


def _runtime_support_expected(env: Any, index: int) -> bool:
    state = getattr(env, "ascento_jump_state", None)
    phase = state.get("phase") if isinstance(state, dict) else None
    return phase is None or int(phase[index].item()) != PHASE_FLIGHT


def _runtime_fall_boundaries(env: Any) -> tuple[float, float]:
    min_height = 0.35
    max_gravity_z = -0.5
    try:
        params = env.termination_manager.get_term_cfg("fallen").params or {}
        min_height = float(params.get("min_height", min_height))
        max_gravity_z = float(params.get("max_gravity_z", max_gravity_z))
    except (AttributeError, ValueError, TypeError):
        pass
    return math.acos(max(-1.0, min(1.0, -max_gravity_z))), min_height


def _tensor_flag(value: Any, index: int) -> bool:
    try:
        if isinstance(value, torch.Tensor):
            return bool(value.reshape(-1)[index].item())
        return bool(value)
    except (IndexError, TypeError, RuntimeError):
        return False


def _runtime_payload(
    checkpoint: CheckpointInfo,
    *,
    task: str,
    follow: bool,
    port: int,
) -> dict[str, Any]:
    return {
        "state": "running",
        "task": task,
        "checkpoint": checkpoint.relative_path,
        "checkpoint_name": checkpoint.name,
        "checkpoint_iteration": checkpoint.iteration,
        "checkpoint_size_bytes": checkpoint.size_bytes,
        "loaded_at": time.time(),
        "follow": follow,
        "port": port,
        "pid": os.getpid(),
    }


def run_viewer(
    *,
    task: str,
    run_dir: Path,
    checkpoint: str,
    host: str,
    port: int,
    device: str,
    status_file: Path | None = None,
    follow: bool = False,
    stable_age_seconds: float = 2.0,
    follow_poll_seconds: float = 2.0,
    jacobian_hz: float = 2.0,
    introspection_dir: Path | None = None,
    viewer_id: str = "standalone",
    run_id: str | None = None,
    capture_dir: Path | None = None,
) -> None:
    """Load one policy environment and serve the interactive Viser viewer."""
    configure_torch_backends()
    run_dir = run_dir.expanduser().resolve()

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    canonical_env_cfg = load_env_cfg(task, play=False)
    agent_cfg = load_rl_cfg(task)

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    structured_action_term = next(
        (
            base_env.action_manager.get_term(name)
            for name in base_env.action_manager.active_terms
            if isinstance(base_env.action_manager.get_term(name), StructuredTargetAction)
        ),
        None,
    )
    if structured_action_term is None:
        env.close()
        raise ValueError("selected task has no StructuredTargetAction term to inspect")
    action_observer = StructuredActionObserver(structured_action_term)
    transition_observer = ViewerTransitionObserver(base_env)
    replay_buffer = PolicyReplayBuffer(step_dt=float(base_env.step_dt), capture_seconds=10.0)
    event_detector = EventTriggerDetector()
    introspection_ipc = (
        IntrospectionIPC(introspection_dir) if introspection_dir is not None else None
    )
    replay_recorder = (
        PolicyReplayRecorder(
            capture_dir or (introspection_dir / "captures"),
            viewer_id=viewer_id,
            run_id=run_id or run_dir.name,
        )
        if capture_dir is not None or introspection_dir is not None
        else None
    )
    sequence_counter = itertools.count(1)
    active_schema: dict[str, Any] = {}
    next_sequence = sequence_counter.__next__
    current_policy: _ViewerPolicy | None = None
    policy_generation = 0

    def load_policy(selection: str) -> _ViewerPolicy:
        nonlocal current_policy, policy_generation
        info = resolve_run_checkpoint(
            run_dir,
            selection,
            stable_age_seconds=stable_age_seconds,
        )
        checkpoint_path = run_dir / info.relative_path
        candidate_generation = policy_generation + 1
        actor = runner.alg.get_policy()
        previous_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
        previous_modes = tuple((module, module.training) for module in actor.modules())
        candidate_introspector: PolicyIntrospector | None = None
        try:
            _load_actor_transactionally(
                runner,
                checkpoint_path,
                canonical_env_cfg,
                device=device,
            )
            adapter = RslRlPolicyAdapter(runner, checkpoint_path, deterministic=True)
            handles = resolve_runtime_handles(runner)
            actor_schema = build_observation_schema(
                base_env.observation_manager,
                group=handles.actor_obs_groups,
            )
            if actor_schema.input_dim != handles.actor_input_dim:
                raise ValueError(
                    "runtime actor observation schema width does not match model input: "
                    f"{actor_schema.input_dim} != {handles.actor_input_dim}"
                )
            critic_schema = None
            if handles.critic is not None:
                critic_schema = build_observation_schema(
                    base_env.observation_manager,
                    group=handles.critic_obs_groups,
                )
                if critic_schema.input_dim != handles.critic_input_dim:
                    raise ValueError(
                        "runtime critic observation schema width does not match model input: "
                        f"{critic_schema.input_dim} != {handles.critic_input_dim}"
                    )
            actor_network = build_policy_branch_schema(
                handles.actor,
                handles.actor_body,
                distribution=handles.distribution,
            )
            critic_network = (
                build_policy_branch_schema(handles.critic, handles.critic_body)
                if handles.critic is not None and handles.critic_body is not None
                else None
            )
            candidate_introspector = PolicyIntrospector(
                handles,
                checkpoint=info.relative_path,
                jacobian_hz=jacobian_hz,
                policy_generation=candidate_generation,
            )
            policy = _ViewerPolicy(
                adapter,
                candidate_introspector,
                action_observer,
                next_sequence,
                wrapper_clip=agent_cfg.clip_actions,
            )
            next_schema = {
                "schema_version": 1,
                "checkpoint": info.relative_path,
                "policy_generation": candidate_generation,
                "actor": actor_schema.to_dict(),
                "critic": critic_schema.to_dict() if critic_schema is not None else None,
                "actor_network": actor_network,
                "critic_network": critic_network,
            }
            if introspection_ipc is not None:
                introspection_ipc.publish_schema(
                    actor_schema,
                    critic_schema,
                    checkpoint=info.relative_path,
                    policy_generation=candidate_generation,
                    actor_network=actor_network,
                    critic_network=critic_network,
                )
        except Exception:
            if candidate_introspector is not None:
                candidate_introspector.close()
            runner.alg.get_policy().load_state_dict(previous_state, strict=True)
            for module, training in previous_modes:
                module.training = training
            raise
        if current_policy is not None:
            replay_buffer.clear()
            event_detector.reset()
            current_policy.close()
        active_schema.clear()
        active_schema.update(next_schema)
        policy_generation = candidate_generation
        current_policy = policy
        _write_json(
            status_file,
            _runtime_payload(info, task=task, follow=follow, port=port),
        )
        return policy

    initial_info = resolve_run_checkpoint(
        run_dir,
        checkpoint,
        stable_age_seconds=stable_age_seconds,
    )
    policy = load_policy(initial_info.relative_path)

    def fetch_available() -> list[tuple[str, str]]:
        now = time.time()
        return [
            (
                info.relative_path,
                format_time_ago(max(0, int(now - info.modified_at))),
            )
            for info in discover_checkpoints(
                run_dir,
                stable_age_seconds=stable_age_seconds,
                now=now,
            )
        ]

    checkpoint_manager = CheckpointManager(
        current_name=initial_info.relative_path,
        fetch_available=fetch_available,
        load_checkpoint=load_policy,
    )
    server = viser.ViserServer(host=host, port=port, label="Ascento Policy Viewer")
    viewer = _FollowViserPlayViewer(
        env,
        policy,
        viser_server=server,
        checkpoint_manager=checkpoint_manager,
        follow=follow,
        follow_poll_seconds=follow_poll_seconds,
        introspection_ipc=introspection_ipc,
        transition_observer=transition_observer,
        replay_buffer=replay_buffer,
        replay_recorder=replay_recorder,
        checkpoint_schema_provider=lambda: dict(active_schema),
        event_detector=event_detector,
    )

    try:
        viewer.run()
    finally:
        if current_policy is not None:
            current_policy.close()
        transition_observer.close()
        action_observer.close()
        try:
            env.close()
        finally:
            server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--stable-age-seconds", type=float, default=2.0)
    parser.add_argument("--follow-poll-seconds", type=float, default=2.0)
    parser.add_argument("--jacobian-hz", type=float, choices=(0, 1, 2, 5, 10), default=2.0)
    parser.add_argument("--introspection-dir", type=Path)
    parser.add_argument("--viewer-id", default="standalone")
    parser.add_argument("--run-id")
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    run_viewer(
        task=args.task,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        host=args.host,
        port=args.port,
        device=args.device,
        status_file=args.status_file,
        follow=args.follow,
        stable_age_seconds=args.stable_age_seconds,
        follow_poll_seconds=args.follow_poll_seconds,
        jacobian_hz=args.jacobian_hz,
        introspection_dir=args.introspection_dir,
        viewer_id=args.viewer_id,
        run_id=args.run_id,
        capture_dir=args.capture_dir,
    )


if __name__ == "__main__":
    main()
