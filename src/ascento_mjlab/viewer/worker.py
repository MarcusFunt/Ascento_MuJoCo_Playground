"""Run an isolated one-environment Ascento policy in mjlab's Viser browser viewer."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict
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
from ascento_mjlab.physics import PHYSICS_PROFILE

from .checkpoints import CheckpointInfo, discover_checkpoints, resolve_run_checkpoint
from .diagnostics import compute_balance_confidence, smooth_value, sparkline


def _write_json(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


class _ViewerPolicy:
    """Adapt the evaluation policy interface to mjlab's callable viewer protocol."""

    def __init__(self, adapter: RslRlPolicyAdapter):
        self.adapter = adapter

    def __call__(self, observations):
        return self.adapter.act(observations)

    def reset(self) -> None:
        self.adapter.reset()


class _FollowViserPlayViewer(ViserPlayViewer):
    """Viser viewer with checkpoint following and an always-visible diagnostic HUD."""

    def __init__(
        self,
        *args,
        follow: bool = False,
        follow_poll_seconds: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._follow = bool(follow)
        self._follow_poll_seconds = max(0.5, float(follow_poll_seconds))
        self._next_follow_poll = 0.0

        self._diagnostic_html = None
        self._diagnostic_freeze = None
        self._diagnostic_reward_breakdown = None
        self._diagnostic_lookahead = None
        self._diagnostic_smoothing = None
        self._diagnostic_history_seconds = None
        self._diagnostic_top_terms = None
        self._diagnostic_next_update = 0.0
        self._diagnostic_prev_tilt: float | None = None
        self._diagnostic_prev_sample_at: float | None = None
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
                self._diagnostic_prev_sample_at = None
                self._diagnostic_smoothed_confidence = None
                self._diagnostic_reset_count = 0
                self._diagnostic_next_update = 0.0

    def _process_actions(self) -> None:
        now = time.monotonic()
        manager = getattr(self, "_ckpt_mgr", None)
        if self._follow and manager is not None and now >= self._next_follow_poll:
            self._next_follow_poll = now + self._follow_poll_seconds
            entries = manager.fetch_available()
            if entries and entries[-1][0] != manager.current_name:
                self._actions.append((ViewerAction.FETCH_CHECKPOINT, "latest"))
        super()._process_actions()

    def sync_env_to_viewer(self) -> None:
        super().sync_env_to_viewer()
        self._update_diagnostic_hud()

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
        reset_detected = (
            self._diagnostic_prev_episode_step is not None
            and episode_step < self._diagnostic_prev_episode_step
        )
        self._diagnostic_prev_episode_step = episode_step

        if (
            reset_detected
            or self._diagnostic_prev_tilt is None
            or self._diagnostic_prev_sample_at is None
        ):
            tilt_rate_rad_s = 0.0
            if reset_detected:
                self._diagnostic_reset_count += 1
        else:
            dt = max(1.0e-6, now - self._diagnostic_prev_sample_at)
            tilt_rate_rad_s = (tilt_rad - self._diagnostic_prev_tilt) / dt

        self._diagnostic_prev_tilt = tilt_rad
        self._diagnostic_prev_sample_at = now

        left_contact = self._wheel_contact(env, "left_wheel_contact", env_idx)
        right_contact = self._wheel_contact(env, "right_wheel_contact", env_idx)
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

        reward_terms = {
            name: float(values[0])
            for name, values in env.reward_manager.get_active_iterable_terms(env_idx)
        }
        reward_rate = sum(reward_terms.values())
        reward_buf = getattr(env.reward_manager, "_reward_buf", None)
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
            "both"
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
              <div style="font-size:1.35em;font-weight:700">{tilt_deg:.2f}° / {fall_tilt_deg:.1f}°</div>
              <div style="opacity:.7">predicted: {predicted_tilt_deg:.2f}°</div>
            </div>
            <div style="padding:0.45em;border:1px solid rgba(127,127,127,.25);border-radius:6px">
              <div style="opacity:.65">Balance confidence*</div>
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

          <div style="margin-top:0.65em;font-family:monospace;font-size:0.78em;white-space:nowrap;overflow:hidden">
            reward&nbsp; {sparkline(reward_history)}<br/>
            tilt&nbsp;&nbsp;&nbsp; {sparkline(tilt_history, minimum=0.0, maximum=fall_tilt_deg)}<br/>
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

    def load_policy(selection: str) -> _ViewerPolicy:
        info = resolve_run_checkpoint(
            run_dir,
            selection,
            stable_age_seconds=stable_age_seconds,
        )
        checkpoint_path = run_dir / info.relative_path
        infos = runner.load(
            str(checkpoint_path),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        require_current_checkpoint_contracts(infos, canonical_env_cfg)
        policy = _ViewerPolicy(
            RslRlPolicyAdapter(runner, checkpoint_path, deterministic=True)
        )
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
    )

    try:
        viewer.run()
    finally:
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
    )


if __name__ == "__main__":
    main()
