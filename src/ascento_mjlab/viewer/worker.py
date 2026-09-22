"""Run an isolated one-environment Ascento policy in mjlab's Viser browser viewer."""

from __future__ import annotations

import argparse
import json
import os
import time
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

from .checkpoints import CheckpointInfo, discover_checkpoints, resolve_run_checkpoint


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
    """Queue checkpoint reloads on the viewer thread instead of a watcher thread."""

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

    def _process_actions(self) -> None:
        now = time.monotonic()
        manager = getattr(self, "_ckpt_mgr", None)
        if self._follow and manager is not None and now >= self._next_follow_poll:
            self._next_follow_poll = now + self._follow_poll_seconds
            entries = manager.fetch_available()
            if entries and entries[-1][0] != manager.current_name:
                self._actions.append((ViewerAction.FETCH_CHECKPOINT, "latest"))
        super()._process_actions()


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
