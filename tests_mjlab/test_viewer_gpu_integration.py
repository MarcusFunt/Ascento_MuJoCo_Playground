import socket
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.viser.viewer import CheckpointManager

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.evaluation.policy import RslRlPolicyAdapter
from ascento_mjlab.viewer.checkpoints import discover_checkpoints
from ascento_mjlab.viewer.worker import (
    _FollowViserPlayViewer,
    _ViewerPolicy,
    _load_actor_transactionally,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA/MuJoCo-Warp GPU runtime",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _save(runner, path: Path, iteration: int) -> None:
    runner.current_learning_iteration = iteration
    runner.save(str(path))


def _viewer_stack(tmp_path: Path):
    task = "Ascento-Balance-Flat"
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    canonical_cfg = load_env_cfg(task, play=False)
    agent_cfg = load_rl_cfg(task)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device="cuda:0")

    def load_policy(name: str):
        path = tmp_path / name
        _load_actor_transactionally(runner, path, canonical_cfg, device="cuda:0")
        return _ViewerPolicy(RslRlPolicyAdapter(runner, path, deterministic=True))

    def fetch_available():
        return [
            (info.relative_path, "")
            for info in discover_checkpoints(tmp_path, stable_age_seconds=0)
        ]

    return env, runner, load_policy, fetch_available


def _run_switch_case(tmp_path: Path, *, follow: bool) -> tuple[str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    env, runner, load_policy, fetch_available = _viewer_stack(tmp_path)
    first = tmp_path / "model_100.pt"
    _save(runner, first, 100)
    policy = load_policy(first.name)
    manager = CheckpointManager(
        current_name=first.name,
        fetch_available=fetch_available,
        load_checkpoint=load_policy,
    )
    server = viser.ViserServer(
        host="127.0.0.1",
        port=_free_port(),
        label="Ascento GPU viewer smoke",
    )
    viewer = _FollowViserPlayViewer(
        env,
        policy,
        viser_server=server,
        checkpoint_manager=manager,
        follow=follow,
        follow_poll_seconds=0.5,
    )
    try:
        viewer.setup()
        before = manager.current_name
        next_path = tmp_path / ("model_300.pt" if follow else "model_200.pt")
        _save(runner, next_path, 300 if follow else 200)
        if follow:
            viewer._next_follow_poll = 0.0
            viewer._process_actions()
        else:
            viewer._handle_custom_action(ViewerAction.FETCH_CHECKPOINT, "latest")
        return before, manager.current_name
    finally:
        viewer.close()
        env.close()
        server.stop()


def test_gpu_manual_checkpoint_switch_uses_two_compatible_checkpoints(tmp_path):
    before, after = _run_switch_case(tmp_path / "manual", follow=False)
    assert before == "model_100.pt"
    assert after == "model_200.pt"


def test_gpu_follow_adopts_newly_published_checkpoint(tmp_path):
    before, after = _run_switch_case(tmp_path / "follow", follow=True)
    assert before == "model_100.pt"
    assert after == "model_300.pt"
