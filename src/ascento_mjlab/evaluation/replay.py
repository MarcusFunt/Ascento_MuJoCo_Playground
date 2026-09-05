"""Replay one exact stored evaluation scenario in a native or Viser viewer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import ascento_mjlab.tasks  # noqa: F401

from .policy import RslRlPolicyAdapter
from .runner import (
    _apply_commands,
    _disturbance_wrench,
    _exact_reset,
    _robot_total_mass,
)
from .schema import CommandPoint, DisturbanceSpec, ScenarioSpec


def _scenario_from_dict(raw: dict) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=str(raw["scenario_id"]),
        family=str(raw["family"]),
        task=str(raw["task"]),
        horizon_steps=int(raw["horizon_steps"]),
        reset={str(k): float(v) for k, v in raw["reset"].items()},
        disturbances=tuple(
            DisturbanceSpec(
                start_step=int(item["start_step"]),
                duration_steps=int(item["duration_steps"]),
                direction=str(item["direction"]),
                equivalent_delta_v=float(item.get("equivalent_delta_v", 0.0)),
                force_n=None if item.get("force_n") is None else float(item["force_n"]),
                torque_nm=tuple(float(x) for x in item.get("torque_nm", (0.0, 0.0, 0.0))),
            )
            for item in raw.get("disturbances", [])
        ),
        commands=tuple(
            CommandPoint(
                step=int(item["step"]),
                name=str(item["name"]),
                values=tuple(float(x) for x in item["values"]),
            )
            for item in raw.get("commands", [])
        ),
        tags=tuple(str(x) for x in raw.get("tags", [])),
    )


def _load_scenario(eval_dir: Path, scenario_id: str) -> ScenarioSpec:
    with (eval_dir / "resolved_scenarios.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if raw["scenario_id"] == scenario_id:
                return _scenario_from_dict(raw)
    raise KeyError(f"scenario not found: {scenario_id}")


class _ReplayPolicy:
    def __init__(self, env, base_env, policy, scenario: ScenarioSpec):
        self.env = env
        self.base_env = base_env
        self.policy = policy
        self.scenario = scenario
        self.step = 0
        self.mass = _robot_total_mass(base_env)
        self.step_dt = base_env.step_dt

    def __call__(self, _observations):
        _apply_commands(self.base_env, [self.scenario], self.step)
        forces, torques, _ = _disturbance_wrench(
            [self.scenario],
            self.step,
            mass=self.mass,
            step_dt=self.step_dt,
            device=self.base_env.device,
        )
        self.base_env.scene["robot"].write_external_wrench_to_sim(forces, torques, body_ids=[0])
        observations = self.env.get_observations()
        action = self.policy.act(observations)
        self.step += 1
        return action


def replay(
    *,
    eval_dir: Path,
    scenario_id: str,
    checkpoint: Path,
    viewer: str,
    device: str,
) -> None:
    scenario = _load_scenario(eval_dir, scenario_id)
    cfg = load_env_cfg(scenario.task, play=True)
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.episode_length_s = (scenario.horizon_steps + 2) * (
        float(cfg.sim.timestep) * int(cfg.decimation)
    )
    base_env = ManagerBasedRlEnv(cfg, device=device, render_mode=None)
    agent_cfg = load_rl_cfg(scenario.task)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(scenario.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = RslRlPolicyAdapter(runner, checkpoint, deterministic=True)

    env.reset()
    _exact_reset(base_env, [scenario])
    policy.reset()
    replay_policy = _ReplayPolicy(env, base_env, policy, scenario)
    print(f"[INFO] Exact scenario: {scenario.scenario_id}")
    print(
        f"[INFO] Horizon: {scenario.horizon_steps} steps ({scenario.horizon_steps * base_env.step_dt:.3f}s)"
    )
    print(
        "[INFO] Manual viewer reset is intentionally not scenario-aware; restart this command for an exact replay."
    )

    try:
        if viewer == "native":
            NativeMujocoViewer(env, replay_policy).run()
        elif viewer == "viser":
            ViserPlayViewer(env, replay_policy).run()
        else:
            raise ValueError(f"unsupported viewer: {viewer}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--viewer", choices=["native", "viser"], default="native")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest = json.loads((args.evaluation / "manifest.json").read_text())
    checkpoint = args.checkpoint or Path(manifest["checkpoint"])
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    replay(
        eval_dir=args.evaluation,
        scenario_id=args.scenario,
        checkpoint=checkpoint,
        viewer=args.viewer,
        device=args.device,
    )


if __name__ == "__main__":
    main()
