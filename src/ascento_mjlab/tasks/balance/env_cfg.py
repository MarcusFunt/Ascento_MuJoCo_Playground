"""Flat-ground balance environment configuration."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.robot_cfg import (
  DEFAULT_ASCENTO_CFG,
  JOINT_NAMES,
  SIM_CFG,
  VIEWER_CONFIG,
)

ROBOT_CFG = SceneEntityCfg(
  "robot",
  joint_names=JOINT_NAMES,
  actuator_names=JOINT_NAMES,
)


def _scene(num_envs: int) -> SceneCfg:
  return SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    num_envs=num_envs,
    env_spacing=2.0,
    extent=2.0,
    entities={"robot": DEFAULT_ASCENTO_CFG},
    sensors=(
      ContactSensorCfg(
        name="left_wheel_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^left_wheel$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        track_air_time=True,
        history_length=5,
      ),
      ContactSensorCfg(
        name="right_wheel_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^right_wheel$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        track_air_time=True,
        history_length=5,
      ),
    ),
  )


def _observations(play: bool) -> dict[str, ObservationGroupCfg]:
  actor_terms = {
    "gravity": ObservationTermCfg(func=mdp.projected_gravity, params={"asset_cfg": ROBOT_CFG}),
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, params={"asset_cfg": ROBOT_CFG}),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, params={"asset_cfg": ROBOT_CFG}),
    "height": ObservationTermCfg(
      func=ascento_mdp.observations.base_height, params={"asset_cfg": ROBOT_CFG}, scale=1.0
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": ROBOT_CFG}),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": ROBOT_CFG}),
    "contacts": ObservationTermCfg(
      func=ascento_mdp.observations.wheel_contacts,
      params={"left_sensor_name": "left_wheel_contact", "right_sensor_name": "right_wheel_contact"},
    ),
    "contact_forces": ObservationTermCfg(
      func=ascento_mdp.observations.wheel_contact_forces,
      params={"left_sensor_name": "left_wheel_contact", "right_sensor_name": "right_wheel_contact"},
      clip=(0.0, 2000.0),
      scale=0.01,
    ),
    "effort": ObservationTermCfg(func=ascento_mdp.observations.actuator_effort, params={"asset_cfg": ROBOT_CFG}, clip=(-40.0, 40.0), scale=0.025),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  critic_terms = {
    **actor_terms,
    "root_pos": ObservationTermCfg(func=lambda env: env.scene["robot"].data.root_link_pos_w),
    "root_lin_vel_w": ObservationTermCfg(func=lambda env: env.scene["robot"].data.root_link_lin_vel_w),
    "root_ang_vel_w": ObservationTermCfg(func=lambda env: env.scene["robot"].data.root_link_ang_vel_w),
  }
  return {
    "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True, enable_corruption=False),
    "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
  }


def _actions() -> dict[str, ActionTermCfg]:
  return {
    "effort": JointEffortActionCfg(
      entity_name="robot",
      actuator_names=JOINT_NAMES,
      scale=40.0,
      clip={".*": (-1.0, 1.0)},
      preserve_order=True,
    )
  }


def ascento_balance_env_cfg(play: bool = False, num_envs: int = 512) -> ManagerBasedRlEnvCfg:
  """Build the validated flat-ground balance configuration.

  ``num_envs`` defaults to a conservative RTX 3060 starting point.  The
  environment itself has no terrain generator; terrain is intentionally kept
  behind the successful flat-ground jump gate.
  """
  cfg = ManagerBasedRlEnvCfg(
    decimation=5,
    scene=_scene(1 if play else num_envs),
    sim=deepcopy(SIM_CFG),
    viewer=deepcopy(VIEWER_CONFIG),
    observations=_observations(play),
    actions=_actions(),
    events={
      "reset_scene_to_default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
      "reset_supported_pose": EventTermCfg(
        func=ascento_mdp.events.reset_root_state_uniform,
        mode="reset",
        params={
          "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0), "roll": (-0.08, 0.08), "pitch": (-0.08, 0.08), "yaw": (-3.14159, 3.14159)},
          "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
          "asset_cfg": ROBOT_CFG,
        },
      ),
    },
    rewards={
      "alive": RewardTermCfg(func=ascento_mdp.rewards.alive, weight=1.0),
      "upright": RewardTermCfg(func=ascento_mdp.rewards.upright, weight=2.0, params={"std": 0.35, "asset_cfg": ROBOT_CFG}),
      "height": RewardTermCfg(func=ascento_mdp.rewards.height_tracking, weight=1.0, params={"target": 0.75, "std": 0.08, "asset_cfg": ROBOT_CFG}),
      "angular_rate": RewardTermCfg(func=ascento_mdp.rewards.angular_rate_penalty, weight=-0.04, params={"asset_cfg": ROBOT_CFG}),
      "effort": RewardTermCfg(func=ascento_mdp.rewards.effort_penalty, weight=-0.0005, params={"asset_cfg": ROBOT_CFG}),
      "action_rate": RewardTermCfg(func=ascento_mdp.rewards.action_rate_penalty, weight=-0.02),
    },
    terminations={
      "fallen": TerminationTermCfg(func=ascento_mdp.terminations.fallen, params={"asset_cfg": ROBOT_CFG}),
      "excessive_velocity": TerminationTermCfg(func=ascento_mdp.terminations.excessive_velocity, params={"asset_cfg": ROBOT_CFG}),
      "nonfinite": TerminationTermCfg(func=ascento_mdp.terminations.nonfinite, params={"asset_cfg": ROBOT_CFG}),
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    },
    metrics={
      "tilt_radians": MetricsTermCfg(func=ascento_mdp.metrics.tilt_radians, params={"asset_cfg": ROBOT_CFG}),
      "applied_effort": MetricsTermCfg(func=ascento_mdp.metrics.applied_effort, params={"asset_cfg": ROBOT_CFG}),
      "root_speed": MetricsTermCfg(func=ascento_mdp.metrics.root_speed, params={"asset_cfg": ROBOT_CFG}),
    },
    episode_length_s=20.0 if not play else 10000.0,
    auto_reset=True,
    scale_rewards_by_dt=True,
  )
  return cfg
