"""Balance recovery curriculum focused on the measured V5 edge cases."""

from __future__ import annotations

from mjlab.managers.event_manager import EventTermCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.tasks.balance.env_cfg import ROBOT_CFG
from ascento_mjlab.tasks.balance_quiet.env_cfg import ascento_balance_quiet_env_cfg

_NORMAL_POSE = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (0.0, 0.0),
    "roll": (-0.08, 0.08),
    "pitch": (-0.08, 0.08),
    "yaw": (-3.14159, 3.14159),
}
_NORMAL_VELOCITY = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.05, 0.05),
    "roll": (-0.10, 0.10),
    "pitch": (-0.10, 0.10),
    "yaw": (-0.10, 0.10),
}
_HARD_POSE_START = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (0.0, 0.0),
    "roll": (-0.10, 0.10),
    "pitch": (0.08, 0.10),
    "yaw": (-3.14159, 3.14159),
}
_HARD_POSE_END = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (0.0, 0.0),
    "roll": (-0.15, 0.15),
    "pitch": (0.08, 0.15),
    "yaw": (-3.14159, 3.14159),
}
_HARD_VELOCITY_START = {
    "x": (-0.10, 0.10),
    "y": (-0.07, 0.07),
    "z": (-0.05, 0.05),
    "roll": (-0.20, 0.20),
    "pitch": (-0.20, 0.20),
    "yaw": (-0.15, 0.15),
}
_HARD_VELOCITY_END = {
    "x": (-0.25, 0.25),
    "y": (-0.10, 0.10),
    "z": (-0.05, 0.05),
    "roll": (-0.50, 0.50),
    "pitch": (-0.50, 0.50),
    "yaw": (-0.25, 0.25),
}


def ascento_balance_recovery_env_cfg(play: bool = False, num_envs: int = 512):
    cfg = ascento_balance_quiet_env_cfg(play=play, num_envs=num_envs)
    cfg.events["reset_supported_pose"] = EventTermCfg(
        func=ascento_mdp.events.mixed_balance_recovery_reset,
        mode="reset",
        params={
            "asset_cfg": ROBOT_CFG,
            "normal_pose_range": _NORMAL_POSE,
            "normal_velocity_range": _NORMAL_VELOCITY,
            "hard_pose_range_start": _HARD_POSE_START,
            "hard_pose_range_end": _HARD_POSE_END,
            "hard_velocity_range_start": _HARD_VELOCITY_START,
            "hard_velocity_range_end": _HARD_VELOCITY_END,
            "hard_fraction_start": 0.0 if play else 0.10,
            "hard_fraction_end": 0.0 if play else 0.30,
            "ramp_control_steps": 120_000,
        },
    )
    cfg.task_id = "Ascento-Balance-Recovery-Flat"
    return cfg
