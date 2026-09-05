"""Recovery specialist configuration."""

from mjlab.envs import mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.tasks.balance.env_cfg import ascento_balance_env_cfg


def ascento_recovery_env_cfg(play: bool = False, num_envs: int = 512):
    cfg = ascento_balance_env_cfg(play=play, num_envs=num_envs)
    cfg.events["reset_supported_pose"].params["pose_range"].update(
        {"roll": (-0.35, 0.35), "pitch": (-0.45, 0.45)}
    )
    cfg.events["reset_supported_pose"].params["velocity_range"].update(
        {
            "x": (-0.8, 0.8),
            "y": (-0.4, 0.4),
            "roll": (-1.5, 1.5),
            "pitch": (-2.0, 2.0),
            "yaw": (-0.5, 0.5),
        }
    )

    # Recovery should not only work when the disturbance happened at reset.
    # mjlab's standard velocity-kick event is deterministic under the env seed,
    # acts directly on the physical root state, and avoids adding sim-to-real
    # noise that is irrelevant to this simulation-only project. Keep it out of
    # play/evaluation configs, where disturbances are specified explicitly by
    # the quantitative scenario suite instead.
    if not play:
        cfg.events["recovery_push"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.5, 3.5),
            params={
                "velocity_range": {
                    "x": (-0.35, 0.35),
                    "y": (-0.15, 0.15),
                    "z": (0.0, 0.0),
                    "roll": (-0.25, 0.25),
                    "pitch": (-0.35, 0.35),
                    "yaw": (-0.20, 0.20),
                }
            },
        )

    cfg.rewards["recovery_progress"] = RewardTermCfg(
        func=ascento_mdp.recovery.recovery_progress,
        weight=2.0,
    )
    cfg.metrics["recovery_success"] = MetricsTermCfg(
        func=ascento_mdp.recovery.RecoverySuccess,
    )
    cfg.episode_length_s = 5.0 if not play else 10000.0
    return cfg
