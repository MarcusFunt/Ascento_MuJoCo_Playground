"""World-target locomotion built directly on the balance observation contract."""

from __future__ import annotations

from mjlab.managers.event_manager import EventTermCfg

from ascento_mjlab import mdp as ascento_mdp
from ascento_mjlab.tasks.balance.env_cfg import ROBOT_CFG
from ascento_mjlab.tasks.balance_quiet.env_cfg import ascento_balance_quiet_env_cfg


def ascento_locomotion_env_cfg(play: bool = False, num_envs: int = 512):
    """Start with settle -> mild push -> recover -> 5--20 cm target -> stop.

    This intentionally preserves the 41-dimensional balance actor observation
    topology, including world-frame target XY and heading error.  It does not
    introduce velocity or height commands, so compatible balance actor weights
    can be transferred without a policy input/output adapter.
    """
    cfg = ascento_balance_quiet_env_cfg(play=play, num_envs=num_envs)
    if not play:
        cfg.events.pop("balance_push", None)
        cfg.events["settle_triggered_sequence"] = EventTermCfg(
            func=ascento_mdp.events.SettleTriggeredLocomotionSequence,
            mode="interval",
            interval_range_s=(0.01, 0.01),
            params={
                "settle_hold_s": 0.75,
                "min_delta_v": 0.05,
                "max_delta_v": 0.15,
                "fore_aft_probability": 0.75,
                "min_target_distance_m": 0.05,
                "max_target_distance_m": 0.20,
                "target_reached_distance_m": 0.035,
                "asset_cfg": ROBOT_CFG,
            },
        )
    cfg.task_id = "Ascento-Locomotion-Flat"
    return cfg
