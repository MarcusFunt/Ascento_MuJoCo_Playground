"""Registered Ascento mjlab tasks."""

# Import configuration modules before touching mjlab's registry. mjlab
# discovers task entry points while its own task package is being imported;
# keeping registry access lazy makes both import directions safe.
from ascento_mjlab.horizon_curriculum import HorizonCurriculumRunner

from .balance.env_cfg import ascento_balance_env_cfg
from .balance.rl_cfg import AscentoBalanceRlCfg
from .jump.env_cfg import ascento_jump_env_cfg
from .jump.rl_cfg import AscentoJumpRlCfg
from .recovery.env_cfg import ascento_recovery_env_cfg
from .recovery.rl_cfg import AscentoRecoveryRlCfg
from .velocity.env_cfg import ascento_velocity_env_cfg
from .velocity.rl_cfg import AscentoVelocityRlCfg


def _register_tasks() -> None:
    """Register tasks after all config modules have finished importing."""
    from mjlab.tasks.registry import register_mjlab_task

    register_mjlab_task(
        task_id="Ascento-Balance-Flat",
        env_cfg=ascento_balance_env_cfg(),
        play_env_cfg=ascento_balance_env_cfg(play=True),
        rl_cfg=AscentoBalanceRlCfg,
        runner_cls=HorizonCurriculumRunner,
    )
    register_mjlab_task(
        task_id="Ascento-Velocity-Flat",
        env_cfg=ascento_velocity_env_cfg(),
        play_env_cfg=ascento_velocity_env_cfg(play=True),
        rl_cfg=AscentoVelocityRlCfg,
        runner_cls=HorizonCurriculumRunner,
    )
    register_mjlab_task(
        task_id="Ascento-Recovery-Flat",
        env_cfg=ascento_recovery_env_cfg(),
        play_env_cfg=ascento_recovery_env_cfg(play=True),
        rl_cfg=AscentoRecoveryRlCfg,
    )
    register_mjlab_task(
        task_id="Ascento-Jump-Flat",
        env_cfg=ascento_jump_env_cfg(),
        play_env_cfg=ascento_jump_env_cfg(play=True),
        rl_cfg=AscentoJumpRlCfg,
    )


_register_tasks()
