"""Registered Ascento mjlab tasks."""

from mjlab.tasks.registry import register_mjlab_task

from .balance.env_cfg import ascento_balance_env_cfg
from .balance.rl_cfg import AscentoBalanceRlCfg
from .jump.env_cfg import ascento_jump_env_cfg
from .jump.rl_cfg import AscentoJumpRlCfg
from .recovery.env_cfg import ascento_recovery_env_cfg
from .recovery.rl_cfg import AscentoRecoveryRlCfg
from .velocity.env_cfg import ascento_velocity_env_cfg
from .velocity.rl_cfg import AscentoVelocityRlCfg

register_mjlab_task(
  task_id="Ascento-Balance-Flat",
  env_cfg=ascento_balance_env_cfg(),
  play_env_cfg=ascento_balance_env_cfg(play=True),
  rl_cfg=AscentoBalanceRlCfg,
)
register_mjlab_task(
  task_id="Ascento-Velocity-Flat",
  env_cfg=ascento_velocity_env_cfg(),
  play_env_cfg=ascento_velocity_env_cfg(play=True),
  rl_cfg=AscentoVelocityRlCfg,
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
