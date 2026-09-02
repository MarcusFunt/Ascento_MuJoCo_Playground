"""Balance and command-tracking task with direct motor-torque actions."""
from .base import AscentoBaseEnv


class AscentoBalance(AscentoBaseEnv):
    """Normal near-upright resets; commands can be widened by curriculum stage."""

    def __init__(self, action_scale: float = 0.35, **kwargs):
        super().__init__(enable_jump_rewards=False, action_scale=action_scale, **kwargs)
