"""Balance and command-tracking task with direct motor-torque actions."""
from .base import AscentoBaseEnv


class AscentoBalance(AscentoBaseEnv):
    """Normal near-upright resets; commands can be widened by curriculum stage."""

    def __init__(self, **kwargs):
        super().__init__(enable_jump_rewards=False, action_scale=0.10, **kwargs)
