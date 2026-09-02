"""Go1Getup-inspired recovery task without position-target action control."""
from .base import AscentoBaseEnv


class AscentoRecovery(AscentoBaseEnv):
    """Uses broad but mechanically reachable tilted/velocity reset states."""

    def __init__(self, **kwargs):
        kwargs.setdefault("reset_tilt", 0.55)
        kwargs.setdefault("reset_angular_velocity", 3.0)
        kwargs.setdefault("reset_linear_velocity", 1.0)
        kwargs.setdefault("reset_leg_variation", 0.30)
        super().__init__(enable_jump_rewards=False, **kwargs)
