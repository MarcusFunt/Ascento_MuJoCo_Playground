"""Six-phase event-tracked jumping environment; PPO remains the controller."""
from .base import AscentoBaseEnv


class AscentoJump(AscentoBaseEnv):
    """Enables jump commands and phase-gated rewards with unchanged action space."""

    def __init__(self, **kwargs):
        kwargs.setdefault("jump_probability", 0.5)
        kwargs.setdefault("max_jump_height", 0.12)
        super().__init__(enable_jump_rewards=True, **kwargs)
