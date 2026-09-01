"""A bounded tanh-Gaussian action distribution for direct-torque PPO."""
from __future__ import annotations

import jax
import jax.numpy as jp

from brax.training import distribution


class BoundedNormalTanhDistribution(distribution.NormalTanhDistribution):
    """Keeps PPO's action mean and standard deviation in a usable range.

    The policy head is otherwise unbounded.  Extreme but finite observations
    can then make the tanh Jacobian dominate entropy and lock actions at their
    limits.  Bounding parameters inside the distribution applies identically
    during rollout, log-prob evaluation, and entropy evaluation.
    """

    minimum_std: float = 0.02
    maximum_std: float = 0.20

    def create_dist(self, parameters):
        loc, raw_scale = jp.split(parameters, 2, axis=-1)
        # PPO's tanh-normal policy emits unconstrained scale logits.  Mapping
        # them through a finite sigmoid interval gives direct-torque rollouts a
        # low, controlled initial variance without killing the scale gradient.
        scale = self.minimum_std + (self.maximum_std - self.minimum_std) * jax.nn.sigmoid(
            jp.clip(raw_scale, -8.0, 8.0)
        )
        return distribution._NormalDistribution(jp.clip(loc, -3.0, 3.0), scale)
