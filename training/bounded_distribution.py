"""A bounded tanh-Gaussian action distribution for direct-torque PPO."""
from __future__ import annotations

import jax.numpy as jp

from brax.training import distribution


class BoundedNormalTanhDistribution(distribution.NormalTanhDistribution):
    """Keeps PPO's action mean and standard deviation in a usable range.

    The policy head is otherwise unbounded.  Extreme but finite observations
    can then make the tanh Jacobian dominate entropy and lock actions at their
    limits.  Bounding parameters inside the distribution applies identically
    during rollout, log-prob evaluation, and entropy evaluation.
    """

    def create_dist(self, parameters):
        loc, raw_scale = jp.split(parameters, 2, axis=-1)
        bounded = jp.concatenate((jp.clip(loc, -3.0, 3.0), jp.clip(raw_scale, -5.0, 1.0)), axis=-1)
        return super().create_dist(bounded)
