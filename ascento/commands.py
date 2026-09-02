"""Command sampling for the fixed six-channel policy interface."""
from __future__ import annotations

import jax
import jax.numpy as jp

from .constants import (
    COMMAND_HEIGHT,
    COMMAND_JUMP_DISTANCE,
    COMMAND_JUMP_HEIGHT,
    COMMAND_JUMP_TRIGGER,
    COMMAND_SIZE,
    COMMAND_VX,
    COMMAND_YAW_RATE,
)


def zero_command(nominal_height: float) -> jax.Array:
    """Returns the neutral command used by the initial balance stage."""
    return jp.array([0.0, 0.0, nominal_height, 0.0, 0.0, 0.0], jp.float32)


def sample_command(
    rng: jax.Array,
    *,
    max_vx: float,
    max_yaw_rate: float,
    height_range: float,
    nominal_height: float,
    jump_probability: float = 0.0,
    max_jump_height: float = 0.0,
    max_jump_distance: float = 0.0,
) -> jax.Array:
    """Samples a task command without sim-to-real observation randomization."""
    key_vx, key_yaw, key_height, key_jump, key_target = jax.random.split(rng, 5)
    command = zero_command(nominal_height)
    command = command.at[COMMAND_VX].set(
        jax.random.uniform(key_vx, (), minval=-max_vx, maxval=max_vx)
    )
    command = command.at[COMMAND_YAW_RATE].set(
        jax.random.uniform(key_yaw, (), minval=-max_yaw_rate, maxval=max_yaw_rate)
    )
    command = command.at[COMMAND_HEIGHT].set(
        jax.random.uniform(
            key_height,
            (),
            minval=nominal_height - height_range,
            maxval=nominal_height + height_range,
        )
    )
    jump = jax.random.bernoulli(key_jump, jump_probability).astype(jp.float32)
    targets = jax.random.uniform(key_target, (2,), minval=0.0, maxval=1.0)
    command = command.at[COMMAND_JUMP_TRIGGER].set(jump)
    command = command.at[COMMAND_JUMP_HEIGHT].set(jump * targets[0] * max_jump_height)
    command = command.at[COMMAND_JUMP_DISTANCE].set(
        jump * (2.0 * targets[1] - 1.0) * max_jump_distance
    )
    assert command.shape == (COMMAND_SIZE,)
    return command
