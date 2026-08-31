"""Minimal PPO balance task; intentionally no teacher, PD, VMC, or imitation."""
import jax.numpy as jp
from .base import AscentoBaseEnv
from .constants import BASE_HEIGHT, STANCE

class AscentoBalance(AscentoBaseEnv):
    """Balance from exact nominal upright resets using direct torque actions."""
    def _get_reward(self, data, action, info, done):
        gravity = self._projected_gravity(data)
        upright_error = jp.sum(jp.square(gravity - jp.asarray([0.0, 0.0, -1.0])))
        upright = jp.exp(-4.0 * upright_error)
        height = jp.exp(-30.0 * jp.square(data.qpos[2] - BASE_HEIGHT))
        still = jp.exp(-0.08 * jp.sum(jp.square(data.qvel[:6])))
        posture = jp.exp(-0.25 * jp.sum(jp.square(data.qpos[7:13] - STANCE)))
        action_rate = jp.sum(jp.square(action - info["last_action"]))
        torque_cost = jp.sum(jp.square(info["last_torque"] / 40.0))
        reward = (2.0 * upright + 0.5 * height + 0.25 * still +
                  0.25 * posture - 0.01 * action_rate -
                  0.01 * torque_cost)
        return reward + jp.where(done, -10.0, 0.0)
