import jax
import jax.numpy as jp
from mujoco import mjx

from ascento import AscentoBalance


def test_sideways_pose_is_terminal_even_when_root_is_above_floor():
    env = AscentoBalance()
    state = env.reset(jax.random.PRNGKey(11))
    # 90 degrees about world Y: the root remains well above the old 10 cm
    # cutoff, but the robot is physically fallen.
    data = state.data.replace(
        qpos=state.data.qpos.at[2].set(0.50).at[3:7].set(
            jp.asarray([jp.sqrt(0.5), 0.0, jp.sqrt(0.5), 0.0])
        )
    )
    data = mjx.forward(env.mjx_model, data)
    gravity, *_rest, nonwheel_collision = env._dynamics_signals(data)
    assert bool(env._get_termination(data, gravity, nonwheel_collision))


def test_actor_observation_remains_bounded_for_extreme_velocity():
    env = AscentoBalance()
    state = env.reset(jax.random.PRNGKey(12))
    data = state.data.replace(qvel=jp.full_like(state.data.qvel, 1e9))
    observation = env._get_obs(data, state.info)
    assert bool(jp.all(jp.isfinite(observation)))
    assert float(jp.max(jp.abs(observation))) <= 3.0


def test_tracking_rewards_are_zero_when_not_upright():
    env = AscentoBalance()
    state = env.reset(jax.random.PRNGKey(13))
    terms = env._get_reward_terms(
        state.data,
        jp.zeros(6),
        state.info,
        jp.asarray([0.0, 0.0, 0.0]),
        jp.zeros(3),
        jp.zeros(3),
        jp.ones(2, dtype=jp.bool_),
        jp.asarray(False),
    )
    assert float(terms["vx_tracking"]) == 0.0
    assert float(terms["yaw_tracking"]) == 0.0
    assert float(terms["height"]) == 0.0
