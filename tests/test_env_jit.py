import jax
import jax.numpy as jp

from ascento import AscentoBalance


def test_thousand_direct_torque_steps_remain_finite_and_batchable():
    env = AscentoBalance()
    state = env.reset(jax.random.PRNGKey(4))

    def rollout(initial):
        def one(current, _):
            nxt = env.step(current, jp.zeros(6))
            return nxt, nxt.data.qpos
        return jax.lax.scan(one, initial, None, length=1000)

    _, qpos = jax.jit(rollout)(state)
    assert bool(jp.all(jp.isfinite(qpos)))
    keys = jax.random.split(jax.random.PRNGKey(5), 4)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    batched = step(reset(keys), jp.zeros((4, 6)))
    assert batched.obs.shape == (4, env.observation_size)
    assert bool(jp.all(jp.isfinite(batched.data.qvel)))
