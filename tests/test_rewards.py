import jax
import jax.numpy as jp

from ascento import AscentoBalance


def test_reward_terms_are_finite_and_upright_is_positive():
    env = AscentoBalance()
    state = env.reset(jax.random.PRNGKey(3))
    state = env.step(state, jp.zeros(6))
    terms = [value for name, value in state.metrics.items() if name.startswith("reward/")]
    assert bool(jp.all(jp.isfinite(jp.asarray(terms))))
    assert float(state.metrics["reward/upright"]) > 0.0
