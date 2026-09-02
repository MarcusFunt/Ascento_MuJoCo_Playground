import jax
import jax.numpy as jp

from ascento import AscentoBalance


def test_named_wheel_contacts_and_forces_are_fixed_shape():
    env = AscentoBalance()
    state = env.reset(jax.random.PRNGKey(2))
    state = env.step(state, jp.zeros(6))
    assert state.info["wheel_contact"].shape == (2,)
    assert state.info["wheel_contact_force"].shape == (2,)
    assert bool(jp.all(state.info["wheel_contact_force"] >= 0.0))
