import jax
import jax.numpy as jp

from ascento import AscentoBalance
from ascento import actuator
from ascento.constants import PEAK_TORQUE


def test_torque_speed_envelope_preserves_braking_authority():
    peak = jp.asarray(PEAK_TORQUE)
    velocity = jp.asarray([11.0, 11.0, 19.0, 11.0, 11.0, 19.0])
    assert bool(jp.all(actuator.torque_speed_envelope(peak, velocity) < peak))
    assert bool(jp.allclose(actuator.torque_speed_envelope(-peak, velocity), peak))


def test_same_substep_direct_torque_application():
    env = AscentoBalance()
    state = env.reset(jax.random.PRNGKey(1))
    data, _, applied = actuator.substep(state.data, state.info["torque_state"], jp.ones(6), env.mjx_model)
    assert bool(jp.allclose(data.ctrl, applied))
    assert float(jp.max(jp.abs(applied[jp.asarray([2, 5])]))) <= 40.0 + 1e-5
