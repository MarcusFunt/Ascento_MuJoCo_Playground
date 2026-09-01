import jax
import jax.numpy as jp

from ascento import AscentoBalance
from ascento.constants import LEG_Q_MAX, LEG_Q_MIN, OBS_SIZE


def test_model_schema_and_no_passive_leg_spring():
    env = AscentoBalance()
    model = env.mj_model
    assert (model.nq, model.nv, model.nu) == (13, 12, 6)
    for name in ("left_hip", "left_knee", "right_hip", "right_knee"):
        joint = model.joint(name)
        assert jp.allclose(joint.range, jp.asarray([LEG_Q_MIN, LEG_Q_MAX]))
        assert model.jnt_stiffness[joint.id] == 0.0


def test_reset_and_step_jit_with_fixed_schema():
    env = AscentoBalance()
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (OBS_SIZE,)
    next_state = step(state, jp.zeros(6))
    assert next_state.obs.shape == (OBS_SIZE,)
    assert bool(jp.all(jp.isfinite(next_state.obs)))
