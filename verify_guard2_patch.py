"""Run after installing MuJoCo to verify the patched model compiles and limits are active."""
import math
import mujoco
import numpy as np
from prepare_ascento import model
from guard2_physics import LEG_Q_MIN, LEG_Q_MAX, Guard2ActuatorModel

assert model.nu == 6, model.nu
for name in ('left_hip', 'left_knee', 'right_hip', 'right_knee'):
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
    assert model.jnt_limited[jid]
    lo,hi=model.jnt_range[jid]
    assert abs(lo-LEG_Q_MIN)<1e-6 and abs(hi-LEG_Q_MAX)<1e-6,(name,lo,hi)
# Smoke-step at the nominal straight-down pose.
d=mujoco.MjData(model)
d.qpos[:7]=[0,0,.25,1,0,0,0]
d.qpos[7:]=[-math.pi,-math.pi,0,-math.pi,-math.pi,0]
mujoco.mj_forward(model,d)
a=Guard2ActuatorModel(model.opt.timestep)
for _ in range(20):
    d.ctrl[:]=a.step(np.array([10,10,10,10,10,10]),d.qpos[7:13],d.qvel[6:12])
    mujoco.mj_step(model,d)
assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
print('GUARD2_MUJOCO_SMOKE_TEST_OK')
