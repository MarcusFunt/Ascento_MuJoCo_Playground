"""Balance environment using Guard 2.0-oriented actuator/contact dynamics."""
import math
import mujoco, numpy as np
from prepare_ascento import model
from guard2_physics import Guard2ActuatorModel, apply_torque

STANCE=np.array([-3.14,-3.14,0,-3.14,-3.14,0],dtype=np.float32)
FLOOR=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
WHEEL_BODIES={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'left_wheel'),mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'right_wheel')}

class AscentoBalanceEnv:
 def __init__(self):
  self.data=mujoco.MjData(model); self.resets=0
  self.actuators=Guard2ActuatorModel(model.opt.timestep)
  self.reset()
 def reset(self,pitch=0.0):
  d=self.data; mujoco.mj_resetData(model,d); self.actuators.reset()
  d.qpos[:7]=[0,0,.25,math.cos(pitch/2),0,math.sin(pitch/2),0]; d.qpos[7:]=STANCE
  mujoco.mj_forward(model,d); self.resets+=1; return self.observe()
 def observe(self):
  d=self.data; return np.r_[2*math.atan2(d.qpos[5],d.qpos[3]),d.qvel[4],d.qvel[0],d.qpos[7:13],d.qvel[6:12]].astype(np.float32)
 def body_hit_ground(self):
  for c in self.data.contact[:self.data.ncon]:
   if FLOOR not in (c.geom1,c.geom2): continue
   other=c.geom2 if c.geom1==FLOOR else c.geom1
   if model.geom_bodyid[other] not in WHEEL_BODIES: return True
  return False
 def step(self,action,auto_reset=True):
  d=self.data
  # The policy still commands desired joint torque; hardware dynamics are applied
  # at each 2 ms physics step instead of clipping once every 10 ms policy step.
  for _ in range(5):
   apply_torque(model,d,self.actuators,action)
   mujoco.mj_step(model,d)
  hit=self.body_hit_ground(); pitch=2*math.atan2(d.qpos[5],d.qpos[3])
  reward=1.0-1.8*pitch*pitch-0.0008*float(np.dot(d.ctrl,d.ctrl))
  if hit: reward-=50.0
  obs=self.observe(); info={'body_ground_hit':hit,'resets':self.resets,'applied_torque':d.ctrl.copy()}
  if hit and auto_reset: obs=self.reset(); info['reset']=True
  return obs,float(reward),hit,info
