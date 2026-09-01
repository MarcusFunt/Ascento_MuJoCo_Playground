import math, mujoco, numpy as np
from prepare_ascento import model
TARGET=np.array([-3.14,-3.14,0,-3.14,-3.14,0])
TORSO=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base')
FLOOR=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
def hit(d):
 for c in d.contact[:d.ncon]:
  if FLOOR in (c.geom1,c.geom2) and TORSO==model.geom_bodyid[c.geom2 if c.geom1==FLOOR else c.geom1]: return True
 return False
def run(pitch):
 d=mujoco.MjData(model); d.qpos[:7]=[0,0,.25,math.cos(pitch/2),0,math.sin(pitch/2),0]; d.qpos[7:]=TARGET; mujoco.mj_forward(model,d)
 peak=0
 for t in range(3000):
  theta=2*math.atan2(d.qpos[5],d.qpos[3]); omega=d.qvel[4]; q=d.qpos[7:13]; v=d.qvel[6:12]
  u=np.clip(-85*theta-20*omega-2*d.qvel[0],-35,35); d.ctrl[:]=np.clip(90*(TARGET-q)-9*v,-40,40); d.ctrl[2]=u; d.ctrl[5]=-u
  mujoco.mj_step(model,d); peak=max(peak,abs(theta))
  if hit(d): return 'FALL',round(peak,3)
 return 'OK',round(peak,3),round(2*math.atan2(d.qpos[5],d.qpos[3]),3)
for p in (-.25,-.15,0,.15,.25): print(p,run(p))
