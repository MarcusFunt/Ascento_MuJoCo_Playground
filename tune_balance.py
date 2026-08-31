import math, mujoco, numpy as np
from prepare_ascento import model
TARGET=np.array([-1.57,2.5,0,-1.57,2.5,0])
def reset(pitch):
 d=mujoco.MjData(model); d.qpos[:7]=[0,0,-.25,math.cos(pitch/2),0,math.sin(pitch/2),0]; d.qpos[7:]=TARGET
 mujoco.mj_forward(model,d); return d
def rollout(kp,kd,pitch):
 d=reset(pitch)
 for t in range(2500):
  q=d.qpos[7:13]; v=d.qvel[6:12]; th=2*math.atan2(d.qpos[5],d.qpos[3]); om=d.qvel[4]
  u=np.clip(kp*th+kd*om,-35,35)
  d.ctrl[:]=np.clip(80*(TARGET-q)-8*v,-40,40); d.ctrl[2]=u; d.ctrl[5]=-u
  mujoco.mj_step(model,d)
  if abs(th)>.75: return 0
 th=2*math.atan2(d.qpos[5],d.qpos[3])
 return 1 if abs(th)<.18 else 0
for kp in (-120,-80,-50,-30,30,50,80,120):
 for kd in (-30,-15,-8,-3,3,8,15,30):
  score=sum(rollout(kp,kd,p) for p in (-.18,-.10,.10,.18))
  if score: print(kp,kd,score)
