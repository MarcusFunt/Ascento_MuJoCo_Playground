import math,mujoco,numpy as np
from prepare_ascento import model
TARGET=np.array([-3.14,-3.14,0,-3.14,-3.14,0]); TORSO=1; FLOOR=0
def hit(d):
 return any(FLOOR in (c.geom1,c.geom2) and TORSO==model.geom_bodyid[c.geom2 if c.geom1==FLOOR else c.geom1] for c in d.contact[:d.ncon])
def run(kp,kd,sign):
 d=mujoco.MjData(model); p=.12; d.qpos[:7]=[0,0,.25,math.cos(p/2),0,math.sin(p/2),0]; d.qpos[7:]=TARGET;mujoco.mj_forward(model,d)
 for t in range(2500):
  th=2*math.atan2(d.qpos[5],d.qpos[3]); u=np.clip(kp*th+kd*d.qvel[4],-40,40);q=d.qpos[7:];v=d.qvel[6:]
  d.ctrl[:]=np.clip(100*(TARGET-q)-10*v,-40,40);d.ctrl[2]=u;d.ctrl[5]=sign*u;mujoco.mj_step(model,d)
  if hit(d):return 0
 return 1
for sign in (-1,1):
 for kp in (-300,-100,100,300):
  for kd in (-60,-20,20,60):
   if run(kp,kd,sign):print('SURVIVES',sign,kp,kd)
