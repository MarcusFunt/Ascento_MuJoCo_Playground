import mujoco, numpy as np
from prepare_ascento import model
d=mujoco.MjData(model)
d.qpos[:7]=[0,0,-.25,1,0,0,0]
d.qpos[7:]=[-1.57,2.5,0,-1.57,2.5,0]
mujoco.mj_forward(model,d)
for name in ('base_link','ascento/head','ascento/wheel_left','ascento/wheel_right'):
    i=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
    print(name,d.xpos[i])
print('qpos',d.qpos)
print('qvel',d.qvel)
for t in range(1000):
    q=d.qpos[7:13]; v=d.qvel[6:12]
    target=np.array([-1.57,2.5,0,-1.57,2.5,0])
    d.ctrl[:]=np.clip(80*(target-q)-8*v,-40,40)
    d.ctrl[2]=d.ctrl[5]=0
    mujoco.mj_step(model,d)
print('after',d.qpos[:7],d.qvel[:6])
