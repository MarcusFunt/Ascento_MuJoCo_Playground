"""Render review videos using the saved policy with Guard 2.0 actuator dynamics."""
import math
from pathlib import Path
import imageio.v2 as imageio
import mujoco, numpy as np, torch
from prepare_ascento import model
from guard2_physics import Guard2ActuatorModel, apply_torque
from balance_env import STANCE

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'videos'; OUT.mkdir(exist_ok=True)
policy=torch.jit.load(str(ROOT/'ascento_balance_policy.ts')).eval()
def reset(pitch):
 d=mujoco.MjData(model); d.qpos[:7]=[0,0,.25,math.cos(pitch/2),0,math.sin(pitch/2),0]; d.qpos[7:]=STANCE; mujoco.mj_forward(model,d); return d

def observe(d):
 return np.r_[2*math.atan2(d.qpos[5],d.qpos[3]),d.qvel[4],d.qvel[0],d.qpos[7:13],d.qvel[6:12]].astype(np.float32)

def record(name,pitch):
 d=reset(pitch); actuator=Guard2ActuatorModel(model.opt.timestep); renderer=mujoco.Renderer(model,height=480,width=640)
 cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
 cam.lookat[:]=[.45,0,.5]; cam.distance=1.55; cam.azimuth=145; cam.elevation=-18
 frames=[]
 for t in range(240):
  with torch.no_grad(): requested=policy(torch.from_numpy(observe(d)).unsqueeze(0)).numpy()[0]
  for _ in range(5):
   apply_torque(model,d,actuator,requested); mujoco.mj_step(model,d)
  if t%3==0:
   renderer.update_scene(d,camera=cam); frames.append(renderer.render())
 renderer.close(); imageio.mimsave(OUT/f'{name}.mp4',frames,fps=20,macro_block_size=1)
 print(name,'frames=',len(frames),'final_pitch=',round(2*math.atan2(d.qpos[5],d.qpos[3]),4))
for name,pitch in [('balance_center',0.0),('balance_left_recovery',-.30),('balance_right_recovery',.30)]: record(name,pitch)
print('VIDEOS_COMPLETE',OUT)
