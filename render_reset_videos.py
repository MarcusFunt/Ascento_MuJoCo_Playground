"""Render current fall-aware balance episodes."""
import math
from pathlib import Path
import imageio.v2 as imageio
import mujoco, torch
from balance_env import AscentoBalanceEnv, model

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'videos';OUT.mkdir(exist_ok=True)
policy=torch.jit.load(str(ROOT/'ascento_balance_policy.ts')).eval()
def record(name,pitch):
 env=AscentoBalanceEnv();env.reset(pitch);renderer=mujoco.Renderer(model,height=480,width=640)
 cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=[0,0,.35];cam.distance=1.7;cam.azimuth=135;cam.elevation=-16
 frames=[];resets=0
 for t in range(300):
  obs=env.observe()
  with torch.no_grad():action=policy(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
  _,_,fallen,_=env.step(action,auto_reset=True);resets+=int(fallen)
  if t%3==0:renderer.update_scene(env.data,camera=cam);frames.append(renderer.render())
 renderer.close();imageio.mimsave(OUT/f'{name}.mp4',frames,fps=20,macro_block_size=1)
 print(name,'frames=',len(frames),'automatic_resets=',resets)
for name,pitch in [('ascento_center_reset',0.0),('ascento_left_reset',-.12),('ascento_right_reset',.12)]:record(name,pitch)
