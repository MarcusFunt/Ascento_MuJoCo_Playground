"""Run the trained policy with immediate fall penalty and reset."""
import time
import mujoco.viewer, torch
from balance_env import AscentoBalanceEnv, model

policy=torch.jit.load('ascento_balance_policy.ts').eval()
env=AscentoBalanceEnv()
reset_requested=False
def keys(_window,key,_scan,action,_mods):
 global reset_requested
 if action==1 and key==259: reset_requested=True
print('Fall-aware Ascento policy running. Body-floor contact: -50 reward and reset. Backspace resets manually.')
with mujoco.viewer.launch_passive(model,env.data,key_callback=keys) as viewer:
 while viewer.is_running():
  if reset_requested: obs=env.reset(); reset_requested=False
  else: obs=env.observe()
  with torch.no_grad(): action=policy(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
  _,reward,fallen,info=env.step(action,auto_reset=True)
  if fallen: print(f'body-ground hit: reward={reward:.1f}; reset #{info["resets"]+1}')
  viewer.sync();time.sleep(.01)
