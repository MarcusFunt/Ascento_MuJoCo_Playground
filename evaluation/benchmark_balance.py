"""Deterministic validation for a PPO policy on exact nominal resets."""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("ASCENTO_JAX_PLATFORM", "cuda")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
import jax
import jax.numpy as jp
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks
from ascento.balance import AscentoBalance


def save_mp4(model, qpos, qvel, output_path, fps):
    """Render one nominal balance trajectory to an MP4 file."""
    import imageio.v2 as imageio
    import mujoco

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=1,
    )
    renderer = mujoco.Renderer(model, height=360, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth = 135
    camera.elevation = -15
    camera.distance = 1.65
    camera.lookat[:] = np.array([0.0, 0.0, 0.36])
    # Render-only lift keeps the game-character silhouette readable under studio lighting.
    model.mat_emission[:] = np.maximum(model.mat_emission, 0.12)
    data = mujoco.MjData(model)
    try:
        for frame_qpos, frame_qvel in zip(qpos, qvel):
            data.qpos[:] = np.asarray(frame_qpos)
            data.qvel[:] = np.asarray(frame_qvel)
            mujoco.mj_forward(model, data)
            camera.lookat[:] = np.asarray(data.qpos[:3])
            camera.lookat[2] += 0.12
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path,
                        default=Path(__file__).parent.parent /
                        "training/artifacts_cuda_smoke")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--save-mp4", type=Path, default=None,
                        help="save the best nominal episode as MP4")
    parser.add_argument("--fps", type=int, default=100,
                        help="output video frame rate; default matches 10 ms control")
    args = parser.parse_args()
    with (args.artifact / "policy_params.pkl").open("rb") as f:
        params = pickle.load(f)
    manifest = json.loads((args.artifact / "training_manifest.json").read_text())
    hidden_sizes = tuple(manifest["network_hidden_layer_sizes"])
    net = networks.make_ppo_networks(
        46, 6, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden_sizes,
        value_hidden_layer_sizes=hidden_sizes,
        policy_obs_key="state", value_obs_key="state",
    )
    policy = networks.make_inference_fn(net)(params, deterministic=True)
    env = AscentoBalance()
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(91), args.episodes)
    state = reset(keys)
    def rollout(state, _):
        action, _ = policy(state.obs, jax.random.PRNGKey(0))
        state = step(state, action)
        return state, (state.reward, state.done, state.data.qpos, state.data.qvel)
    state, (rewards, dones, qpos, qvel) = jax.jit(
        lambda s: jax.lax.scan(rollout, s, None, length=600)
    )(state)
    rewards = np.asarray(jax.device_get(rewards))
    dones = np.asarray(jax.device_get(dones))
    qpos = np.asarray(jax.device_get(qpos))
    qvel = np.asarray(jax.device_get(qvel))
    alive = 1.0 - dones
    survival = alive.cumprod(axis=0).sum(axis=0)
    result = {
        "backend": jax.default_backend(),
        "episodes": args.episodes,
        "survival_steps": survival.tolist(),
        "mean_survival_steps": float(survival.mean()),
        "mean_return": float(rewards.sum(axis=0).mean()),
        "full_episode_fraction": float((survival >= 600).mean()),
        "nominal_reset": True,
        "policy_source": "Brax PPO rollouts; random initialization; no teacher/PD seed",
    }
    if args.save_mp4 is not None:
        episode_returns = rewards.sum(axis=0)
        best_episode = int(np.argmax(episode_returns))
        save_mp4(
            env.mj_model,
            qpos[:, best_episode],
            qvel[:, best_episode],
            args.save_mp4,
            args.fps,
        )
        result["best_episode_index"] = best_episode
        result["best_episode_return"] = float(episode_returns[best_episode])
        result["mp4_path"] = str(args.save_mp4.resolve())
    print(json.dumps(result, indent=2))
    (args.artifact / "balance_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
