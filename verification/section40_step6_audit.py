"""Section 40 Step 4/6 audit under CUDA batched RL load."""
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("ASCENTO_JAX_PLATFORM", "cuda")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")

import jax
import jax.numpy as jp
import mujoco
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks
from mujoco_playground._src import wrapper

from ascento import actuator
from ascento.balance import AscentoBalance
from ascento.constants import (
    ACTUATOR_SPECS,
    LEG_INDEX,
    LEG_LIMIT_MARGIN,
    LEG_Q_MAX,
    LEG_Q_MIN,
    PEAK_TORQUE,
    WHEEL_INDEX,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path(os.environ.get(
    "ASCENTO_AUDIT_ARTIFACT",
    str(ROOT / "training" / "artifacts_cuda_smoke"),
))
def scalar(value):
    return float(jax.device_get(value))


def max_abs(value):
    return scalar(jp.max(jp.abs(value)))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def make_policy():
    with (ARTIFACT / "policy_params.pkl").open("rb") as handle:
        params = pickle.load(handle)
    manifest = json.loads(
        (ARTIFACT / "training_manifest.json").read_text()
    )
    hidden = tuple(manifest["network_hidden_layer_sizes"])
    net = networks.make_ppo_networks(
        46, 6, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden,
        value_hidden_layer_sizes=hidden,
        policy_obs_key="state", value_obs_key="state",
    )
    return networks.make_inference_fn(net)(params, deterministic=True), manifest
def check_model(env):
    model = env.mj_model
    require(model.nq == 13 and model.nv == 12 and model.nu == 6,
            "unexpected model dimensions")

    leg_ranges = []
    for name in ("left_hip", "left_knee", "right_hip", "right_knee"):
        joint_range = model.joint(name).range
        leg_ranges.append([float(joint_range[0]), float(joint_range[1])])
        require(jp.allclose(joint_range, jp.asarray([LEG_Q_MIN, LEG_Q_MAX]),
                             atol=1e-5),
                f"joint range mismatch: {name}")

    wheel_bodies = {
        model.body("left_wheel").id,
        model.body("right_wheel").id,
    }
    wheel_geoms = [
        i for i in range(model.ngeom)
        if int(model.geom_bodyid[i]) in wheel_bodies
    ]
    require(len(wheel_geoms) == 4, "expected four wheel collision geoms")
    for i in wheel_geoms:
        geom = model.geom(i)
        require(int(geom.condim[0]) == 6, "wheel condim is not 6")
        require(jp.allclose(geom.friction, jp.asarray([0.8, 0.02, 0.002]),
                             atol=1e-6),
                "wheel friction changed")
        require(jp.allclose(geom.solref, jp.asarray([0.006, 1.0]),
                             atol=1e-6),
                "wheel solref changed")
        require(jp.allclose(
            geom.solimp, jp.asarray([0.9, 0.97, 0.001, 0.5, 2.0]),
            atol=1e-6,
        ), "wheel solimp changed")

    expected_peak = jp.asarray([40.0, 40.0, 8.0, 40.0, 40.0, 8.0])
    require(jp.allclose(jp.asarray(PEAK_TORQUE), expected_peak),
            "leg/wheel peak torque indexing changed")
    require(tuple(LEG_INDEX.tolist()) == (0, 1, 3, 4),
            "leg actuator indices changed")
    require(tuple(WHEEL_INDEX.tolist()) == (2, 5),
            "wheel actuator indices changed")
    return {
        "dims": [model.nq, model.nv, model.nu],
        "leg_ranges": leg_ranges,
        "wheel_geom_count": len(wheel_geoms),
        "wheel_condim": 6,
        "wheel_friction": [0.8, 0.02, 0.002],
        "wheel_solref": [0.006, 1.0],
        "wheel_solimp": [0.9, 0.97, 0.001, 0.5, 2.0],
        "peak_torque": expected_peak.tolist(),
    }
def check_actuator_unit_properties():
    peak = jp.asarray(PEAK_TORQUE)
    zero_speed = actuator.torque_speed_envelope(
        peak, jp.zeros((6,), dtype=jp.float32)
    )
    require(jp.allclose(zero_speed, peak), "zero-speed saturation failed")

    half_speed = jp.asarray(
        [6.0, 6.0, 10.0, 6.0, 6.0, 10.0], dtype=jp.float32
    )
    motoring = actuator.torque_speed_envelope(peak, half_speed)
    require(jp.allclose(motoring, peak * 0.5, atol=1e-5),
            "motoring derating failed")

    braking = actuator.torque_speed_envelope(-peak, half_speed)
    require(jp.allclose(braking, peak, atol=1e-5),
            "braking-quadrant authority failed")

    return {
        "zero_speed_saturation": True,
        "motoring_derating": True,
        "braking_quadrant_full_authority": True,
    }


def trace_substeps(data, torque_state, action, mjx_model):
    """Return the exact pre-final-substep state and applied torque."""
    def one(carry, _):
        current_data, current_torque_state = carry
        pre_qvel = current_data.qvel[6:12]
        pre_qpos = current_data.qpos[7:13]
        next_data, next_torque_state, applied = actuator.substep(
            current_data, current_torque_state, action, mjx_model
        )
        return (next_data, next_torque_state), (
            pre_qvel, pre_qpos, next_torque_state, applied
        )
    (data, torque_state), trace = jax.lax.scan(
        one, (data, torque_state), None, length=5
    )
    pre_qvel = trace[0][-1]
    pre_qpos = trace[1][-1]
    filtered = trace[2][-1]
    applied = trace[3][-1]
    return data, pre_qvel, pre_qpos, filtered, applied


def main():
    require(jax.default_backend() == "gpu",
            f"wrong backend: {jax.default_backend()}")
    env = AscentoBalance()
    policy, manifest = make_policy()
    model_report = check_model(env)
    actuator_report = check_actuator_unit_properties()
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    single_step = jax.jit(env.step)
    train_env = wrapper.wrap_for_brax_training(
        env, episode_length=600, action_repeat=1, full_reset=False
    )
    train_reset = jax.jit(train_env.reset)
    train_step = jax.jit(train_env.step)
    batch_size = 32
    rollout_steps = 3000
    keys = jax.random.split(jax.random.PRNGKey(123), batch_size)
    state = reset(keys)
    state_single = env.reset(keys[0])
    probe_action = jp.linspace(-0.8, 0.8, 6, dtype=jp.float32)
    single_next = single_step(state_single, probe_action)
    batch_next = step(
        state,
        jp.broadcast_to(probe_action, (batch_size, 6)),
    )
    qpos_diff = max_abs(batch_next.data.qpos[0] - single_next.data.qpos)
    qvel_diff = max_abs(batch_next.data.qvel[0] - single_next.data.qvel)
    print('SINGLE_BATCH_DIFF', qpos_diff, qvel_diff)
    require(qpos_diff < 1e-3 and qvel_diff < 1e-2,
            "single-env and batched MJX trajectories diverged beyond tolerance")

    single_contact = single_next.data._impl.contact
    batch_contact = batch_next.data._impl.contact
    contact_diffs = {}
    for field in ("dim", "dist", "geom1", "geom2", "solref", "solimp"):
        lhs = getattr(batch_contact, field)[0]
        rhs = getattr(single_contact, field)
        contact_diffs[field] = max_abs(lhs - rhs)
    print('CONTACT_DIFFS', contact_diffs)
    print('CONTACT_DIM_SINGLE', jax.device_get(single_contact.dim))
    print('CONTACT_DIM_BATCH', jax.device_get(batch_contact.dim[0]))
    require(max(contact_diffs[field] for field in contact_diffs if field != 'dim') < 1e-5,
            "single-env and batched contact fields diverged")
    # Use the same Vmap/Episode/AutoReset path used by PPO training.
    state = train_reset(keys)

    # This is the exact saved PPO policy, evaluated for thousands of
    # batched policy steps (15,000 physics substeps).
    def rollout(carry, _):
        current, rng = carry
        rng, action_key = jax.random.split(rng)
        action, _ = policy(current.obs, action_key)
        previous_qvel = current.data.qvel[..., 6:12]
        previous_qpos = current.data.qpos[..., 7:13]
        _, raw_qvel, raw_qpos, raw_filtered, raw_applied = jax.vmap(
            trace_substeps, in_axes=(0, 0, 0, None)
        )(
            current.data, current.info["torque_state"], action,
            env.mjx_model,
        )
        nxt = train_step(current, action)
        return (nxt, rng), (
            action,
            previous_qvel,
            previous_qpos,
            nxt.data.qpos,
            nxt.data.qvel,
            raw_qvel,
            raw_qpos,
            raw_filtered,
            raw_applied,
            nxt.done,
        )

    (state, _), trajectory = jax.jit(
        lambda s: jax.lax.scan(rollout, s, None, length=rollout_steps)
    )((state, jax.random.PRNGKey(77)))
    (
        actions, prev_qvel, prev_qpos, qpos, qvel,
        raw_qvel, raw_qpos, filtered, applied, done
    ) = trajectory
    jax.block_until_ready(qpos)
    finite = all(
        bool(jax.device_get(jp.all(jp.isfinite(x))))
        for x in (actions, prev_qvel, prev_qpos, qpos, qvel,
                  raw_qvel, raw_qpos, filtered, applied)
    )
    require(finite, "non-finite value found during batched PPO rollout")

    leg_q = qpos[:, :, 7:13][:, :, [0, 1, 3, 4]]
    leg_min = scalar(jp.min(leg_q))
    leg_max = scalar(jp.max(leg_q))
    lower_ok = bool(jax.device_get(jp.all(leg_q >= LEG_Q_MIN - 1e-4)))
    upper_ok = bool(jax.device_get(jp.all(leg_q <= LEG_Q_MAX + 1e-4)))
    print('LEG_Q_RANGE', leg_min, leg_max, 'DONE', scalar(jp.sum(done)), 'OK', lower_ok and upper_ok)

    peak = jp.asarray(PEAK_TORQUE)
    applied_abs = jp.abs(applied)
    require(bool(jax.device_get(jp.all(applied_abs <= peak + 1e-4))),
            "actual PPO torque exceeded actuator peak")

    # Verify the braking branch using torque/velocity pairs produced by PPO.
    braking_mask = (filtered * raw_qvel) < -1e-5
    stop_mask = (
        ((raw_qpos <= LEG_Q_MIN + LEG_LIMIT_MARGIN) & (filtered < 0.0))
        | ((raw_qpos >= LEG_Q_MAX - LEG_LIMIT_MARGIN) & (filtered > 0.0))
    )
    valid_braking = braking_mask & ~stop_mask & (done[..., None] < 0.5)
    braking_count = int(jax.device_get(jp.sum(valid_braking)))
    braking_error = max_abs(
        jp.where(valid_braking, jp.abs(applied) - jp.abs(filtered), 0.0)
    )
    print('BRAKING_CHECK', braking_count, braking_error)
    require(braking_count > 0, "PPO rollout produced no braking-quadrant samples")
    require(braking_error < 1e-4,
            "actual PPO braking samples were derated incorrectly")
    # End-to-end no-delay check with an action from the saved PPO policy.
    state0 = env.reset(jax.random.PRNGKey(0))
    action0, _ = policy(state0.obs, jax.random.PRNGKey(1))
    next_data, _, applied0 = actuator.substep(
        state0.data, state0.info["torque_state"], action0, env.mjx_model
    )
    command_delay_error = max_abs(next_data.ctrl - applied0)
    require(command_delay_error < 1e-5,
            "data.ctrl did not receive the same-substep applied torque")

    report = {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "artifact_timesteps": manifest["timesteps"],
        "batch_size": batch_size,
        "rollout_policy_steps": rollout_steps,
        "rollout_physics_substeps": rollout_steps * 5,
        "finite_all_values": finite,
        "single_batch_qpos_max_abs_diff": qpos_diff,
        "single_batch_qvel_max_abs_diff": qvel_diff,
        "single_batch_contact_max_abs_diff": contact_diffs,
        "joint_limits_respected": lower_ok and upper_ok,
        "observed_leg_q_min": leg_min,
        "observed_leg_q_max": leg_max,
        "configured_leg_q_range": [float(LEG_Q_MIN), float(LEG_Q_MAX)],
        "actual_applied_torque_max_by_index": [
            scalar(jp.max(applied_abs[:, :, i])) for i in range(6)
        ],
        "braking_samples_from_policy": braking_count,
        "braking_authority_max_error": braking_error,
        "same_substep_ctrl_error": command_delay_error,
        "model": model_report,
        "actuator_unit_properties": actuator_report,
        "no_teacher": manifest["teacher"],
        "no_pd_seed": manifest["pd_seed"],
        "restore_params": manifest["restore_params"],
    }
    output = ARTIFACT / "section40_step6_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
