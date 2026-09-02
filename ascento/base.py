"""Shared MJX environment for direct-torque Ascento curriculum tasks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

from . import actuator, commands, jump, observations, rewards
from .constants import (
    BASE_HEIGHT,
    CTRL_DT,
    FALL_GRAVITY_Z_MAX,
    FALL_HEIGHT,
    MAX_BASE_ANGULAR_VELOCITY,
    MAX_BASE_LINEAR_VELOCITY,
    MAX_JOINT_VELOCITY,
    N_ACTUATORS,
    N_SUBSTEPS,
    OBS_SIZE,
    SIM_DT,
    STANCE,
)

_REWARD_KEYS = (
    "upright", "vx_tracking", "yaw_tracking", "height", "posture", "stable",
    "action_rate", "action_smooth", "action_magnitude", "lateral_drift", "vertical_velocity", "angular_velocity",
    "joint_limit", "torque_saturation", "collision",
    "jump_crouch", "jump_thrust", "jump_height", "jump_clearance", "jump_landing",
    "jump_recovery", "jump_failure",
)


class AscentoBaseEnv(mjx_env.MjxEnv):
    """A fixed-schema, side-effect-free MJX environment with six torque actions."""

    def __init__(
        self,
        xml_path: str | None = None,
        *,
        episode_length: int = 600,
        max_vx: float = 0.0,
        max_yaw_rate: float = 0.0,
        height_range: float = 0.0,
        reset_tilt: float = 0.08,
        reset_angular_velocity: float = 0.30,
        reset_linear_velocity: float = 0.20,
        reset_leg_variation: float = 0.05,
        reset_wheel_velocity: float = 2.0,
        jump_probability: float = 0.0,
        max_jump_height: float = 0.0,
        max_jump_distance: float = 0.0,
        enable_jump_rewards: bool = False,
        action_scale: float = 1.0,
    ) -> None:
        config = config_dict.create(ctrl_dt=CTRL_DT, sim_dt=SIM_DT, impl="jax", episode_length=episode_length)
        super().__init__(config)
        self._xml_path = str(Path(xml_path or Path(__file__).parent.parent / "model" / "ascento_guard2_mjx.xml"))
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = SIM_DT
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._base_body_id = self._mj_model.body("base").id
        self._wheel_body_ids = jp.asarray((self._mj_model.body("left_wheel").id, self._mj_model.body("right_wheel").id))
        wheel_ids = tuple(self._wheel_body_ids.tolist())
        self._non_wheel_body_ids = jp.asarray(tuple(body_id for body_id in range(1, self._mj_model.nbody) if body_id not in wheel_ids))
        self._nominal_qpos = jp.asarray([0.0, 0.0, BASE_HEIGHT, 1.0, 0.0, 0.0, 0.0] + STANCE.tolist())
        self._zero_action = jp.zeros((N_ACTUATORS,), dtype=jp.float32)
        self._max_vx, self._max_yaw_rate, self._height_range = float(max_vx), float(max_yaw_rate), float(height_range)
        self._reset_tilt = float(reset_tilt)
        self._reset_angular_velocity, self._reset_linear_velocity = float(reset_angular_velocity), float(reset_linear_velocity)
        self._reset_leg_variation = float(reset_leg_variation)
        self._reset_wheel_velocity = float(reset_wheel_velocity)
        self._jump_probability, self._max_jump_height, self._max_jump_distance = float(jump_probability), float(max_jump_height), float(max_jump_distance)
        self._enable_jump_rewards = bool(enable_jump_rewards)
        self._action_scale = float(action_scale)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return N_ACTUATORS

    @property
    def observation_size(self) -> int:
        return OBS_SIZE

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def model_assets(self) -> dict[str, bytes]:
        return {}

    @property
    def nominal_qpos(self) -> jax.Array:
        return self._nominal_qpos

    def _sample_initial_state(self, rng: jax.Array):
        """Samples task difficulty, never sensor noise or command latency."""
        key_roll, key_pitch, key_pose, key_linear, key_angular, key_wheels = jax.random.split(rng, 6)
        roll = jax.random.uniform(key_roll, (), minval=-self._reset_tilt, maxval=self._reset_tilt)
        pitch = jax.random.uniform(key_pitch, (), minval=-self._reset_tilt, maxval=self._reset_tilt)
        cr, sr, cp, sp = jp.cos(roll / 2.0), jp.sin(roll / 2.0), jp.cos(pitch / 2.0), jp.sin(pitch / 2.0)
        qpos = self._nominal_qpos.at[3:7].set(jp.asarray([cr * cp, sr * cp, cr * sp, sr * sp]))
        leg_delta = jax.random.uniform(key_pose, (4,), minval=-self._reset_leg_variation, maxval=self._reset_leg_variation)
        qpos = qpos.at[jp.asarray([7, 8, 10, 11])].add(leg_delta)
        qvel = jp.zeros((self._mj_model.nv,), dtype=jp.float32)
        qvel = qvel.at[:3].set(jax.random.uniform(key_linear, (3,), minval=-self._reset_linear_velocity, maxval=self._reset_linear_velocity))
        qvel = qvel.at[3:6].set(jax.random.uniform(key_angular, (3,), minval=-self._reset_angular_velocity, maxval=self._reset_angular_velocity))
        qvel = qvel.at[jp.asarray([8, 11])].set(
            jax.random.uniform(
                key_wheels,
                (2,),
                minval=-self._reset_wheel_velocity,
                maxval=self._reset_wheel_velocity,
            )
        )
        return qpos, qvel

    def _reset_info(self, rng: jax.Array) -> dict[str, Any]:
        rng, command_key = jax.random.split(rng)
        info = {
            "rng": rng,
            "last_action": self._zero_action,
            "last_last_action": self._zero_action,
            "torque_state": self._zero_action,
            "last_torque": self._zero_action,
            "wheel_contact": jp.zeros((2,), dtype=jp.bool_),
            "wheel_contact_force": jp.zeros((2,), dtype=jp.float32),
            "steps": jp.asarray(0, dtype=jp.int32),
            "command": commands.sample_command(
                command_key, max_vx=self._max_vx, max_yaw_rate=self._max_yaw_rate,
                height_range=self._height_range, nominal_height=BASE_HEIGHT,
                jump_probability=self._jump_probability, max_jump_height=self._max_jump_height,
                max_jump_distance=self._max_jump_distance,
            ),
        }
        return dict(info, **jump.initial_jump_info())

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, state_key = jax.random.split(rng)
        qpos, qvel = self._sample_initial_state(state_key)
        data = mjx_env.make_data(self._mj_model, qpos=qpos, qvel=qvel, ctrl=self._zero_action, device=jax.devices()[0])
        data = mjx.forward(self._mjx_model, data)
        info = self._reset_info(rng)
        metrics = {f"reward/{key}": jp.asarray(0.0) for key in _REWARD_KEYS}
        metrics.update({
            "metric/body_height": data.qpos[2],
            "metric/gravity_z": jp.asarray(-1.0),
            "metric/max_abs_qvel": jp.max(jp.abs(data.qvel)),
            "metric/phase": jp.asarray(0.0),
            "metric/wheel_contact": jp.asarray(0.0),
        })
        return mjx_env.State(data, self._get_obs(data, info), jp.asarray(0.0), jp.asarray(0.0), metrics, info)

    def _dynamics_signals(self, data: mjx.Data):
        body_to_world = data.xmat[self._base_body_id]
        gravity = body_to_world.T @ jp.asarray([0.0, 0.0, -1.0])
        local_linear_velocity = body_to_world.T @ data.qvel[:3]
        # MuJoCo free-joint angular velocity is already expressed in the
        # local body frame; only the translational velocity is world-frame.
        local_angular_velocity = data.qvel[3:6]
        wheel_contact, wheel_force = observations.wheel_contacts_and_forces(data, self._wheel_body_ids)
        nonwheel_collision = observations.non_wheel_contact(data, self._non_wheel_body_ids)
        return gravity, local_linear_velocity, local_angular_velocity, wheel_contact, wheel_force, nonwheel_collision

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        return observations.make_observation(data, info, self._base_body_id, self._wheel_body_ids)

    def _get_reward_terms(self, data, action, info, gravity, local_linear_velocity, local_angular_velocity, wheel_contact, nonwheel_collision):
        terms = rewards.base_terms(data, action, info, gravity, local_linear_velocity, local_angular_velocity, nonwheel_collision)
        if self._enable_jump_rewards:
            terms.update(rewards.jump_terms(data, info, gravity, wheel_contact))
        else:
            terms.update({key: jp.asarray(0.0) for key in _REWARD_KEYS if key.startswith("jump_")})
        return terms

    def _get_termination(
        self, data: mjx.Data, gravity: jax.Array, nonwheel_collision: jax.Array
    ) -> jax.Array:
        """Terminates physically failed and numerically unsafe trajectories."""
        finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        excessive_velocity = (
            (jp.linalg.norm(data.qvel[:3]) > MAX_BASE_LINEAR_VELOCITY)
            | (jp.linalg.norm(data.qvel[3:6]) > MAX_BASE_ANGULAR_VELOCITY)
            | jp.any(jp.abs(data.qvel[6:12]) > MAX_JOINT_VELOCITY)
        )
        fallen = (data.qpos[2] < FALL_HEIGHT) | (gravity[2] > FALL_GRAVITY_Z_MAX)
        return fallen | nonwheel_collision | excessive_velocity | ~finite

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Balance begins with a deliberately reduced direct-torque envelope.
        # The policy still has a fixed [-1, 1] action interface, while later
        # curriculum environments can expose the full actuator authority.
        action = jp.clip(action, -1.0, 1.0) * self._action_scale
        data, torque_state, torque = actuator.rollout_substeps(state.data, state.info["torque_state"], action, self._mjx_model, N_SUBSTEPS)
        gravity, local_linear_velocity, local_angular_velocity, wheel_contact, wheel_force, nonwheel_collision = self._dynamics_signals(data)
        info = dict(
            state.info, last_last_action=state.info["last_action"], last_action=action,
            torque_state=torque_state, last_torque=torque, steps=state.info["steps"] + 1,
            wheel_contact=wheel_contact, wheel_contact_force=wheel_force,
        )
        info = jump.update_jump_info(
            info, data, wheel_contact, gravity, data.xpos[self._wheel_body_ids, 2]
        )
        # Reward the transition against the *previous* action history.  ``info``
        # already stores the newly applied action for the next observation, so
        # passing it directly here would make the rate term identically zero.
        reward_info = dict(
            info,
            last_action=state.info["last_action"],
            last_last_action=state.info["last_last_action"],
        )
        terms = self._get_reward_terms(data, action, reward_info, gravity, local_linear_velocity, local_angular_velocity, wheel_contact, nonwheel_collision)
        done = self._get_termination(data, gravity, nonwheel_collision).astype(jp.float32)
        reward = sum(terms.values()) + jp.where(done > 0, -10.0, 0.0)
        metrics = dict(state.metrics)
        metrics.update({f"reward/{key}": value for key, value in terms.items()})
        metrics.update({
            "metric/body_height": data.qpos[2],
            "metric/gravity_z": gravity[2],
            "metric/max_abs_qvel": jp.max(jp.abs(data.qvel)),
            "metric/phase": info["jump_phase"].astype(jp.float32),
            "metric/wheel_contact": jp.mean(wheel_contact.astype(jp.float32)),
        })
        return mjx_env.State(data, self._get_obs(data, info), reward, done, metrics, info)


AscentoEnv = AscentoBaseEnv
