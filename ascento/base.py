"""MuJoCo Playground-style MJX base environment."""
from pathlib import Path
from typing import Any
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import mujoco
from mujoco_playground._src import mjx_env
from . import actuator
from .constants import (
    BASE_HEIGHT, CTRL_DT, FALL_HEIGHT, JOINT_NAMES, N_ACTUATORS, N_SUBSTEPS,
    SIM_DT, STANCE, UPRIGHT_GRAVITY_Z_MIN,
)
OBS_SIZE = 46

class AscentoBaseEnv(mjx_env.MjxEnv):
    """Static-MJCF Ascento environment with direct six-motor effort actions."""
    def __init__(self, xml_path: str | None = None,
                 episode_length: int = 600) -> None:
        config = config_dict.create(
            ctrl_dt=CTRL_DT, sim_dt=SIM_DT, impl="jax",
            episode_length=episode_length,
        )
        super().__init__(config)
        self._xml_path = str(Path(xml_path or Path(__file__).parent.parent /
                                 "model" / "ascento_guard2_mjx.xml"))
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = SIM_DT
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._base_body_id = self._mj_model.body("base").id
        self._wheel_body_ids = jp.asarray([
            self._mj_model.body("left_wheel").id,
            self._mj_model.body("right_wheel").id,
        ])
        self._imu_site_id = self._mj_model.site("imu_reference").id
        self._joint_qpos = jp.arange(7, 13)
        self._joint_qvel = jp.arange(6, 12)
        self._nominal_qpos = jp.asarray(
            [0.0, 0.0, BASE_HEIGHT, 1.0, 0.0, 0.0, 0.0] +
            STANCE.tolist()
        )
        self._zero_action = jp.zeros((N_ACTUATORS,), dtype=jp.float32)

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

    def _reset_info(self) -> dict[str, Any]:
        return {
            "last_action": self._zero_action,
            "last_last_action": self._zero_action,
            "torque_state": self._zero_action,
            "last_torque": self._zero_action,
            "steps": jp.asarray(0, dtype=jp.int32),
        }

    def reset(self, rng: jax.Array) -> mjx_env.State:
        del rng
        data = mjx_env.make_data(
            self._mj_model, qpos=self._nominal_qpos,
            qvel=jp.zeros((self._mj_model.nv,)), ctrl=self._zero_action,
            device=jax.devices()[0],
        )
        data = mjx.forward(self._mjx_model, data)
        info = self._reset_info()
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.asarray(0.0), jp.asarray(0.0),
                             {}, info)

    def _projected_gravity(self, data: mjx.Data) -> jax.Array:
        return data.xmat[self._base_body_id].T @ jp.asarray([0.0, 0.0, -1.0])

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        gravity = self._projected_gravity(data)
        wheel_z = data.xpos[self._wheel_body_ids, 2]
        contacts = (wheel_z <= 0.255).astype(jp.float32)
        return jp.concatenate((
            gravity.astype(jp.float32),
            data.qvel[:6].astype(jp.float32),
            data.qpos[7:13].astype(jp.float32) - STANCE,
            data.qvel[6:12].astype(jp.float32),
            contacts,
            info["last_torque"].astype(jp.float32) / 40.0,
            info["last_action"].astype(jp.float32),
            info["last_last_action"].astype(jp.float32),
            jp.zeros((3,), dtype=jp.float32),
            jp.zeros((2,), dtype=jp.float32),
        ))

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        gravity = self._projected_gravity(data)
        # Height is the hard balance terminal; orientation remains a dense reward.
        # This avoids killing recovery trajectories on a transient attitude error.
        del gravity
        return data.qpos[2] < FALL_HEIGHT

    def _get_reward(self, data: mjx.Data, action: jax.Array,
                    info: dict[str, Any], done: jax.Array) -> jax.Array:
        del data, action, info, done
        return jp.asarray(0.0)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        data, torque_state, torque = actuator.rollout_substeps(
            state.data, state.info["torque_state"], action,
            self._mjx_model, N_SUBSTEPS,
        )
        done = self._get_termination(data).astype(jp.float32)
        info = dict(state.info)
        info.update(
            last_last_action=state.info["last_action"],
            last_action=action,
            torque_state=torque_state,
            last_torque=torque,
            steps=state.info["steps"] + 1,
        )
        reward = self._get_reward(data, action, info, done)
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward, done, {}, info)
