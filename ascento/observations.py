"""JAX-safe contact helpers and the stable Ascento actor observation."""
from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx

from .constants import NUM_JUMP_PHASES, PEAK_TORQUE, WHEEL_RADIUS


def wheel_contacts_and_forces(data: mjx.Data, wheel_body_ids: jax.Array):
    """Returns named wheel contact booleans and external-contact force magnitudes.

    ``cfrc_ext`` is produced by MuJoCo's contact solver and remains available in
    MJX batches.  The small geometry-height fallback makes a resting wheel
    consistently contact-positive before the first solver force is accumulated.
    """
    wheel_z = data.xpos[wheel_body_ids, 2]
    wrench = data._impl.cfrc_ext[wheel_body_ids, 3:]
    force = jp.linalg.norm(wrench, axis=1)
    contact = (force > 1e-3) | (wheel_z <= WHEEL_RADIUS + 3e-3)
    return contact, jp.where(contact, force, 0.0)


def non_wheel_contact(data: mjx.Data, non_wheel_body_ids: jax.Array):
    """Whether a non-wheel body has a material external-contact wrench."""
    wrench = data._impl.cfrc_ext[non_wheel_body_ids, 3:]
    return jp.any(jp.linalg.norm(wrench, axis=1) > 5.0)


def bounded_kinematics(data: mjx.Data, body_to_world: jax.Array):
    """Returns bounded, dimensionless actor kinematics.

    Raw MJX velocities are unbounded during a failed contact trajectory.  They
    are still used by the physics and failure checks, but an actor should never
    receive values large enough to saturate its policy network.
    """
    linear_velocity = jp.clip(body_to_world.T @ data.qvel[:3] / 5.0, -3.0, 3.0)
    # Free-joint rotational velocity is body-local in MuJoCo.  Applying the
    # base rotation here would transform it a second time.
    angular_velocity = jp.clip(data.qvel[3:6] / 10.0, -3.0, 3.0)
    joint_velocity = jp.clip(data.qvel[6:12] / 20.0, -3.0, 3.0)
    return linear_velocity, angular_velocity, joint_velocity


def make_observation(
    data: mjx.Data,
    info: dict,
    base_body_id: int,
    wheel_body_ids: jax.Array,
) -> jax.Array:
    """Builds the 49-value observation shared by every curriculum stage."""
    body_to_world = data.xmat[base_body_id]
    gravity = body_to_world.T @ jp.asarray([0.0, 0.0, -1.0])
    linear_velocity, angular_velocity, joint_velocity = bounded_kinematics(data, body_to_world)
    joint_q = data.qpos[7:13]
    leg_q = joint_q[jp.asarray([0, 1, 3, 4])]
    contact, force = wheel_contacts_and_forces(data, wheel_body_ids)
    phase = jax.nn.one_hot(info["jump_phase"], NUM_JUMP_PHASES, dtype=jp.float32)
    obs = jp.concatenate((
        gravity,
        linear_velocity,
        angular_velocity,
        jp.clip((data.qpos[2:3] - 0.75) / 0.25, -3.0, 3.0),
        leg_q,
        joint_velocity,
        contact.astype(jp.float32),
        jp.clip(force / 250.0, 0.0, 2.0),
        info["last_torque"] / jp.asarray(PEAK_TORQUE),
        info["last_action"],
        info["command"],
        phase,
        jp.asarray([info["phase_steps"]], dtype=jp.float32) / 100.0,
    )).astype(jp.float32)
    return obs
