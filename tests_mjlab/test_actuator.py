from types import SimpleNamespace

import mujoco
import torch
from mjlab.actuator import Actuator, ActuatorCmd

from ascento_mjlab.actuator import (
    AscentoTorqueActuator,
    AscentoTorqueActuatorCfg,
    torque_speed_limit,
)


def test_torque_speed_envelope_and_controller_guard():
    velocity = torch.tensor([[0.0, 6.0, 12.0, -11.0]])
    requested = torch.tensor([[40.0, 40.0, 40.0, -40.0]])
    result = torque_speed_limit(40.0, 12.0, 4.0, velocity, requested)

    assert torch.allclose(result[0, 0], torch.tensor(40.0))
    assert torch.allclose(result[0, 1], torch.tensor(0.0))
    assert torch.allclose(result[0, 2], torch.tensor(0.0))
    # Negative velocity and negative torque is motoring above the wheel/leg
    # controller limit, so it is guarded too.
    assert torch.allclose(result[0, 3], torch.tensor(0.0))


def test_braking_remains_available_above_controller_speed():
    velocity = torch.tensor([[6.0]])
    requested = torch.tensor([[-40.0]])
    result = torque_speed_limit(40.0, 12.0, 4.0, velocity, requested)
    assert result.item() < 0.0


def test_actuator_uses_active_model_timestep(monkeypatch):
    # The base initializer needs a full entity/model setup. It is not relevant
    # to this timing contract, so isolate the custom initializer here.
    monkeypatch.setattr(Actuator, "initialize", lambda self, *args: None)
    actuator = object.__new__(AscentoTorqueActuator)
    actuator.cfg = AscentoTorqueActuatorCfg(
        target_names_expr=("joint",),
        peak_torque=40.0,
        no_load_speed=12.0,
        controller_speed_limit=4.0,
        response_time=0.1,
    )
    actuator._target_ids = torch.tensor([0])
    model = mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.007"/></mujoco>')

    actuator.initialize(model, None, SimpleNamespace(nworld=1), "cpu")

    assert actuator._physics_dt == 0.007


def test_actuator_response_uses_initialized_timestep():
    actuator = object.__new__(AscentoTorqueActuator)
    actuator.cfg = AscentoTorqueActuatorCfg(
        target_names_expr=("joint",),
        peak_torque=40.0,
        no_load_speed=12.0,
        controller_speed_limit=4.0,
        response_time=0.1,
    )
    actuator._physics_dt = 0.02
    actuator._filtered = torch.zeros((1, 1))
    cmd = ActuatorCmd(
        position_target=torch.zeros((1, 1)),
        velocity_target=torch.zeros((1, 1)),
        effort_target=torch.full((1, 1), 10.0),
        pos=torch.zeros((1, 1)),
        vel=torch.zeros((1, 1)),
    )

    result = actuator.compute(cmd)

    expected = 10.0 * (1.0 - torch.exp(torch.tensor(-0.02 / 0.1)))
    assert torch.allclose(result, expected.reshape(1, 1))
