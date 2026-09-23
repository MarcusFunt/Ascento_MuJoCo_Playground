from __future__ import annotations

import sys
import types

import torch
from torch import nn

import ascento_mjlab.viewer.attribution as attribution
from ascento_mjlab.viewer.introspection_contract import PolicyRuntimeHandles


class CenteringNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_mean", torch.tensor([[10.0, 20.0]]))

    @property
    def mean(self):
        return self._mean.squeeze(0)

    def forward(self, values):
        return values - self._mean


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.mlp.weight.copy_(torch.tensor([[1.0, 1.0], [3.0, -1.0]]))


def test_normalizer_mean_baseline_maps_to_zero_normalized_observation():
    model = Model()
    normalizer = CenteringNormalizer()
    handles = PolicyRuntimeHandles(
        actor=model,
        critic=None,
        actor_normalizer=normalizer,
        critic_normalizer=None,
        distribution=None,
        actor_body=model.mlp,
        critic_body=None,
        actor_body_name="mlp",
        critic_body_name=None,
        actor_obs_groups=("actor",),
        critic_obs_groups=(),
        actor_input_dim=2,
        critic_input_dim=None,
        action_dim=2,
    )

    baseline = attribution.normalizer_mean_baseline(handles)

    assert baseline.tolist() == [10.0, 20.0]
    assert normalizer(baseline.unsqueeze(0)).tolist() == [[0.0, 0.0]]


def test_integrated_gradients_uses_selected_deterministic_action_and_reports_delta(monkeypatch):
    model = Model()
    normalizer = CenteringNormalizer()
    handles = PolicyRuntimeHandles(
        actor=model,
        critic=None,
        actor_normalizer=normalizer,
        critic_normalizer=None,
        distribution=None,
        actor_body=model.mlp,
        critic_body=None,
        actor_body_name="mlp",
        critic_body_name=None,
        actor_obs_groups=("actor",),
        critic_obs_groups=(),
        actor_input_dim=2,
        critic_input_dim=None,
        action_dim=2,
    )

    class FakeIntegratedGradients:
        def __init__(self, forward):
            self.forward = forward

        def attribute(self, inputs, *, baselines, n_steps, return_convergence_delta):
            assert n_steps == 32 and return_convergence_delta is True
            grads = model.mlp.weight[1].detach().unsqueeze(0)
            values = (inputs - baselines) * grads
            delta = self.forward(inputs) - self.forward(baselines) - values.sum(dim=-1)
            return values, delta

    monkeypatch.setattr(attribution, "find_spec", lambda _name: object())
    fake_captum = types.ModuleType("captum")
    fake_attr = types.ModuleType("captum.attr")
    fake_attr.IntegratedGradients = FakeIntegratedGradients
    fake_captum.attr = fake_attr
    monkeypatch.setitem(sys.modules, "captum", fake_captum)
    monkeypatch.setitem(sys.modules, "captum.attr", fake_attr)
    model.train()

    result = attribution.explain_integrated_gradients(
        handles,
        explanation_id="explain-1",
        input_raw=[12.0, 23.0],
        baseline_raw=[10.0, 20.0],
        action_index=1,
        action_name="right_wheel",
        baseline_kind="normalizer_mean",
        baseline_description="raw running mean; maps to normalized zero",
        n_steps=32,
    )

    assert result.attribution == [6.0, -3.0]
    assert abs(result.absolute_fraction[0] - 2 / 3) < 1e-6
    assert abs(result.absolute_fraction[1] - 1 / 3) < 1e-6
    assert abs(result.convergence_delta) < 1e-6
    assert result.action_name == "right_wheel"
    assert model.training is True
    assert result.to_dict()["method"] == "integrated_gradients"
