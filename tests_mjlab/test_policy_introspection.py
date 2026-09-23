from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from ascento_mjlab.viewer.introspection import PolicyIntrospector
from ascento_mjlab.viewer.introspection_contract import (
    build_policy_branch_schema,
    resolve_runtime_handles,
)


class _ScaleNormalizer(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - 2.0) * 0.5


class _Gaussian(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.std_type = "scalar"
        self.std_param = nn.Parameter(torch.full((output_dim,), 0.25))
        self._std = None

    def update(self, mean: torch.Tensor) -> None:
        self._std = self.std_param.clamp(1e-6, 1e6).expand_as(mean)

    @property
    def std(self) -> torch.Tensor:
        return self._std


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_groups = {"actor": ("actor",)}
        self.obs_normalizer = _ScaleNormalizer()
        self.mlp = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
        self.distribution = _Gaussian(2)

    def forward(self, observations, *, stochastic_output: bool = False):
        del stochastic_output
        return self.mlp(self.obs_normalizer(observations["actor"]))


class _Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_groups = {"critic": ("critic",)}
        self.obs_normalizer = nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(4, 3), nn.ELU(), nn.Linear(3, 1))

    def forward(self, observations):
        return self.mlp(self.obs_normalizer(observations["critic"]))


def _runner():
    actor = _Actor()
    critic = _Critic()

    class _Algorithm:
        def get_policy(self):
            return actor

    return SimpleNamespace(alg=_Algorithm()), actor, critic


def test_runtime_handles_resolve_actor_critic_normalizers_distribution_and_dimensions():
    runner, actor, critic = _runner()
    runner.alg.critic = critic

    handles = resolve_runtime_handles(runner)

    assert handles.actor is actor
    assert handles.critic is critic
    assert handles.actor_normalizer is actor.obs_normalizer
    assert handles.critic_normalizer is critic.obs_normalizer
    assert handles.distribution is actor.distribution
    assert handles.actor_obs_groups == ("actor",)
    assert handles.critic_obs_groups == ("critic",)
    assert handles.actor_input_dim == 3
    assert handles.critic_input_dim == 4
    assert handles.action_dim == 2


def test_introspection_observes_one_authoritative_forward_and_exact_model_values():
    torch.manual_seed(7)
    runner, actor, critic = _runner()
    runner.alg.critic = critic
    handles = resolve_runtime_handles(runner)
    observations = {
        "actor": torch.tensor([[2.0, 4.0, 0.0]]),
        "critic": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    }
    expected = actor(observations, stochastic_output=False).detach().clone()
    introspector = PolicyIntrospector(handles, checkpoint="model_17.pt")
    call_count = 0

    def authoritative_forward():
        nonlocal call_count
        call_count += 1
        return actor(observations, stochastic_output=False)

    capture = introspector.capture_action(
        observations,
        authoritative_forward,
        sequence_id=17,
    )
    frame = capture.frame

    assert call_count == 1
    assert torch.equal(capture.action, expected)
    assert torch.equal(frame.actor_output, expected[0])
    assert torch.equal(frame.raw_observation, observations["actor"][0])
    assert torch.equal(frame.normalized_observation, torch.tensor([0.0, 1.0, -1.0]))
    assert torch.isfinite(torch.tensor(frame.critic_value))
    assert frame.policy_std.shape == (2,)
    assert set(frame.hidden_activations) == {"actor.mlp.1", "critic.mlp.1"}
    assert frame.sequence_id == 17
    assert frame.policy_generation == 1
    assert frame.to_dict()["policy_generation"] == 1

    introspector.close()


def test_runtime_policy_branch_schema_tracks_exact_loaded_layer_shapes_and_statistics():
    _runner_instance, actor, _critic = _runner()
    schema = build_policy_branch_schema(
        actor,
        actor.mlp,
        distribution=actor.distribution,
    )

    assert schema["layers"] == [3, 4, 2]
    assert schema["activation"] == "TANH"
    assert schema["distribution"] == "Gaussian"
    assert schema["std_parameters"] == 2
    assert schema["parameter_count"] == sum(parameter.numel() for parameter in actor.parameters())
    assert schema["linear_layers"][0]["name"] == "0"
    assert schema["linear_layers"][0]["input_size"] == 3
    assert schema["linear_layers"][1]["output_size"] == 2
    assert all(torch.isfinite(torch.tensor(layer["weight_rms"])) for layer in schema["linear_layers"])
