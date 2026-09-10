from types import SimpleNamespace

import torch

from ascento_mjlab.evaluation.policy import RslRlPolicyAdapter


class _Actor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, observations, *, stochastic_output: bool):
        del stochastic_output
        return self.linear(observations)


def test_evaluation_policy_act_disables_autograd_graph_retention():
    actor = _Actor()
    algorithm = SimpleNamespace(get_policy=lambda: actor, eval_mode=lambda: None)
    adapter = RslRlPolicyAdapter(
        SimpleNamespace(alg=algorithm, device="cpu"), "checkpoint.pt", deterministic=True
    )

    action = adapter.act(torch.ones((3, 2)))

    assert not action.requires_grad
