import jax
import numpy as np
import pytest

from brax.training.acme import running_statistics

from training.ppo_config import network_factory


@pytest.mark.parametrize("hidden_sizes", [(16,), (16, 16), (16, 16, 16), (16, 16, 16, 16)])
def test_network_factory_initializes_the_policy_output_for_any_depth(hidden_sizes):
    action_size = 6
    network = network_factory(hidden_sizes, initial_noise_std=0.10)(
        10, action_size, running_statistics.normalize
    )

    params = network.policy_network.init(jax.random.PRNGKey(0))["params"]
    output_key = f"hidden_{len(hidden_sizes)}"
    output = params[output_key]

    np.testing.assert_allclose(output["kernel"], 0.0)
    np.testing.assert_allclose(output["bias"][:action_size], 0.0)
    assert np.all(np.isfinite(np.asarray(output["bias"][action_size:])))
    for index in range(len(hidden_sizes)):
        assert np.any(np.asarray(params[f"hidden_{index}"]["kernel"]) != 0.0)
