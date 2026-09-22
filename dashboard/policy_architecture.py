"""Read the layer dimensions recorded in a saved PPO policy checkpoint."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_LINEAR_WEIGHT = re.compile(r"^mlp\.(\d+)\.weight$")


def _linear_sizes(state: dict[str, Any]) -> list[int]:
    weights: list[tuple[int, Any]] = []
    for key, value in state.items():
        match = _LINEAR_WEIGHT.fullmatch(key)
        if match:
            weights.append((int(match.group(1)), value))

    weights.sort(key=lambda item: item[0])
    if not weights:
        return []

    sizes: list[int] = []
    for _, weight in weights:
        shape = getattr(weight, "shape", ())
        if len(shape) != 2:
            return []
        output_size, input_size = (int(shape[0]), int(shape[1]))
        if sizes and sizes[-1] != input_size:
            return []
        if not sizes:
            sizes.append(input_size)
        sizes.append(output_size)
    return sizes


def _activation_names(agent_config_path: Path) -> tuple[str, str]:
    if not agent_config_path.is_file():
        return "Unknown", "Unknown"
    try:
        text = agent_config_path.read_text(encoding="utf-8").replace("!!python/tuple", "")
        config = yaml.safe_load(text)
    except (OSError, yaml.YAMLError):
        return "Unknown", "Unknown"
    if not isinstance(config, dict):
        return "Unknown", "Unknown"

    def read(name: str) -> str:
        branch = config.get(name)
        activation = branch.get("activation") if isinstance(branch, dict) else None
        return str(activation).upper() if activation else "Unknown"

    return read("actor"), read("critic")


def inspect_policy_checkpoint(checkpoint_path: Path, relative_path: str) -> dict[str, Any]:
    """Return a compact, checkpoint-backed actor/critic layer description."""
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise ValueError("The checkpoint could not be read safely.") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint does not contain a policy state.")

    actor_state = checkpoint.get("actor_state_dict")
    critic_state = checkpoint.get("critic_state_dict")
    if not isinstance(actor_state, dict) or not isinstance(critic_state, dict):
        raise ValueError("The checkpoint does not contain both actor and critic policies.")

    actor_sizes = _linear_sizes(actor_state)
    critic_sizes = _linear_sizes(critic_state)
    if len(actor_sizes) < 2 or len(critic_sizes) < 2:
        raise ValueError("The checkpoint network layers are not a supported MLP.")

    actor_activation, critic_activation = _activation_names(
        checkpoint_path.parent / "params" / "agent.yaml"
    )
    distribution = "Gaussian" if "distribution.std_param" in actor_state else "Policy"
    std_parameter = actor_state.get("distribution.std_param")
    std_parameters = int(std_parameter.numel()) if hasattr(std_parameter, "numel") else 0

    iteration = checkpoint.get("iter")
    if not isinstance(iteration, int):
        match = re.search(r"(?:model|checkpoint)[_-]?(\d+)", checkpoint_path.stem, re.IGNORECASE)
        iteration = int(match.group(1)) if match else None

    return {
        "available": True,
        "checkpoint": relative_path,
        "iteration": iteration,
        "actor": {
            "layers": actor_sizes,
            "activation": actor_activation,
            "distribution": distribution,
            "std_parameters": std_parameters,
        },
        "critic": {
            "layers": critic_sizes,
            "activation": critic_activation,
        },
    }
