"""Tools for measuring observation-normalizer drift in transferred policies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def observation_term_slice(
    term_names: Sequence[str],
    term_dimensions: Sequence[Sequence[int]],
    term_name: str,
) -> tuple[int, int]:
    """Return the flattened [start, stop) interval for a concatenated observation term."""
    if len(term_names) != len(term_dimensions):
        raise ValueError("observation term names and dimensions must have equal lengths")
    if term_name not in term_names:
        raise ValueError(f"observation term {term_name!r} was not found")
    offset = 0
    for name, dimensions in zip(term_names, term_dimensions, strict=True):
        size = math.prod(dimensions)
        if name == term_name:
            return offset, offset + size
        offset += size
    raise AssertionError("requested observation term was not resolved")


def target_error_normalizer_summary(
    checkpoint: Mapping[str, Any],
    target_error_slice: tuple[int, int],
) -> dict[str, Any]:
    """Extract the actor's target-error observation mean, scale, and sample count."""
    actor_state = checkpoint.get("actor_state_dict")
    if not isinstance(actor_state, Mapping):
        raise ValueError("checkpoint has no actor_state_dict")
    mean = actor_state.get("obs_normalizer._mean")
    scale = actor_state.get("obs_normalizer._std")
    if scale is None:
        variance = actor_state.get("obs_normalizer._var")
        if variance is not None:
            scale = torch.sqrt(torch.clamp(variance, min=0.0))
    if mean is None or scale is None:
        raise ValueError("actor checkpoint lacks observation-normalizer mean or scale")
    start, stop = target_error_slice
    mean = torch.as_tensor(mean).detach().float().flatten()
    scale = torch.as_tensor(scale).detach().float().flatten()
    if start < 0 or stop <= start or stop > mean.numel() or stop > scale.numel():
        raise ValueError("target-error observation slice is outside the normalizer state")
    count = actor_state.get("obs_normalizer.count")
    return {
        "mean": mean[start:stop].tolist(),
        "scale": scale[start:stop].tolist(),
        "count": None if count is None else float(torch.as_tensor(count).item()),
        "indices": [start, stop],
    }


def compare_target_error_normalizers(
    reference_checkpoint: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
    target_error_slice: tuple[int, int],
) -> dict[str, Any]:
    """Compare adaptive checkpoint statistics with the frozen transfer statistics."""
    reference = target_error_normalizer_summary(reference_checkpoint, target_error_slice)
    candidate = target_error_normalizer_summary(candidate_checkpoint, target_error_slice)
    reference_scale = torch.tensor(reference["scale"])
    candidate_scale = torch.tensor(candidate["scale"])
    return {
        "reference": reference,
        "candidate": candidate,
        "mean_delta": (torch.tensor(candidate["mean"]) - torch.tensor(reference["mean"])).tolist(),
        "scale_ratio": (candidate_scale / reference_scale.clamp_min(1.0e-12)).tolist(),
    }
