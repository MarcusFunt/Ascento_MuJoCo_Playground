"""Versioned deterministic anchor observation and teacher-action replay data."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .evaluation.schema import ScenarioSpec

_REPLAY_SCHEMA_VERSION = 1
_REQUIRED_METADATA = (
    "source_checkpoint_sha256",
    "source_actor_normalizer_sha256",
    "source_task_contract_sha256",
    "source_action_contract_sha256",
    "source_plant_contract_sha256",
    "actor_observation_contract_sha256",
    "suite_id",
    "suite_definition_sha256",
    "resolved_scenario_sha256",
)
_PRECISION_ANCHOR = 0
_RECOVERY_RETARGET_ANCHOR = 1


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible contract using canonical key and number ordering."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tensor_mapping_sha256(values: Mapping[str, Any]) -> str:
    """Hash tensor state by key, dtype, shape, and contiguous CPU bytes."""
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def actor_normalizer_sha256(actor_state: Mapping[str, Any]) -> str:
    """Hash only the frozen empirical-normalizer tensors in an actor state dict."""
    normalizer = {
        name: value for name, value in actor_state.items() if name.startswith("obs_normalizer.")
    }
    if not normalizer:
        raise ValueError("actor state has no observation-normalizer state")
    return tensor_mapping_sha256(normalizer)


def anchor_replay_arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash replay arrays independently of ZIP timestamps or compression details."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if value.dtype.hasobject:
            raise ValueError(f"replay array {name!r} must not use object dtype")
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        if value.dtype.kind in {"U", "S"}:
            canonical = json.dumps(
                value.astype(str).tolist(), ensure_ascii=False, separators=(",", ":")
            )
            digest.update(canonical.encode("utf-8"))
        else:
            digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class AnchorReplayDataset:
    """Frozen actor observations and deterministic teacher mean-action labels."""

    observations: torch.Tensor
    teacher_actions: torch.Tensor
    cohort_ids: torch.Tensor
    scenario_ids: tuple[str, ...]
    attempt_ids: torch.Tensor
    steps: torch.Tensor
    metadata: dict[str, Any]

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a balanced precision/recovery sample, drawn with replacement."""
        if batch_size < 2:
            raise ValueError("reference replay batch_size must be at least 2")
        precision = torch.nonzero(self.cohort_ids == _PRECISION_ANCHOR, as_tuple=False).flatten()
        recovery = torch.nonzero(
            self.cohort_ids == _RECOVERY_RETARGET_ANCHOR, as_tuple=False
        ).flatten()
        if precision.numel() == 0 or recovery.numel() == 0:
            raise ValueError("reference replay must contain both anchor cohorts")
        precision_count = batch_size // 2
        recovery_count = batch_size - precision_count
        precision_draw = precision[
            torch.randint(precision.numel(), (precision_count,), generator=generator)
        ]
        recovery_draw = recovery[
            torch.randint(recovery.numel(), (recovery_count,), generator=generator)
        ]
        indices = torch.cat((precision_draw, recovery_draw))
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        target_device = torch.device(device)
        return (
            self.observations[indices].to(target_device),
            self.teacher_actions[indices].to(target_device),
        )


class AnchorReplayRecorder:
    """Collect raw actor inputs and deterministic teacher actions on fixed scenarios."""

    def __init__(self, *, stride_steps: int = 10) -> None:
        if isinstance(stride_steps, bool) or stride_steps < 1:
            raise ValueError("stride_steps must be a positive integer")
        self.stride_steps = int(stride_steps)
        self._observations: list[np.ndarray] = []
        self._teacher_actions: list[np.ndarray] = []
        self._cohort_ids: list[int] = []
        self._scenario_ids: list[str] = []
        self._attempt_ids: list[int] = []
        self._steps: list[int] = []
        self._seen: set[tuple[str, int]] = set()
        self.observation_dim: int | None = None
        self.action_dim: int | None = None

    def record(
        self,
        *,
        observations: torch.Tensor,
        teacher_actions: torch.Tensor,
        scenarios: list[ScenarioSpec],
        step: int,
        active: torch.Tensor,
    ) -> None:
        """Record active environments at a fixed cadence with target-attempt identity."""
        if isinstance(step, bool) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if step % self.stride_steps:
            return
        if observations.ndim != 2 or teacher_actions.ndim != 2:
            raise ValueError("observations and teacher_actions must be rank-2 batches")
        if observations.shape[0] != len(scenarios) or teacher_actions.shape[0] != len(scenarios):
            raise ValueError("scenario count must match the observation and action batch")
        active_cpu = torch.as_tensor(active, dtype=torch.bool).detach().cpu().flatten()
        if active_cpu.numel() != len(scenarios):
            raise ValueError("active mask must contain one value per scenario")
        obs_cpu = observations.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions_cpu = teacher_actions.detach().to(dtype=torch.float32, device="cpu").numpy()
        if not np.isfinite(obs_cpu).all() or not np.isfinite(actions_cpu).all():
            raise ValueError("replay observations and teacher actions must be finite")
        if self.observation_dim is None:
            self.observation_dim = int(obs_cpu.shape[1])
            self.action_dim = int(actions_cpu.shape[1])
        elif self.observation_dim != obs_cpu.shape[1] or self.action_dim != actions_cpu.shape[1]:
            raise ValueError("replay observation/action dimensions changed during collection")

        for index in torch.nonzero(active_cpu, as_tuple=False).flatten().tolist():
            scenario = scenarios[index]
            dedupe_key = (scenario.scenario_id, int(step))
            if dedupe_key in self._seen:
                continue
            self._seen.add(dedupe_key)
            attempt_id = sum(
                command.name == "world_target_offset" and command.step <= step
                for command in scenario.commands
            )
            disturbances_started = [
                disturbance.start_step
                for disturbance in scenario.disturbances
                if disturbance.start_step <= step
            ]
            cohort_id = _RECOVERY_RETARGET_ANCHOR if disturbances_started else _PRECISION_ANCHOR
            self._observations.append(obs_cpu[index].copy())
            self._teacher_actions.append(actions_cpu[index].copy())
            self._cohort_ids.append(cohort_id)
            self._scenario_ids.append(scenario.scenario_id)
            self._attempt_ids.append(attempt_id)
            self._steps.append(int(step))

    def arrays(self) -> dict[str, np.ndarray]:
        """Return validated, non-object arrays for serialization."""
        if not self._observations or self.observation_dim is None or self.action_dim is None:
            raise ValueError("cannot serialize an empty anchor replay")
        return {
            "observations": np.asarray(self._observations, dtype=np.float32).reshape(
                -1, self.observation_dim
            ),
            "teacher_actions": np.asarray(self._teacher_actions, dtype=np.float32).reshape(
                -1, self.action_dim
            ),
            "cohort_ids": np.asarray(self._cohort_ids, dtype=np.int64),
            "scenario_ids": np.asarray(self._scenario_ids, dtype=np.str_),
            "attempt_ids": np.asarray(self._attempt_ids, dtype=np.int64),
            "steps": np.asarray(self._steps, dtype=np.int64),
        }


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_observation_dim: int | None = None,
    expected_action_dim: int | None = None,
) -> None:
    required = {
        "observations",
        "teacher_actions",
        "cohort_ids",
        "scenario_ids",
        "attempt_ids",
        "steps",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"anchor replay is missing arrays: {', '.join(missing)}")
    observations = np.asarray(arrays["observations"])
    actions = np.asarray(arrays["teacher_actions"])
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("anchor replay observations/actions must be rank 2")
    sample_count = observations.shape[0]
    if sample_count < 1 or actions.shape[0] != sample_count:
        raise ValueError("anchor replay must contain aligned observations and actions")
    for name in required - {"observations", "teacher_actions"}:
        if np.asarray(arrays[name]).shape != (sample_count,):
            raise ValueError(f"anchor replay array {name!r} must have one value per sample")
    if expected_observation_dim is not None and observations.shape[1] != expected_observation_dim:
        raise ValueError(
            f"anchor replay observation width {observations.shape[1]} does not match "
            f"expected width {expected_observation_dim}"
        )
    if expected_action_dim is not None and actions.shape[1] != expected_action_dim:
        raise ValueError(
            f"anchor replay action width {actions.shape[1]} does not match expected width "
            f"{expected_action_dim}"
        )
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("anchor replay observations/actions contain non-finite values")
    cohorts = np.asarray(arrays["cohort_ids"])
    if not np.isin(cohorts, (_PRECISION_ANCHOR, _RECOVERY_RETARGET_ANCHOR)).all():
        raise ValueError("anchor replay cohort IDs must be precision=0 or recovery=1")
    if not np.any(cohorts == _PRECISION_ANCHOR) or not np.any(cohorts == _RECOVERY_RETARGET_ANCHOR):
        raise ValueError("anchor replay must contain both precision and recovery cohorts")


def save_anchor_replay(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically save the NPZ data and a hash-checked JSON provenance sidecar."""
    target = Path(path).expanduser().resolve()
    if target.suffix != ".npz":
        raise ValueError("anchor replay path must end in .npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(value) for name, value in arrays.items()}
    _validate_arrays(payload)
    missing = [name for name in _REQUIRED_METADATA if not metadata.get(name)]
    if missing:
        raise ValueError(f"anchor replay metadata is missing: {', '.join(missing)}")

    manifest = dict(metadata)
    manifest.update(
        {
            "schema_version": _REPLAY_SCHEMA_VERSION,
            "sample_count": int(payload["observations"].shape[0]),
            "observation_dim": int(payload["observations"].shape[1]),
            "action_dim": int(payload["teacher_actions"].shape[1]),
            "samples_by_cohort": {
                "precision_anchor": int(np.count_nonzero(payload["cohort_ids"] == 0)),
                "recovery_retarget_anchor": int(np.count_nonzero(payload["cohort_ids"] == 1)),
            },
            "dataset_sha256": anchor_replay_arrays_sha256(payload),
        }
    )
    manifest["metadata_sha256"] = canonical_sha256(manifest)

    temporary_npz = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary_json = target.with_suffix(".json").with_name(
        f".{target.with_suffix('.json').name}.{os.getpid()}.tmp"
    )
    try:
        with temporary_npz.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary_json.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_npz, target)
        os.replace(temporary_json, target.with_suffix(".json"))
    finally:
        temporary_npz.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
    return manifest


def load_anchor_replay(
    path: str | Path,
    *,
    expected_observation_dim: int | None = None,
    expected_action_dim: int | None = None,
) -> AnchorReplayDataset:
    """Load and validate a frozen replay dataset without unpickling."""
    target = Path(path).expanduser().resolve()
    manifest_path = target.with_suffix(".json")
    if not target.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"anchor replay data or provenance sidecar is missing: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _REPLAY_SCHEMA_VERSION:
        raise ValueError("unsupported anchor replay schema version")
    metadata_digest = manifest.pop("metadata_sha256", None)
    if metadata_digest != canonical_sha256(manifest):
        raise ValueError("anchor replay metadata hash does not match")
    missing = [name for name in _REQUIRED_METADATA if not manifest.get(name)]
    if missing:
        raise ValueError(f"anchor replay metadata is missing: {', '.join(missing)}")

    with np.load(target, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    _validate_arrays(
        arrays,
        expected_observation_dim=expected_observation_dim,
        expected_action_dim=expected_action_dim,
    )
    if anchor_replay_arrays_sha256(arrays) != manifest.get("dataset_sha256"):
        raise ValueError("anchor replay dataset hash does not match")
    if arrays["observations"].shape[0] != manifest.get("sample_count"):
        raise ValueError("anchor replay sample count does not match its manifest")

    return AnchorReplayDataset(
        observations=torch.from_numpy(np.asarray(arrays["observations"], dtype=np.float32)),
        teacher_actions=torch.from_numpy(np.asarray(arrays["teacher_actions"], dtype=np.float32)),
        cohort_ids=torch.from_numpy(np.asarray(arrays["cohort_ids"], dtype=np.int64)),
        scenario_ids=tuple(str(value) for value in arrays["scenario_ids"].tolist()),
        attempt_ids=torch.from_numpy(np.asarray(arrays["attempt_ids"], dtype=np.int64)),
        steps=torch.from_numpy(np.asarray(arrays["steps"], dtype=np.int64)),
        metadata=manifest,
    )


def validate_anchor_replay_contract(
    metadata: Mapping[str, Any],
    *,
    actor_observation_contract_sha256: str,
    action_contract_sha256: str,
    plant_contract_sha256: str,
    observation_dim: int,
    action_dim: int,
) -> None:
    """Reject replay data collected for a different actor, action, or plant ABI."""
    expected = {
        "actor_observation_contract_sha256": actor_observation_contract_sha256,
        "source_action_contract_sha256": action_contract_sha256,
        "source_plant_contract_sha256": plant_contract_sha256,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"anchor replay {name} does not match the active training contract")
