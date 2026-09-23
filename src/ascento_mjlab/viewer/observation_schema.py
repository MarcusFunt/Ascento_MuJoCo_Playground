"""Semantic feature layout derived from the live observation manager."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ObservationTermSchema:
    name: str
    start: int
    end: int
    shape: tuple[int, ...]
    source: str
    scale: Any
    clip: Any


@dataclass(frozen=True)
class ObservationFeature:
    index: int
    term: str
    dimension: int
    label: str
    unit: str | None
    source: str
    scale: Any
    clip: Any


@dataclass(frozen=True)
class ObservationSchema:
    group: str
    input_dim: int
    terms: tuple[ObservationTermSchema, ...]
    features: tuple[ObservationFeature, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_UNITS = {
    "gravity": "dimensionless",
    "projected_gravity": "dimensionless",
    "base_lin_vel": "m/s",
    "base_ang_vel": "rad/s",
    "height": "m",
    "joint_pos": "rad",
    "joint_vel": "rad/s",
    "contacts": "boolean",
    "contact_forces": "N",
    "effort": "Nm",
    "actions": "normalized action",
    "world_target_error": "m",
    "world_target_heading_error": "rad",
    "root_pos": "m",
    "root_lin_vel_w": "m/s",
    "root_ang_vel_w": "rad/s",
}
_AXES = {
    "gravity": ("x", "y", "z"),
    "base_lin_vel": ("x", "y", "z"),
    "base_ang_vel": ("x", "y", "z"),
    "root_pos": ("x", "y", "z"),
    "root_lin_vel_w": ("x", "y", "z"),
    "root_ang_vel_w": ("x", "y", "z"),
    "world_target_error": ("x", "y"),
}


def build_observation_schema(manager: Any, *, group: str | tuple[str, ...]) -> ObservationSchema:
    """Build flattened offsets/labels from manager term names and runtime shapes."""

    if isinstance(group, tuple):
        if not group:
            raise ValueError("actor observation schema requires at least one runtime group")
        schemas = [build_observation_schema(manager, group=name) for name in group]
        offset = 0
        terms: list[ObservationTermSchema] = []
        features: list[ObservationFeature] = []
        for schema in schemas:
            terms.extend(
                ObservationTermSchema(
                    name=term.name,
                    start=term.start + offset,
                    end=term.end + offset,
                    shape=term.shape,
                    source=term.source,
                    scale=term.scale,
                    clip=term.clip,
                )
                for term in schema.terms
            )
            features.extend(
                ObservationFeature(
                    index=feature.index + offset,
                    term=feature.term,
                    dimension=feature.dimension,
                    label=feature.label,
                    unit=feature.unit,
                    source=feature.source,
                    scale=feature.scale,
                    clip=feature.clip,
                )
                for feature in schema.features
            )
            offset += schema.input_dim
        return ObservationSchema(
            group="+".join(group),
            input_dim=offset,
            terms=tuple(terms),
            features=tuple(features),
        )

    active_terms = getattr(manager, "active_terms", None)
    term_dimensions = getattr(manager, "group_obs_term_dim", None)
    group_dimensions = getattr(manager, "group_obs_dim", None)
    concatenated = getattr(manager, "group_obs_concatenate", None)
    if not isinstance(active_terms, dict) or group not in active_terms:
        raise ValueError(f"observation group {group!r} is not active")
    if not isinstance(term_dimensions, dict) or group not in term_dimensions:
        raise ValueError(f"observation group {group!r} has no runtime term dimensions")
    if isinstance(concatenated, dict) and not concatenated.get(group, False):
        raise ValueError(f"observation group {group!r} is not concatenated")

    names = tuple(active_terms[group])
    shapes = tuple(tuple(int(dim) for dim in shape) for shape in term_dimensions[group])
    if len(names) != len(shapes):
        raise ValueError(f"observation group {group!r} term/dimension count mismatch")

    terms: list[ObservationTermSchema] = []
    features: list[ObservationFeature] = []
    offset = 0
    for name, shape in zip(names, shapes, strict=True):
        cfg = manager.get_term_cfg(group, name)
        source = _source_name(getattr(cfg, "func", None))
        scale = _plain_value(getattr(cfg, "scale", None))
        clip = _plain_value(getattr(cfg, "clip", None))
        width = math.prod(shape) if shape else 1
        terms.append(
            ObservationTermSchema(
                name=name,
                start=offset,
                end=offset + width,
                shape=shape,
                source=source,
                scale=scale,
                clip=clip,
            )
        )
        labels = _dimension_labels(name, shape, getattr(cfg, "params", {}) or {})
        unit = _UNITS.get(name) or _UNITS.get(source)
        for dimension in range(width):
            features.append(
                ObservationFeature(
                    index=offset + dimension,
                    term=name,
                    dimension=dimension,
                    label=labels[dimension],
                    unit=unit,
                    source=source,
                    scale=scale,
                    clip=clip,
                )
            )
        offset += width

    if isinstance(group_dimensions, dict) and group in group_dimensions:
        declared_width = math.prod(group_dimensions[group])
        if declared_width != offset:
            raise ValueError(
                f"observation group {group!r} dimension mismatch: terms total {offset}, "
                f"manager declares {declared_width}"
            )
    return ObservationSchema(
        group=group,
        input_dim=offset,
        terms=tuple(terms),
        features=tuple(features),
    )


def _dimension_labels(name: str, shape: tuple[int, ...], params: dict[str, Any]) -> list[str]:
    width = math.prod(shape) if shape else 1
    if name in {"joint_pos", "joint_vel", "effort"}:
        asset_cfg = params.get("asset_cfg")
        joint_names = getattr(asset_cfg, "joint_names", None)
        if isinstance(joint_names, (list, tuple)) and len(joint_names) == width:
            return [str(item) for item in joint_names]
    if name in {"contacts", "contact_forces"}:
        sensor_names = [
            params[key]
            for key in ("left_sensor_name", "right_sensor_name")
            if isinstance(params.get(key), str)
        ]
        if len(sensor_names) == width:
            return sensor_names
    axes = _AXES.get(name, ())
    if len(axes) == width:
        return list(axes)
    if width == 1:
        return [name]
    return [f"{name}[{index}]" for index in range(width)]


def _source_name(func: Any) -> str:
    if func is None:
        return "unknown"
    return str(getattr(func, "__name__", type(func).__qualname__))


def _plain_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return tuple(_plain_value(item) for item in value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    return str(value)
