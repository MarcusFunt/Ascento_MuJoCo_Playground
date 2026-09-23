from __future__ import annotations

from types import SimpleNamespace

import pytest

from ascento_mjlab.viewer.observation_schema import build_observation_schema


def _term_cfg(func_name: str, *, scale=None, clip=None, params=None):
    return SimpleNamespace(
        func=SimpleNamespace(__name__=func_name),
        scale=scale,
        clip=clip,
        params=params or {},
    )


class _Manager:
    active_terms = {"actor": ["gravity", "joint_pos", "effort", "target_error"]}
    group_obs_dim = {"actor": (12,)}
    group_obs_term_dim = {"actor": [(3,), (4,), (4,), (1,)]}
    group_obs_concatenate = {"actor": True}

    def get_term_cfg(self, group: str, name: str):
        assert group == "actor"
        return {
            "gravity": _term_cfg("projected_gravity"),
            "joint_pos": _term_cfg(
                "joint_pos_rel",
                scale=0.5,
                params={"asset_cfg": SimpleNamespace(joint_names=("hip_l", "knee_l", "hip_r", "knee_r"))},
            ),
            "effort": _term_cfg("actuator_effort", clip=(-65.0, 65.0)),
            "target_error": _term_cfg("target_error"),
        }[name]


def test_observation_schema_flattens_runtime_term_dimensions_and_metadata():
    schema = build_observation_schema(_Manager(), group="actor")

    assert schema.input_dim == 12
    assert [(term.name, term.start, term.end) for term in schema.terms] == [
        ("gravity", 0, 3),
        ("joint_pos", 3, 7),
        ("effort", 7, 11),
        ("target_error", 11, 12),
    ]
    assert [feature.label for feature in schema.features[3:7]] == [
        "hip_l",
        "knee_l",
        "hip_r",
        "knee_r",
    ]
    assert [feature.unit for feature in schema.features[:3]] == ["dimensionless"] * 3
    assert schema.features[3].scale == 0.5
    assert schema.features[7].clip == (-65.0, 65.0)
    assert schema.features[0].source == "projected_gravity"
    assert schema.to_dict()["input_dim"] == 12


def test_observation_schema_rejects_nonconcatenated_or_inconsistent_runtime_groups():
    manager = _Manager()
    manager.group_obs_concatenate = {"actor": False}
    with pytest.raises(ValueError, match="concatenated"):
        build_observation_schema(manager, group="actor")

    manager.group_obs_concatenate = {"actor": True}
    manager.group_obs_dim = {"actor": (13,)}
    with pytest.raises(ValueError, match="dimension"):
        build_observation_schema(manager, group="actor")


def test_observation_schema_combines_multiple_actor_groups_in_runtime_order():
    class _MultiGroupManager:
        active_terms = {"actor": ["gravity"], "target": ["target_error"]}
        group_obs_dim = {"actor": (3,), "target": (2,)}
        group_obs_term_dim = {"actor": [(3,)], "target": [(2,)]}
        group_obs_concatenate = {"actor": True, "target": True}

        def get_term_cfg(self, group: str, name: str):
            func_name = "projected_gravity" if name == "gravity" else "target_error"
            return _term_cfg(func_name)

    schema = build_observation_schema(_MultiGroupManager(), group=("actor", "target"))

    assert schema.group == "actor+target"
    assert schema.input_dim == 5
    assert [term.start for term in schema.terms] == [0, 3]
    assert [feature.index for feature in schema.features] == list(range(5))
