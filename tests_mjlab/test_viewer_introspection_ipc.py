from __future__ import annotations

from types import SimpleNamespace

import pytest

from ascento_mjlab.viewer.ipc import CaptureBusyError, ExplanationBusyError, IntrospectionIPC


def test_ipc_publishes_schema_and_atomic_latest_frame_at_configured_rate(tmp_path):
    ipc = IntrospectionIPC(tmp_path, publish_hz=10)
    actor_schema = SimpleNamespace(to_dict=lambda: {"group": "actor", "input_dim": 3})
    critic_schema = SimpleNamespace(to_dict=lambda: {"group": "critic", "input_dim": 4})
    ipc.publish_schema(actor_schema, critic_schema, checkpoint="model_10.pt")

    assert ipc.read_schema() == {
        "schema_version": 1,
        "checkpoint": "model_10.pt",
        "policy_generation": 1,
        "actor": {"group": "actor", "input_dim": 3},
        "critic": {"group": "critic", "input_dim": 4},
        "actor_network": None,
        "critic_network": None,
    }

    frame = SimpleNamespace(
        sequence_id=10,
        to_dict=lambda: {"sequence_id": 10, "actor_output": [0.2, -0.1]},
    )
    assert ipc.publish_frame(frame, now=5.0) is True
    assert ipc.publish_frame(frame, now=5.05) is False
    assert ipc.publish_frame(frame, force=True, now=5.05) is True
    assert ipc.read_latest() == {"sequence_id": 10, "actor_output": [0.2, -0.1]}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["latest.json", "schema.json"]


def test_ipc_does_not_publish_invalid_json_values(tmp_path):
    ipc = IntrospectionIPC(tmp_path)
    frame = SimpleNamespace(to_dict=lambda: {"value": float("nan")})

    try:
        ipc.publish_frame(frame, force=True, now=1.0)
    except ValueError as error:
        assert "JSON" in str(error) or "Out of range" in str(error)
    else:
        raise AssertionError("non-finite telemetry must not be persisted as JSON")
    assert ipc.read_latest() is None


def test_manual_capture_requests_are_queued_and_consumed_once(tmp_path):
    ipc = IntrospectionIPC(tmp_path)
    first = ipc.request_manual_capture()
    with pytest.raises(CaptureBusyError, match="already queued"):
        ipc.request_manual_capture(requested_by="operator")
    first_request = ipc.consume_manual_capture_request()
    second = ipc.request_manual_capture(requested_by="operator")

    requests = [first_request, ipc.consume_manual_capture_request()]

    assert {row["request_id"] for row in requests if row is not None} == {first, second}
    assert {row["requested_by"] for row in requests if row is not None} == {"dashboard", "operator"}
    assert ipc.consume_manual_capture_request() is None


def test_explanation_request_and_result_round_trip(tmp_path):
    ipc = IntrospectionIPC(tmp_path)
    request_id = ipc.queue_explanation_request(
        {"action_index": 2, "input_raw": [0.1, 0.2], "baseline_kind": "normalizer_mean"}
    )

    request = ipc.consume_explanation_request()
    assert request == {
        "action_index": 2,
        "input_raw": [0.1, 0.2],
        "baseline_kind": "normalizer_mean",
        "explanation_id": request_id,
    }
    assert ipc.consume_explanation_request() is None
    ipc.publish_explanation(request_id, {"explanation_id": request_id, "status": "complete"})
    assert ipc.read_explanation(request_id) == {
        "explanation_id": request_id,
        "status": "complete",
    }


def test_only_one_explanation_can_be_queued_or_running_per_viewer(tmp_path):
    ipc = IntrospectionIPC(tmp_path)
    request_id = ipc.queue_explanation_request({"action_index": 0})
    assert ipc.consume_explanation_request()["explanation_id"] == request_id
    with pytest.raises(ExplanationBusyError, match="already queued or running"):
        ipc.queue_explanation_request({"action_index": 1})

    ipc.publish_explanation(request_id, {"explanation_id": request_id, "status": "complete"})
    next_id = ipc.queue_explanation_request({"action_index": 1})
    assert next_id != request_id
