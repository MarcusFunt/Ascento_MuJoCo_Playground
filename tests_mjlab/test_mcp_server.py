import pytest

from ascento_mjlab import mcp_server, operations


def test_mcp_artifact_root_resolves_relative_paths_from_the_checkout(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("ASCENTO_ARTIFACT_ROOT", "managed-runs")
    monkeypatch.setattr(operations, "repo_root", lambda: checkout)

    assert mcp_server._root() == (checkout / "managed-runs").resolve()


def test_mcp_exposes_run_and_evaluation_operations_when_the_optional_extra_is_present():
    if mcp_server.FastMCP is None:
        pytest.skip("MCP optional dependency is not installed")

    names = set(mcp_server.mcp._tool_manager._tools)
    assert {
        "start_run",
        "update_run_metadata",
        "get_run_checkpoint",
        "list_evaluation_suites",
        "evaluate_policy",
        "get_evaluation_report",
        "archive_evaluation_report",
        "capture_policy",
        "run_evaluator_preflight",
    } <= names
