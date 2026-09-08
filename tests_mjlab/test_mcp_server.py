import pytest

from ascento_mjlab import mcp_server


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
