"""Keep machine-readable interface inventories aligned with executable surfaces."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from ascento_mjlab.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


def _json(name: str) -> dict:
    with (ROOT / "docs" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return action.choices


def _options(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option != "-h" and option != "--help"
    }


def _documented_options(payload: dict, *groups: str) -> set[str]:
    return {
        item["name"]
        for group in groups
        for item in payload.get(group, [])
    }


def _mcp_tool_names() -> set[str]:
    source = ROOT / "src" / "ascento_mjlab" / "mcp_server.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "tool"
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "mcp"
            ):
                names.add(node.name)
    return names


def test_machine_readable_docs_are_valid_and_identify_the_project() -> None:
    project = _json("project-manifest.json")
    cli = _json("cli-reference.json")
    mcp = _json("mcp-tools.json")

    assert project["schema_version"] == 1
    assert project["repository"]["name"] == "Ascento MuJoCo Playground"
    assert cli["program"] == "ascento"
    assert mcp["server"]["name"] == "ascento"


def test_cli_inventory_tracks_the_parser_command_tree() -> None:
    documented = _json("cli-reference.json")["commands"]
    parser = build_parser()
    root_commands = _subcommands(parser)

    assert set(documented) == set(root_commands)
    for name in ("run", "evaluate", "tools", "mcp", "dashboard"):
        assert set(documented[name]["subcommands"]) == set(_subcommands(root_commands[name]))


def test_cli_inventory_tracks_first_class_option_sets() -> None:
    documented = _json("cli-reference.json")["commands"]
    parser = build_parser()
    root_commands = _subcommands(parser)

    run_common = _documented_options(documented["run"], "common_options")
    for name, payload in documented["run"]["subcommands"].items():
        source = _subcommands(root_commands["run"])[name]
        expected = run_common | _documented_options(payload, "options")
        assert _options(source) == expected

    evaluation_common = _documented_options(documented["evaluate"], "common_options")
    for name, payload in documented["evaluate"]["subcommands"].items():
        source = _subcommands(root_commands["evaluate"])[name]
        expected = evaluation_common | _documented_options(payload, "options")
        assert _options(source) == expected

    assert _options(root_commands["capture"]) == _documented_options(
        documented["capture"], "options"
    )

    dashboard_common = _documented_options(documented["dashboard"], "common_options")
    for name, payload in documented["dashboard"]["subcommands"].items():
        source = _subcommands(root_commands["dashboard"])[name]
        expected = dashboard_common | _documented_options(payload, "options")
        assert _options(source) == expected


def test_mcp_inventory_tracks_decorated_tool_functions() -> None:
    documented = {item["name"] for item in _json("mcp-tools.json")["tools"]}
    assert documented == _mcp_tool_names()


def test_runs_page_matches_the_current_curriculum_and_selection_gate() -> None:
    source = (ROOT / "dashboard" / "frontend" / "src" / "RunsPage.jsx").read_text(
        encoding="utf-8"
    )

    assert "six consecutive 512-episode windows" in source
    assert "balance_gate_v5 before selecting it" in source
    assert "three 512-episode windows" not in source
    assert "balance_gate_v2 before selecting it" not in source
