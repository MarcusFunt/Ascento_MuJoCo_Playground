# Ascento documentation

This directory is the maintained map of the repository. It distinguishes the
contracts that are evaluated from the heuristics used to train, monitor, or
operate them.

| Need | Read |
| --- | --- |
| Understand the simulator, plant, modules, and boundaries | [Architecture](architecture.md) |
| Set up, update, and operate a workstation or container | [Operations](operations.md) |
| Train or continue a policy | [Training](training.md) |
| Decide whether a checkpoint passes | [Evaluation](evaluation.md) |
| Use every `ascento` command | [CLI reference](cli-reference.md) |
| Let an agent monitor or operate the project | [MCP reference](mcp-reference.md) |
| Use or diagnose the web control plane | [Dashboard](dashboard.md) |
| Compare the learning/control topology with Wheel-Legged-Lab | [Wheel-Legged-Lab comparison](wheeled-legged-lab-comparison.md) |
| Modify the project and validate a change | [Development](development.md) |
| Resolve common run, artifact, graphics, or integration failures | [Troubleshooting](troubleshooting.md) |

## Documentation contract

Human-readable pages explain intent, workflows, safety boundaries, and how to
interpret results. The JSON files are stable inventories for tooling:

- [project-manifest.json](project-manifest.json) records project-level runtime,
  task, artifact, and script contracts;
- [cli-reference.json](cli-reference.json) describes the complete command tree
  and options; and
- [mcp-tools.json](mcp-tools.json) describes the MCP tool surface and which
  calls mutate state or can take significant time.

The implementation remains authoritative for executable behavior. When an
interface changes, update the matching human page and JSON inventory in the
same change. In particular, update the CLI inventory with
`src/ascento_mjlab/cli.py`, the MCP inventory with
`src/ascento_mjlab/mcp_server.py`, suite descriptions with
`benchmarks/suites/*.toml`, and runtime values with the files named in
`project-manifest.json`.

Generated run, capture, checkpoint, and evaluation artifacts are intentionally
ignored by Git. They are evidence for a particular execution, not source
documentation.
