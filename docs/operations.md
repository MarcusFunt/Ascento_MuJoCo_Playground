# Operations and deployment

## Supported runtime

The maintained environment is Linux/WSL2 on x86_64, Python 3.11–3.13, and `uv`.
CUDA uses the pinned CUDA 12.8-compatible Torch extra when a usable NVIDIA GPU
is available. CPU is supported for development and CI.

## Preferred install/update path

Run the maintenance script from an existing checkout:

```bash
bash scripts/maintain.sh
```

It can also bootstrap a fresh Linux/WSL2 installation from GitHub. The script
selects CPU or CUDA, synchronizes the locked Python environment, reconciles the
frontend with `npm ci`, rebuilds the Docker image, preserves generated run
artifacts, and starts the local dashboard by default.

```bash
curl -fsSL https://raw.githubusercontent.com/MarcusFunt/Ascento_MuJoCo_Playground/main/scripts/maintain.sh | bash
```

Important options:

```text
--install-dir PATH       explicit checkout location
--branch NAME            branch to install/update (default main)
--compute auto|cu128|cpu selected compute backend
--skip-system-install    do not install OS/Docker/NVIDIA dependencies
--skip-docker-build      synchronize source dependencies only
--no-start-dashboard     do not start the dashboard after build
--force                  discard tracked local changes/local-only commits
```

The maintainer intentionally refuses tracked modifications and local-only
commits without `--force`. Commit or preserve meaningful work before updating;
do not use `--force` to work around unexplained state.

The unified CLI forwards the same script:

```bash
ascento maintain -- --compute cu128 --skip-system-install
```

### WSL-native CUDA profile

The repository includes `config/maintenance.wsl-cu128.env` for the local
Ubuntu WSL installation. It exports an explicit Linux-native checkout path and
selects CUDA:

```bash
source config/maintenance.wsl-cu128.env
bash scripts/maintain.sh
```

The profile targets `/root/Ascento_MuJoCo_Playground`; its `logs/`,
`checkpoints/`, `captures/`, and `evaluations/` bind mounts therefore remain on
WSL's ext4 filesystem instead of a Windows/OneDrive checkout. It must be
sourced and run from Ubuntu WSL.

### Start a managed CUDA training run on this workstation

Use an interactive Ubuntu WSL terminal for a training session. On the first
native setup, the profile is still in the Windows checkout, so source it from
there and let maintenance clone the Linux-native checkout:

```bash
cd /mnt/c/Users/marcu/OneDrive/Dokumenter/GitHub/Ascento_MuJoCo_Playground
source config/maintenance.wsl-cu128.env
bash scripts/maintain.sh
```

For every later run, work from the native checkout. Maintenance makes the
environment match `origin/main`, rebuilds the CUDA Docker image, and restarts
the Dashboard; do not invoke it while a managed run is active.

```bash
cd /root/Ascento_MuJoCo_Playground
source config/maintenance.wsl-cu128.env
bash scripts/maintain.sh
```

Then run the controller probe and start training. Give each experiment an
explicit purpose, tags, iteration count, and seed. `--foreground` is required
for this workstation workflow: it keeps the interactive WSL parent alive until
the managed run reaches a terminal state. Leave that terminal open.

```bash
cd /root/Ascento_MuJoCo_Playground
.venv/bin/ascento tools controller-probe -- --device cuda:0 --json

.venv/bin/ascento run start \
  --task Ascento-Balance-Flat \
  --display-name "balance foundation experiment" \
  --purpose "State the hypothesis being tested" \
  --tag structured_targets_v1 --tag cuda --tag wsl-native \
  --envs 512 --iterations 7500 --seed 123 \
  --foreground --interval 60 --json
```

Balance uses a per-environment world target initialized at the supported reset
pose: both the absolute XY position and the initial world yaw are latched. The
policy observes wrapped target-heading error; a bounded outer heading servo
commands the yaw-rate direction needed to return to it. Do not add a
wheel-target magnitude penalty: it conflicts with recovery and target-return
behavior. Use `ASCENTO_BALANCE_DRIFT_PENALTY_SCALE` only to adjust the small
instantaneous-speed regularizer in a named ablation.

The command prints the managed run ID. In a second Ubuntu WSL terminal, inspect
it without disturbing training:

```bash
cd /root/Ascento_MuJoCo_Playground
.venv/bin/ascento run progress <run-id> --json
.venv/bin/ascento run telemetry <run-id> --limit 50 --json
```

Request a graceful stop only when intended:

```bash
.venv/bin/ascento run stop <run-id> --reason user_requested
```

### Unsafe WSL launch patterns

Do not launch maintenance or training as a background child of a one-shot WSL
invocation, including these patterns:

```bash
wsl.exe -d Ubuntu -- bash -lc 'bash scripts/maintain.sh &'
wsl.exe -d Ubuntu -- bash -lc 'nohup bash scripts/maintain.sh &'
```

On this Windows/WSL installation, returning from `wsl.exe` tears down the
invocation's detached descendants. `nohup` only ignores a hangup signal; it
does not preserve the WSL command session. The visible symptom is that the
launcher returns a PID but no native checkout, setup log, managed run directory,
or training process appears. Run the foreground command in an interactive WSL
terminal instead.

Also avoid passing a complex shell pipeline or embedded script through
`Start-Process wsl.exe -ArgumentList`. Windows can reserialize those arguments
before Bash sees them. Use a normal Ubuntu WSL terminal for the commands above.

Finally, do not invoke only `docker compose -f docker/compose.yaml` when CUDA
is required: the base Compose file defaults to the CPU extra and has no GPU
device request. The WSL profile plus `scripts/maintain.sh` writes the `cu128`
environment file and applies the GPU overlay.

## Docker services

The base compose file is CPU-safe. Add the GPU overlay for CUDA:

```bash
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml -f docker/compose.gpu.yaml build
```

`logs/` is read-write for the dashboard because managed training creates
metadata, status, logs, and checkpoints there. `checkpoints/`, `captures/`, and
`evaluations/` are read-only in the dashboard container. Docker's control socket
is never mounted into that container.

The dashboard normally listens only at `127.0.0.1:8000`. Set
`ASCENTO_DASHBOARD_PORT` to choose another local port.

## Dashboard host supervisor

The web UI cannot run arbitrary host commands. `scripts/install_supervisor.sh`
installs a user-level systemd service with a narrow Unix-socket protocol:

```bash
bash scripts/install_supervisor.sh
```

It supports repository status and one guarded `scripts/maintain.sh` request.
It rejects GUI updates when training is active, the checkout is dirty, the branch
is not `main`, local-only commits exist, or `origin/main` cannot be verified.
The supervisor lives outside Docker so it can finish an accepted update while
the dashboard restarts.

## Private remote dashboard access

Optional Tailscale access uses a sidecar that owns the dashboard networking:

```bash
bash scripts/setup_tailscale.sh
```

It does not expose the trainer, use Tailscale Funnel, or publish the dashboard
to the public internet. Tailnet ACLs decide access. The enrollment credential is
used only to create persistent Tailscale state and is then removed from the
container environment.

Useful variables are `ASCENTO_TAILSCALE_HOSTNAME` and
`ASCENTO_TAILSCALE_EXTRA_ARGS`. See `dashboard/README.md` for the exact
security model and setup examples.

## Artifact locations and environment variables

| Variable | Default | Consumer |
| --- | --- | --- |
| `ASCENTO_ARTIFACT_ROOT` | `logs/rsl_rl` | Dashboard, CLI, MCP managed-run discovery |
| `ASCENTO_EVALUATION_ROOT` | `evaluations` | CLI and MCP evaluation artifact operations |
| `ASCENTO_STALE_AFTER_SECONDS` | `90` | Dashboard stale-run detection |
| `ASCENTO_DASHBOARD_DIST` | `dashboard/frontend/dist` | Dashboard static frontend location |
| `ASCENTO_DASHBOARD_PORT` | `8000` | Compose/launcher local port |
| `ASCENTO_COMPUTE_EXTRA` | context-specific | Preflight and maintained Docker compute extra |
| `ASCENTO_DISABLE_DENSE_SHAPING` | unset | Recovery/jump training ablation switch |

Repository provenance variables (`ASCENTO_REPOSITORY_COMMIT` and
`ASCENTO_REPOSITORY_BRANCH`) are injected by the maintained image when Git is
not available inside a container. They should describe the image being run, not
be used to disguise an unknown revision.

The complete machine-readable variable inventory, including maintenance,
balance-experiment, OpenGL, Tailscale, and preflight inputs, is in
[project-manifest.json](project-manifest.json). `TS_AUTHKEY` is intentionally
not included in that inventory as a configurable value: it is a sensitive
one-time Tailscale enrollment credential and must not be committed or logged.

## Codex MCP registration

After running the WSL CUDA maintenance profile, replace any existing Ascento
registration with the native project-local stdio server:

```powershell
.\scripts\register_codex_mcp.ps1 -Replace
```

It registers `/root/Ascento_MuJoCo_Playground/scripts/ascento_mcp_server.sh`
and explicitly sets `ASCENTO_ARTIFACT_ROOT` to
`/root/Ascento_MuJoCo_Playground/logs/rsl_rl`. Run the WSL CUDA maintenance
profile first; registration refuses to proceed when that native launcher does
not exist. The launcher uses the project’s WSL virtual environment and reserves
stdout for MCP JSON-RPC. Use `-Distro <name>` for another distribution.

Open a new Codex task or restart the app after registration: an already-running
task cannot acquire a newly registered MCP tool surface.

## Routine checks

```bash
ascento dashboard status --json
ascento run list --active --json
ascento evaluate list --json
git status --short --branch
```

For low-impact monitoring, read `run progress`, `run telemetry --limit`, and
`run logs --tail` at 30–60 second intervals instead of forcing repeated GUI
refreshes.
