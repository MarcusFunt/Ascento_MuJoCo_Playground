# Ascento Training Dashboard

A small local web dashboard for monitoring Brax/MJX training over Tailscale.
It reads the artifacts the trainer already writes and does not own or supervise
the PPO process itself.

## What it shows

- current stage, progress, steps/s, elapsed time and ETA from `telemetry.jsonl`;
- automatically selected PPO/evaluation metric charts;
- live captured console output using Server-Sent Events;
- recent traceback/error excerpts;
- the newest deterministic MuJoCo checkpoint preview and rollout statistics;
- previous runs discovered under the artifact root.

The web UI does not expose an arbitrary shell. Its only write action is a fixed
"Render latest" endpoint that invokes the repository's deterministic checkpoint
renderer for the selected run.

## Install

From the repository root, with the existing Python environment activated:

```bash
pip install -r requirements-dashboard.txt
cd dashboard/frontend
npm install
npm run build
cd ../..
```

For frontend development, run `npm run dev` in `dashboard/frontend`; Vite proxies
`/api` requests to `127.0.0.1:8000`.

## Start the dashboard

```bash
./scripts/run_dashboard.sh
```

The production server intentionally binds only to `127.0.0.1:8000`. To make it
available to your tailnet without exposing it on the LAN:

```bash
tailscale serve --bg 8000
```

Open the HTTPS Tailscale Serve URL from another device on the same tailnet.
Do not use Tailscale Funnel for this private monitor.

Set a different artifact location when needed:

```bash
ASCENTO_ARTIFACT_ROOT=/path/to/runs ./scripts/run_dashboard.sh
```

## Start a monitored training run

The normal trainer already writes `telemetry.jsonl` and checkpoints. To also
capture console output and reliable process state, launch it through:

```bash
ASCENTO_JAX_PLATFORM=cuda python -m dashboard.launch -- \
  --stage balance \
  --timesteps 50000000 \
  --num-envs 1024
```

This creates a unique directory such as:

```text
training/artifacts/20260902_110500_balance/
├── checkpoint/
├── telemetry.jsonl
├── training.log
├── run_status.json
├── policy_params.pkl
└── training_manifest.json
```

Any additional arguments after `--` are forwarded to `training.train`, except
`--output`, which is managed by the launcher so runs cannot accidentally mix
telemetry from different sessions.

## Rendering checkpoints

The dashboard can render the newest numeric checkpoint on demand. This runs a
short deterministic MuJoCo rollout in a separate process and stores its PNG and
`progress_renders.jsonl` under `<run>/renders/`.

Rendering uses the same JAX backend environment as the dashboard process. On a
GPU training machine it can therefore temporarily compete with PPO for GPU
resources; use it as an occasional inspection tool rather than a live video
stream.
