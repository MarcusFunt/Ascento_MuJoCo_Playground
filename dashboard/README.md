# Ascento Training Dashboard

A small local web dashboard for monitoring mjlab/RSL-RL training over Tailscale.
It reads run artifacts and does not own or supervise the PPO process itself.

## What it shows

- current stage, progress, steps/s, elapsed time and ETA from `telemetry.jsonl`;
- TensorBoard/RSL-RL metric artifacts and console output;
- live captured console output using Server-Sent Events;
- recent traceback/error excerpts;
- available RSL-RL checkpoints and motion-capture export location;
- previous runs discovered under the artifact root.

Motion export is intentionally a separate `capture-motion` command so capture
does not compete with training or become a second rollout framework.

## Install

From the repository root, after installing the project with uv:

```bash
uv sync --extra cu128
cd dashboard/frontend
npm install
npm run build
cd ../..
```

For frontend development, run `npm run dev` in `dashboard/frontend`; Vite proxies
`/api` requests to `127.0.0.1:8000`.

## Start the dashboard

```bash
python -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8000
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
ASCENTO_ARTIFACT_ROOT=/path/to/logs python -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8000
```

## Start a monitored training run

The standard mjlab trainer writes RSL-RL checkpoints and TensorBoard events. To
capture console output for the dashboard, launch it through:

```bash
python -m dashboard.launch --task Ascento-Balance-Flat \
  -- --env.scene.num-envs 512 --agent.max-iterations 10000
```

The artifacts live under the configured RSL-RL log root. The dashboard treats
these artifacts as read-only and leaves checkpoint playback/capture to the
normal mjlab tools.

For motion exports, use `capture-motion` after training, for example:

```bash
capture-motion Ascento-Jump-Flat --checkpoint /path/to/model_10000.pt --takes 20
```
