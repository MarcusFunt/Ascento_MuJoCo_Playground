# Browser Policy Viewer: plan and execution

## Goal

Allow an operator to visualize an Ascento MuJoCo policy from a managed training run in a
browser without attaching a renderer to the training process. The viewer must support a
specific checkpoint, the latest completed checkpoint, and an optional follow mode that
moves to newly completed checkpoints while training continues.

The browser viewer is an observer. Training remains authoritative and must continue if the
viewer crashes, disconnects, reloads a checkpoint, or is stopped.

## Chosen stack

| Layer | Choice | Reason |
| --- | --- | --- |
| Physics | MuJoCo Warp / mjlab 1.6.0 | Existing simulator and task registry |
| Policy | RSL-RL / PyTorch checkpoints | Existing `model_*.pt` artifacts |
| Browser 3D | mjlab `ViserPlayViewer` + mjviser | Native browser viewer with controls and checkpoint UI |
| Dashboard | Existing React + Vite | No second frontend |
| Control API | Existing FastAPI service | Already owns run lifecycle and artifact discovery |
| Viewer transport | Viser HTTP/WebSocket | Native scene/state synchronization |
| Isolation | Managed viewer subprocess | No coupling to PPO trainer timing or process state |
| Remote access | Existing Tailscale sidecar | Viewer stays on the Tailnet |
| Checkpoint selection | Run-scoped discovery | Prevent arbitrary filesystem checkpoint loading |
| Live updates | Viewer-thread checkpoint follow | Reload only at a safe viewer-loop boundary |

## Architecture

```text
Tailnet browser
  |
  +-- React dashboard :8000
  |      |
  |      +-- FastAPI
  |             |
  |             +-- RunService
  |             +-- ViewerService
  |                    |
  |                    +-- one managed subprocess
  |
  +-- Viser :8081 <----+
         |
         +-- ManagerBasedRlEnv (1 environment)
         +-- RSL-RL runner
         +-- deterministic policy adapter
         +-- checkpoint contract validation
         +-- CheckpointManager
```

The trainer and viewer communicate only through durable checkpoint files.

```text
trainer (many envs) ---> model_13500.pt ---> viewer (one env)
```

No viewer object, WebSocket, renderer, or browser request is allowed to enter the training
process.

## Implementation phases

### 1. Run-scoped checkpoint discovery

Implemented in `src/ascento_mjlab/viewer/checkpoints.py`.

Requirements:

- Discover only `model_*.pt` and `checkpoint_*.pt` below the selected run.
- Reject absolute paths and parent traversal.
- Ignore zero-byte and still-fresh files.
- Sort by checkpoint iteration, modification time, then relative path.
- Represent the selected checkpoint by a run-relative path.
- Expose iteration, size, mtime, and age for the dashboard.

A short stability age prevents the viewer from racing a checkpoint writer. If a load or
contract check still fails, the viewer worker surfaces the error and can be restarted; the
training process is unaffected.

### 2. Standalone viewer worker

Implemented in `src/ascento_mjlab/viewer/worker.py`.

The worker:

1. Loads the selected task with `play=True`.
2. Forces one environment.
3. Builds the existing RSL-RL runner.
4. Loads the requested checkpoint in actor-only mode.
5. Validates it with `require_current_checkpoint_contracts()` against the canonical
   training configuration.
6. Wraps inference with `RslRlPolicyAdapter(..., deterministic=True)`.
7. Starts an explicit `viser.ViserServer` on the requested host/port.
8. Gives mjlab a `CheckpointManager` for manual switching.
9. Writes a runtime status file after every successful checkpoint load.
10. Closes the environment and Viser server on exit.

The worker can also be launched directly:

```bash
python -m ascento_mjlab.viewer.worker \
  --task Ascento-Balance-Flat \
  --run-dir logs/rsl_rl/<run> \
  --checkpoint latest \
  --port 8081
```

### 3. Safe follow mode

Implemented in the worker as a small `ViserPlayViewer` subclass.

Follow mode does not mutate Torch objects from a filesystem watcher thread. Instead, the
viewer loop periodically checks the run-scoped checkpoint list. If a newer stable
checkpoint exists, it queues mjlab's normal `FETCH_CHECKPOINT` action. The checkpoint is
therefore loaded on the viewer thread, contract validation runs first, and mjlab resets the
environment/policy after a successful swap.

This also resets recurrent-policy state through the existing policy reset hook.

### 4. Managed viewer lifecycle

Implemented in `dashboard/viewer_service.py`.

The first release intentionally has one viewer slot. This keeps GPU and port ownership
predictable and avoids giving the dashboard Docker control.

The service owns:

- subprocess start/stop,
- viewer PID and state,
- a fixed Viser port,
- worker logs,
- worker runtime status,
- checkpoint selection,
- checkpoint/training iteration lag,
- startup readiness via a local TCP probe,
- cleanup on dashboard shutdown.

Viewer states are:

`starting -> running -> stopping -> stopped`

or

`starting/running -> failed`

A viewer failure never changes dashboard health and never signals the trainer.

### 5. HTTP API

Implemented in `dashboard/app.py`.

Read operations:

- `GET /api/runs/{run_id}/checkpoints`
- `GET /api/viewers`
- `GET /api/viewers/{viewer_id}`
- `GET /api/viewers/{viewer_id}/logs`

Control operations:

- `POST /api/viewers`
- `DELETE /api/viewers/{viewer_id}`

Control operations require `X-Ascento-Control: 1`, matching the dashboard's existing
privileged-control convention. Tailnet ACLs remain the network/authentication boundary.

Example start request:

```json
{
  "run_id": "abc123",
  "checkpoint": "latest",
  "follow": true
}
```

### 6. Runtime truth and lag

The worker writes the actual loaded checkpoint after initial load and after every Viser
checkpoint hot-swap. The dashboard therefore does not infer policy state from the start
request.

Status includes:

- loaded checkpoint,
- policy iteration,
- current training iteration,
- lag in iterations,
- fixed/manual vs follow mode,
- viewer port,
- process state and exit code.

This is deliberately labeled as a policy preview rather than claiming the browser is
rendering the trainer's exact in-memory PPO iteration.

### 7. React integration

Implemented in `dashboard/frontend/src/RunsPage.jsx` and `viewer.css`.

The selected-run panel now provides:

- stable checkpoint dropdown,
- follow-latest toggle,
- start button,
- viewer process state,
- loaded/training iteration and lag,
- full-screen browser viewer link,
- stop button,
- failure/log hint.

The viewer opens as its own page instead of an iframe. Viser already has a substantial UI
for camera, commands, rewards, metrics, overlays, contacts, groups, speed, reset, and
checkpoint switching; full-screen use preserves that interface.

### 8. Tailnet and workstation access

The viewer process runs inside the dashboard container and therefore shares the Tailscale
sidecar's network namespace.

Tailnet peers reach:

```text
http://ascento-dashboard:8000   dashboard
http://ascento-dashboard:8081   Viser
```

`docker/compose.tailscale.yaml` also maps viewer port 8081 to loopback so the workstation
can open it locally without exposing it on the LAN.

No Docker socket is added to the dashboard.

## Failure behavior

| Failure | Behavior |
| --- | --- |
| No checkpoint | Start returns conflict; trainer unaffected |
| Checkpoint still being written | Not offered until stable |
| Contract mismatch | Viewer reload fails and the worker reports/exits; trainer unaffected |
| Viser crash | Viewer state becomes failed |
| GPU OOM in viewer | Viewer dies; trainer remains separate |
| Browser disconnect | Worker remains available |
| Manual checkpoint load fails | Viewer reports the failure; worker may exit; trainer unaffected |
| Training stops | Viewer continues with its loaded checkpoint |
| Dashboard shutdown | Managed viewer receives SIGTERM |
| Unknown viewer ID | API returns 404 |

## Validation gates

The implementation is considered ready when all of the following pass:

1. Checkpoint discovery is path-contained, ordered, and stability-filtered.
2. Viewer service launches only a run-scoped checkpoint.
3. A second active viewer is rejected.
4. Stop targets only the viewer process group.
5. API routes are registered and control headers are enforced.
6. Frontend production build succeeds.
7. Ruff and dashboard/mjlab unit tests pass.
8. Docker Compose renders with the Tailscale overlay.
9. On a GPU host, a real Ascento checkpoint reaches Viser and can be switched manually.
10. Follow mode advances only after a new stable checkpoint appears.

CI covers 1-8. Gate 9-10 require the existing GPU-capable host because GitHub's standard
runner does not have the project's CUDA/MuJoCo-Warp runtime.

## Deliberate non-goals for this PR

- No direct attachment to the trainer's in-memory environment.
- No Docker socket or container-per-viewer orchestration.
- No WebRTC/video stream.
- No MuJoCo-WASM duplicate simulator.
- No iframe embedding.
- No arbitrary checkpoint filesystem paths.
- No unbounded viewer concurrency.

If multi-viewer support is needed later, `ViewerService` can allocate a bounded port pool
and process table without changing the worker or browser protocol.
