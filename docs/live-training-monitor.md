# Lightweight Live PPO Monitor

The new live monitor appears in Run Detail and Analyze. It visualizes actual
PPO rollout states, not a second evaluation simulator or continuous 3D render.

## Architecture

AscentoProvenanceRunner attaches a monitored step wrapper only for training
runs with a persistent log directory. At most every 0.5 seconds of wall-clock
time, one to four environment rows are compacted into a single tensor and
transferred to CPU in one copy. A bounded background writer atomically updates
a live_monitor.json snapshot. The dashboard reads this via read-only REST and
Server-Sent Events endpoints, then draws the state as a lightweight 2D SVG
and small diagnostic graphs. Stale display frames may be dropped; simulation
steps are never dropped or delayed to accommodate a browser.

## Configuration

- ASCENTO_LIVE_MONITOR=0 disables monitoring (enabled by default).
- ASCENTO_LIVE_MONITOR_ENVS=0,1,2,3 selects up to four environment indices.
  Default is environment 0 only.
- ASCENTO_LIVE_MONITOR_INTERVAL_S=0.5 sets the minimum wall-clock interval
  between samples, from 0.2 to 10 seconds.

The UI reports measured collection overhead per sample in milliseconds. The
small GPU-to-CPU transfer still introduces an occasional synchronization:
compare training throughput with monitoring disabled before tuning frequency.

## Interpretation

The map shows the selected environment's world XY pose and movement trail,
heading, active target and actual velocity. The requested vector is derived
from the scalar speed command toward that target (when both exist). It is not
a vector directly emitted by the actor.

Tilt, root height, wheel contacts, reward component rates and the actual
dt-scaled step reward help expose stationary or collapsed reward farming.

Termination resets matter: mjlab performs auto-reset before the step returns.
A reset marker tells the UI to start a new trajectory. The reward for that
frame is from the preceding transition, while pose and target may already be
from the next episode; use the detailed Rerun recorder for terminal forensics.

These are individual selected environments, not aggregate PPO success metrics.
Already-running training processes must restart using this version to acquire
the sampler. Existing captures and the Rerun replay viewer are unchanged.
