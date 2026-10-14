# Deep Analysis workspace

Deep Analysis is a read-only exploratory interface at /deep-analysis.
It is separate from the evaluation workflow: evaluation runs benchmarks,
checks acceptance criteria and publishes authoritative pass/fail results.
Deep Analysis only inspects captured observations and recorded evaluator flags.

## Source selection is mandatory

- Single checkpoint: choose a training lineage and an exact checkpoint file.
- Compare checkpoints: choose two independent A/B policies; matching iteration
  numbers across different runs are never treated as identical checkpoints.
  Matching scenario IDs are preferred when both captures exist. Mismatched
  scenarios and tasks receive visible warnings.
- Live PPO stream: choose an active dashboard run. Subscribe to the existing
  SSE live-monitor stream, accepting only actual_ppo_rollout frames. This is
  the in-memory PPO policy, not necessarily the latest frozen checkpoint.

The source remains visible throughout analysis. No checkpoint is simulated
simply by selecting it. No data are invented when a capture is missing.

## Analysis sub-tabs

- Spatial: world XY, goal-relative positions, reference target radii, spatial
  hotspots colored by tilt, target error or speed.
- Signals: body tilt, distance, speed, heading, angular velocity, wheel
  contact, actions, joint position, velocity and actuator effort.
- Rewards: real per-step total rewards and weighted reward contributions.
- Events: flags and dwell timers already recorded by the independent evaluator.
- Statistics: sample histograms and full-frame descriptive statistics.
- Rerun 3D: lazily loaded RRD viewer for existing checkpoint-linked recordings.
- Provenance: checkpoint lineage, exact paths, metadata and available hashes.

Live analysis exposes spatial traces, limited signals, weighted reward rates
and source provenance. The other tabs clearly indicate that a live PPO monitor
does not contain dense NPZ history, RRD files or evaluator verdicts.

## API and security

GET /api/deep-analysis/catalog lists checkpoints and matching captures.
GET /api/deep-analysis/episode?checkpoint_id=...&recording_id=...
returns a provenance-verified, read-only NPZ excerpt. It never recomputes
evaluation results. Paths are constrained within trusted roots, symlinks are
rejected, and capture reads are bounded to 64 MiB and ~600 display samples.
Descriptive summary metrics use full-resolution original arrays.

The existing monitor SSE and replay RRD endpoints are reused.

## Tests

Run pytest tests_dashboard/test_deep_analysis.py and the dashboard test suite.
Run TypeScript typecheck and Vite production build. The browser smoke test
uses Chrome to select A and B policies, inspect their real captures, visit
analysis sub-tabs, and capture desktop and mobile screenshots.

Screenshot evidence is in docs/screenshots/deep-analysis-*.png.
