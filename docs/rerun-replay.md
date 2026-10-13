# Rerun + MJCF episode replay

The Ascento replay exporter converts existing NPZ motion captures into timestamped Rerun recordings.
It uses the standalone MuJoCo robot XML and the open-source rerun-loader-mjcf library
to reconstruct **all articulated body transforms** without rerunning the policy or simulator.

## Install (optional)

Rerun is deliberately not a dependency of the training environment.

    uv sync --extra rerun

This extra contains rerun-sdk and rerun-loader-mjcf; the expected versions are
tracked in uv.lock. To test or run the exporter without changing the existing
virtual environment:

    uv run --no-sync --with rerun-sdk==0.33.1 --with rerun-loader-mjcf==1.0.1 \
      python -m ascento_mjlab.tools.export_rerun --help

## Record and export an evaluation

Use the established motion capture path (or the unified ascento capture command).
From the project checkout:

    .venv/bin/python -m ascento_mjlab.tools.capture_motion \
      --task Ascento-Locomotion-Speed-Flat --checkpoint /path/to/compatible.pt \
      --takes 1 --steps 1000 --output-dir captures/rerun-demo

Then export the resulting motion capture:

    .venv/bin/python -m ascento_mjlab.tools.export_rerun \
      captures/rerun-demo/take_000.npz \
      --output captures/rerun-demo/take_000.rrd

If dependencies were not installed into the existing virtual environment, use
the uv run --no-sync --with command shown above in place of .venv/bin/python.

Open the RRD file with the Rerun native viewer, or serve it using Rerun's
web viewer. Saved RRD files are independent of the training process.
Use --stride 5 for a lightweight preview and --collision to show collision geometry.

## Captured signals

Existing recording fields remain compatible: root and joint poses, actions, actuator
effort, contacts, some motion commands and physics/checkpoint provenance.

New captures additionally include:

| Field | Units | Meaning |
| --- | --- | --- |
| root_lin_vel_w | m/s | World-frame root link linear velocity |
| target_xy | m | Active world target (where available) |
| target_yaw | rad | Active world target yaw (where available) |
| speed_command | m/s | Applied speed command in speed-conditioned tasks |
| target_attempt_id | identifier | Target attempt, when available |
| reward_terms_rate | reward/s | Per-term weighted reward rates, from mjlab RewardManager |
| reward_total | reward | Actual dt-scaled reward delivered at each policy step |
| reward_dt | s | Environment step duration used to scale reward |

Metadata contains the named reward terms, joint ordering, capture schema version
and SHA-256 of the source robot XML. Missing optional channels in older captures
do not prevent replay; the corresponding overlays and charts will be absent.

The exporter rejects recordings whose weighted reward terms do not reproduce
the actual timestep reward, or whose joint names, quaternion shapes, timestamps,
or robot XML checksum are inconsistent. The reward values are read from the
RewardManager; reward functions are not called a second time.

## Spatial views and limitations

The RRD contains full 3D MuJoCo robot body geometry and motion via MJCFLogger,
a top-down 2D path with robot and target markers, measured and requested
speed vectors where available, and time-series plots of reward components.

Requested velocity is an illustrative target-directed vector derived from the
scalar speed command and the current world target. It must not be confused with
a vector directly returned by the policy or with actual robot velocity.

Historical kinematics are recomputed from recorded root/joint positions via
MuJoCo forward kinematics. This reconstructs articulated poses; it does **not**
simulate the original control forces, reproduce the original MuJoCo Warp contact
solver, or establish that a counterfactual action would succeed. Saved contact
flags and other dynamical signals retain their original recorded meaning.

Captures are intentionally produced for selected single-environment evaluations,
not for every training environment on each GPU step. The converter runs offline
so it introduces no dependency or GPU data transfer overhead into PPO training.

The exporter writes reusable .rrd recordings; the Analyze page embeds the
Rerun web viewer and indexes captures under the captures/ directory.


## Embedded Analyze viewer

The React dashboard embeds the Rerun web viewer directly on Analyze.
The backend lists recordings under captures/ and generates RRD files on demand
using POST /api/replays/export/<capture>.npz. Recorded files are served only
from the dashboard's own origin under /api/replays/files/.

The normal scripts/run_dashboard.sh launcher uses --extra dashboard --extra rerun.
The frontend pins @rerun-io/web-viewer-react to 0.33.1, matching the Python
Rerun SDK. Rebuild the frontend after updating the repository.

Choose a capture, generate an RRD if needed, then click Open 3D viewer.
This is opt-in because the WebAssembly renderer is relatively large.
The recording is served from the dashboard origin, without a separate viewer
service or public RRD URL.

The viewer shows persisted kinematic recordings, not live simulation.
Older NPZ captures can be converted but lack the newer reward and target signals.

## Reproduce a benchmark scenario and diagnose settling

Ordinary capture_motion playback does not reproduce the benchmark's resolved
reset, timed world-target commands or applied disturbances. For checkpoint
comparisons, use the exact-suite evaluator capture command instead:

~~~bash
cd /root/Ascento_MuJoCo_Playground
CKPT="logs/rsl_rl/<run>/<task>/<timestamp>/model_16300.pt"
/root/.cache/ascento-mjlab/cu128/bin/python3 -m ascento_mjlab.tools.capture_benchmark   --suite goal_settle_v5_comprehensive_v1 --checkpoint "$CKPT"   --family precision_short --index 0 --index 1   --output-dir captures/benchmark-v5-model16300 --rrd
~~~

You may specify a fully-qualified scenario with --scenario-id
goal_settle_v5_comprehensive_v1/precision_short/000000. Repeat --family and
--index to select additional matched cases. Explicit selection is required:
the command never substitutes a stochastic playback case for a missing suite
case. All scenarios use the same resolved seeds and intervention timings as
the comprehensive evaluator. The batch-size option controls resource use;
--batch-size 1 is the conservative default.

Saved NPZ/RRD artifacts use family_index names. Their metadata records the
full scenario (reset, target commands, disturbance, horizon), suite hash,
checkpoint hash, robot XML hash and reward term names. The JSON manifest links
scenario IDs and physical outcome metrics to saved files.

The Rerun view includes synchronized contact flags (left, right, both),
body tilt, roll/pitch/yaw rates, target/heading error, speed, action-change
RMS, action second difference and actuator effort RMS. In the same timeline,
it records each strict evaluator waypoint condition (distance, heading,
speed, tilt, angular_xy, yaw_rate, both_wheels), the conjunction pass flag,
the active-target mask and consecutive settling dwell in seconds.
Recovery conditions are distinct; they include the height constraint,
a slightly different speed/heading envelope and a separate post-disturbance
dwell timer.

The evaluator also emits waypoint_max_consecutive_dwell_s and
waypoint_stop_<condition>_pass_fraction (measured during its final 2-second
window). These are diagnostic metrics and do not change gate acceptance.
Individual wheel-contact flags indicate binary sensor contact, not normal
force. Older captures without exact evaluator diagnostics still export but
cannot provide those missing series.

Dependencies were installed into the project .venv and its independent
CUDA-enabled evaluation environment. The default .venv contains CPU-only
PyTorch and cannot execute CUDA benchmark captures; use the CUDA interpreter
shown above for actual GPU playback and RRD generation. Installing Rerun does
not require restarting an already-running PPO training process.
