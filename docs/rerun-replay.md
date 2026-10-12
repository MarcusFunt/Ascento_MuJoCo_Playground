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

The current implementation writes a reusable .rrd recording. A dashboard-embedded
viewer and indexed multi-episode UI are separate follow-up integrations.
