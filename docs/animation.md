# Motion capture and Blender animation

The capture pipeline records the simulated robot as named state channels in a
compressed NPZ file. `tools/blender/import_motion.py` builds an editable Blender
scene from that capture and the Ascento URDF visual geometry. Blender is an
external application; it is not a Python dependency of this project.

## Capture and prepare a clip

Capture a policy rollout with the project CLI:

```bash
uv run --extra cu128 ascento capture --task Ascento-Jump-Flat \
  --checkpoint logs/rsl_rl/ascento_jump/model_10000.pt \
  --takes 1 --steps 1000 --output-dir captures/jump
```

The capture includes joint names, root transforms, contact and jump state when
available, applied effort, commands, task and physics contracts, and checkpoint
provenance. The joint names let the importer map state channels to the URDF
independently of their array order. To trim and resample a take around an event:

```bash
uv run --extra cu128 ascento tools clip-motion -- captures/jump/take_000.npz \
  --event takeoff --pre-roll 0.5 --post-roll 1.0 --fps 24 \
  --output captures/jump/jump_short.npz
```

## Build a Blender scene

The description argument accepts the Ascento URDF, its extracted package
directory, or a ZIP package. The warehouse FBX package is vendored and is the
default for `--warehouse`; its source and attribution are recorded beside the
asset archive. A different FBX, package directory, or ZIP can still be passed
explicitly:

```bash
blender --background --python tools/blender/import_motion.py -- \
  --capture captures/jump/jump_short.npz \
  --description /path/to/ascento_description.zip \
  --output captures/blender/jump_short/scene.blend \
  --video-output captures/blender/jump_short/render.mp4
```

The importer evaluates the URDF joint tree, creates the visual meshes and
materials, packs imported textures into the `.blend`, and aligns wheel support
with the warehouse floor. It preserves root and joint animation, adds takeoff
and landing markers when the capture contains those events, and can optionally
render a PNG sequence. The default root frame is `ascento/base`.

For a moving-camera sequence, add an ordered list of shot presets:

```bash
blender --background --python tools/blender/import_motion.py -- \
  --capture captures/jump/jump_short.npz \
  --description /path/to/ascento_description.zip \
  --output captures/blender/jump_cinematic/scene.blend \
  --video-output captures/blender/jump_cinematic/render.mp4 \
  --camera-shots low_front side_follow orbit \
  --resolution 1920
```

Available presets are `low_front`, `side_follow`, `rear_chase`, and `orbit`.
The importer divides the clip across the selected cuts and binds timeline
markers to each camera. Aim follows the evaluated robot bounds; depth of field
is off by default and can be enabled with `--cinematic-dof` when camera shots
are selected. At least one output frame is required for each selected shot.

The warehouse model and required attribution details are documented in
[`tools/blender/assets/README.md`](../tools/blender/assets/README.md). Every
scene gets a sibling `*.manifest.json` containing the source capture SHA-256,
policy kind and checkpoint path/hash, robot-description and warehouse hashes,
Blender/importer versions, render settings, output hashes, and status. A
checkpointed capture without a recorded checkpoint hash must be rendered with
`--checkpoint`; the importer verifies that file against any hash already in
the capture. Zero-policy captures are explicitly recorded as such. Captures
whose policy cannot be identified are rejected rather than silently rendered
without checkpoint provenance.

MP4 output uses Blender's FFmpeg support with MPEG-4/H.264. `--render-dir` can
also write a PNG sequence; requesting both the PNG sequence and MP4 renders the
animation twice. Keep the scene, manifest, video, and optional frames under
`captures/blender` so the dashboard can serve them from the read-only WSL
captures mount.

When Blender is installed on Windows and the project/dashboard captures live
in WSL, run the checked-in wrapper from this checkout. It invokes this
checkout's importer and writes directly through the WSL UNC path into the
Linux checkout's mounted captures directory:

```powershell
.\scripts\render_blender.ps1 `
  -Capture '\\wsl.localhost\Ubuntu\root\Ascento_MuJoCo_Playground\captures\jump\take_000.npz' `
  -Description 'C:\path\to\ascento_description.zip' `
  -Name 'jump-take-000'
```

Set `ASCENTO_WSL_REPOSITORY` or pass `-WslRepository` when the Linux checkout
is at another path. For a legacy capture that lacks checkpoint provenance,
pass `-Checkpoint` with the checkpoint that generated it. The Ascento robot
description archive remains a local input asset and is not committed.

The Analyze page's Blender pipeline panel polls `GET /api/blender/renders` and
plays MP4s or previews scenes from `captures/blender`. The dashboard container
mounts `captures` read-only; Blender writes through the WSL host path and the
dashboard only serves completed/partial artifacts and their manifests.

## Camera checks

Run the Blender-specific checks with Blender's bundled Python:

```bash
blender --background --factory-startup --python tests_blender/test_cinematics.py
```
