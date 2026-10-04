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
directory, or a ZIP package. The optional warehouse is a separate Sketchfab
asset and is not included in the repository; download it and pass its ZIP,
package directory, or FBX explicitly with `--warehouse`:

```bash
blender --background --python tools/blender/import_motion.py -- \
  --capture captures/jump/jump_short.npz \
  --description /path/to/ascento_description.zip \
  --warehouse /path/to/warehouse_fbx_model_free.zip \
  --output captures/blender/jump_short.blend \
  --render-dir captures/blender/jump_short_frames
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
  --warehouse /path/to/warehouse_fbx_model_free.zip \
  --output captures/blender/jump_cinematic.blend \
  --camera-shots low_front side_follow orbit \
  --render-dir captures/blender/jump_cinematic_frames \
  --resolution 1920
```

Available presets are `low_front`, `side_follow`, `rear_chase`, and `orbit`.
The importer divides the clip across the selected cuts and binds timeline
markers to each camera. Aim follows the evaluated robot bounds; depth of field
is off by default and can be enabled with `--cinematic-dof` when camera shots
are selected. At least one output frame is required for each selected shot.

The warehouse model and required attribution details are documented in
[`tools/blender/assets/README.md`](../tools/blender/assets/README.md). The
archive is not vendored because the source requires an authenticated download.
The Ascento robot description archive supplied with the local workspace remains
an input asset and is not committed to the repository.

## Camera checks

Run the Blender-specific checks with Blender's bundled Python:

```bash
blender --background --factory-startup --python tests_blender/test_cinematics.py
```
