"""Compatibility overlay for the pinned mjlab 1.6.0 package.

Based on https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/__init__.py
(Apache-2.0; see the installed mjlab distribution's LICENSE). This preserves
upstream initialization while replacing the deprecated Warp quiet setting
with Warp's documented log_level API. See
https://github.com/NVIDIA/warp/blob/main/design/deprecations.md#pending-removal
for the deprecation schedule. Remove this overlay when upstream ships the
equivalent fix.
"""

import os
import sys
import traceback
from importlib.metadata import distribution, entry_points
from pathlib import Path
from pkgutil import extend_path

# Keep the upstream mjlab subpackages discoverable when this module overlays
# the dependency's package initializer.
__path__ = extend_path(__path__, __name__)

_mjlab_distribution = distribution("mjlab")
if _mjlab_distribution.version != "1.6.0":
    raise RuntimeError(
        "The local mjlab compatibility overlay only supports mjlab==1.6.0; "
        f"found {_mjlab_distribution.version}"
    )

# The upstream module exposes this path for bundled robot and terrain assets.
# Point to the installed dependency, not this small overlay directory.
MJLAB_SRC_PATH: Path = Path(_mjlab_distribution.locate_file("mjlab")).resolve()

# Default to EGL for GPU-accelerated offscreen rendering on Linux. Must be set
# before any mujoco import: mujoco's gl_context module captures MUJOCO_GL once
# at load time. Override with e.g. MUJOCO_GL=osmesa on clusters without EGL.
# Linux-only because mujoco's gl_context rejects "egl" on macOS/Windows and
# raises at import. On those platforms we leave MUJOCO_GL alone so mujoco
# defaults to GLFW.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import tyro  # noqa: E402
import warp as wp  # noqa: E402

TYRO_FLAGS = (
    # Don't let users switch between types in unions. This produces a simpler CLI
    # with flatter helptext, at the cost of some flexibility. Type changes can
    # just be done in code.
    tyro.conf.AvoidSubcommands,
    # Disable automatic flag conversion (e.g., use --flag False instead of --no-flag).
    tyro.conf.FlagConversionOff,
    # Use Python syntax for collections: --tuple (1,2,3) instead of --tuple 1 2 3.
    # Helps with wandb sweep compatibility:
    # https://brentyi.github.io/tyro/wandb_sweeps/
    tyro.conf.UsePythonSyntaxForLiteralCollections,
)


def _configure_warp() -> None:
    """Configure Warp globally for mjlab."""
    wp.config.enable_backward = False

    # Preserve mjlab's upstream default and MJLAB_WARP_QUIET override using the
    # public replacement for warp.config.quiet.
    quiet = os.environ.get("MJLAB_WARP_QUIET", "0").lower() in ("1", "true", "yes")
    wp.config.log_level = wp.LOG_WARNING if quiet else wp.LOG_INFO


def _import_registered_packages() -> None:
    """Import packages registered through the mjlab.tasks entry-point group."""
    mjlab_tasks = entry_points().select(group="mjlab.tasks")
    for entry_point in mjlab_tasks:
        try:
            entry_point.load()
        except Exception:
            print(
                f"[WARN] Failed to load task package '{entry_point.name}' ({entry_point.value}):",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)


def _configure_mediapy() -> None:
    """Point mediapy at the bundled imageio-ffmpeg binary."""
    import imageio_ffmpeg
    import mediapy

    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


_configure_warp()
_configure_mediapy()
_import_registered_packages()
