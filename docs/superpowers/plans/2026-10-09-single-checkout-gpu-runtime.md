# Single-checkout GPU runtime

## Goal

Make `/root/Ascento_MuJoCo_Playground` on clean, current `main` the only native training checkout. Treat the Windows clone as a synchronized editor mirror and Docker as an immutable build from that WSL checkout. Select CUDA explicitly for training/evaluation when the workload and host support it, while leaving the web dashboard itself CPU-capable. Make run provenance identify the canonical source and actual executor.

## Steps

1. Add test-first runtime policy coverage for rejecting a second, stale, or dirty checkout; validating a Docker image against the mounted canonical revision manifest; and recording execution/source provenance.
2. Implement the runtime policy and integrate it with dashboard/CLI managed run creation and launch; choose CUDA for training by default and fail early with clear guidance if a GPU-backed launch cannot access CUDA.
3. Add an external CUDA venv launcher; route preflight and the waypoint sweep through it; change maintenance to update that venv without replacing the MCP service venv.
4. Pin Compose runtime provenance to the canonical revision manifest, keep GPU compose as the maintained training image, avoid local port collisions, and update operational docs.
5. Verify targeted tests and runtime checks, commit/push `main`, synchronize the Windows mirror, and archive saved outputs plus Git recovery refs before removing stale linked worktrees.

## Safety

Do not stop the active MCP processes, start training, alter remote Tailscale state, or delete unarchived `logs`, `checkpoints`, `captures`, `evaluations`, or other saved outputs. Preserve branch commits in Git refs when retiring worktrees.
