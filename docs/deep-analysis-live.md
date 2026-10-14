# Deep Analysis: live PPO monitoring, Phase 2

## Identity and freshness contract

The catalog lists a run under **Live** only if a valid
`actual_ppo_rollout` sample was recorded within the last 10 seconds.
Older samples appear under *Recent sources (not streaming)* with their
actual timestamp. The V6 supervisor running a benchmark does **not**
count as training; neither do old dashboard runs with lifecycle state
`stopping`.

Source IDs use the form `alias~run-id` and resolve only into explicitly
mounted artifact roots. They are not filesystem paths supplied by clients.
Each source is labeled with its worktree alias and run-relative lineage.
Checkpoint comparisons and historical captures remain unchanged.

A live SSE connection sends fresh `frame` events and `status` events
when the source becomes stale or unavailable. The frontend reconnects
automatically, avoids duplicate and out-of-order frames, starts a new
history when the PPO sequence/global step resets, and separates episodes
on the sampled environment's reset/episode-step field. A one-second
freshness clock prevents an old snapshot appearing live if no SSE
update arrives.

## V6 on Marcus-computer

The dashboard container does not otherwise see the V6 worktree.
Mount its artifact root read-only by adding the optional overlay:

    cd /root/Ascento_MuJoCo_Playground/docker
    docker compose -f compose.yaml -f compose.gpu.yaml \
      -f compose.tailscale.yaml -f compose.deep-live-v6.yaml \
      up -d --no-deps --no-build --force-recreate dashboard

This does not restart the database, Tailscale, supervisor or GPU workers.
When a new V6 training run writes live_monitor.json beneath its training
directory, the monitor becomes eligible automatically. When it stops
writing, it is removed from the live selection within 10 seconds.

Other worktrees can be connected with another explicitly configured,
read-only artifact mount and an alias in `ASCENTO_DEEP_LIVE_ROOTS`.
The server never scans arbitrary host folders or launches a policy.

## Validation

    .venv/bin/python -m pytest tests_dashboard/test_deep_live.py
    cd dashboard/frontend && npm run typecheck && npm run build

During periods with no active monitored PPO run, verify that the recent
V6 source is visible as *not streaming*, and do not interpret the
supervisor's independent evaluation as live training. A real streaming
end-to-end check still requires actual fresh PPO training samples.
