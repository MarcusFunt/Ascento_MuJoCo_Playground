# Waypoint Heading Diversity Sweep Implementation Plan

> **For agentic workers:** Use the executing-plans workflow to implement this plan. Steps use checkbox syntax to track progress.

**Goal:** Run a bounded, matched sweep of independent waypoint-heading diversity and identify whether it improves multi-waypoint driving without damaging balance.

**Architecture:** Add an opt-in task family that changes only the final heading distribution for regular long-range targets. A foreground orchestration script will create identical actor transfers, train three sequential arms, run four immutable suites for every checkpoint, and conditionally extend the best nonzero heading mixture within a nine-hour limit.

**Tech Stack:** Python, Ascento CLI, mjlab/MuJoCo Warp, PyTorch CUDA, immutable TOML evaluation suites, Codex Process Jobs.

**Spec:** `docs/superpowers/specs/2026-10-06-waypoint-heading-sweep.md`

## Global Constraints

- Preserve the gate-hold model_299 checkpoint as an experiment; do not overwrite it or change the default task.
- The only experimental variable is the fraction of regular targets with a ±90° final-heading offset: 0%, 12.5%, or 25%.
- Keep seed 73, 512 environments, rewards, action/observation ABI, target distances, 25% gate-hold subset, and PPO configuration fixed.
- Use uppercase `True` for `--agent.resume`.
- Stop all new work by the nine-hour wall-clock limit.

## Review Focus

- Reset and repeated regular targets must use the same offset fraction; gate-hold retargets must retain current heading.
- A 0% arm must be behaviorally equivalent to the existing heading rule, apart from matched random-number consumption.
- Candidate checkpoints must be transferred to each suite's registered task before evaluation, preserving immutable suite compatibility.
- A valid FAIL report is data; INVALID or incomplete reports cannot support promotion or extension.
- Deadline handling must request graceful managed-run stops and must not leave an evaluation running outside the job.

---

### Task 1: Add an opt-in heading-mixture task family

**Files:**
- Modify: `src/ascento_mjlab/mdp/events.py`
- Modify: `src/ascento_mjlab/tasks/locomotion/env_cfg.py`
- Modify: `src/ascento_mjlab/tasks/__init__.py`
- Modify: `dashboard/task_catalog.py`
- Modify: `src/ascento_mjlab/viewer/worker.py`

**Interfaces:**
- Add `quarter_turn_heading_fraction: float | None = None` to regular-target initialization and sequence configuration; sweep tasks set it explicitly, including 0.0 for the matched control.
- Add a task config builder that accepts the fraction and task ID; register 0%, 12.5%, and 25% task IDs.

- [x] Validate the fraction in `[0, 1]`, sample a signed ±π/2 offset for each regular target, and apply it to both reset and repeated target headings. Consume the same random draws at fraction zero to keep matched arms aligned.
- [x] Register the three opt-in tasks in mjlab, the dashboard catalog, and the waypoint-capable viewer set.
- [x] Inspect the resolved config contract to confirm only task identity and the heading-mixture parameter differ among arms.

### Task 2: Add the bounded sweep controller and plan record

**Files:**
- Create: `scripts/overnight_waypoint_heading_sweep.py`
- Create: `docs/superpowers/specs/2026-10-06-waypoint-heading-sweep.md`
- Create: `docs/superpowers/plans/2026-10-06-waypoint-heading-sweep.md`

- [ ] Start three 5,000-iteration runs sequentially from actor-only transfers of the same preserved model_299 checkpoint, with seed 73 and 512 environments.
- [ ] Transfer each completed checkpoint to the task required by each of the four immutable suites and retain complete evaluation manifests/reports.
- [ ] Compare gate counts and paired metrics. Extend the strongest nonzero mixture by 3,000 iterations only when waypoint gates improve and sequence/Balance hard-gate counts do not worsen.
- [ ] Enforce a nine-hour deadline, write machine-readable run/evaluation/decision summaries, and gracefully stop a managed run if the deadline is reached.

### Task 3: Launch the overnight process job

**Files:**
- No further source changes; experiment artifacts are written below ignored `sweeps/`, `logs/`, `checkpoints/`, and `evaluations/` roots.

- [x] Confirm the worktree is clean and the base CUDA environment imports PyTorch with the RTX 3060 available.
- [ ] Launch the foreground sweep controller through Codex Process Jobs and return immediately with the job ID.
