# Repository Reconciliation and Clean Closeout Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the recovered implementation into `origin/main` and leave the user's main checkout clean without losing source changes, run evidence, reports, or transfer checkpoints.

**Architecture:** Continue from the existing clean `hygiene/recovery` worktree and its eight recovery commits. Verify the archived dirty checkout, incorporate the one local-main-only commit and any newly fetched upstream commits, run all applicable validation, and review through a PR. Only after the source is present in the merged revision should the original checkout be normalized; preserve local artifacts in place and make them ignored rather than deleting them.

**Tech Stack:** Git in WSL, `uv`, Python 3.12, pytest, Ruff, npm/Vite, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-15-repository-recovery-design.md`

## Global Constraints

- Preserve `/root/ascento-recovery/20260921-dirty-checkout/` and its checksummed archives until the user confirms the final checkout is correct.
- Do not use `git reset --hard`, `git clean`, force-push, delete branches/worktrees, or delete generated artifacts.
- Do not start training, run evaluations, or capture videos during repository recovery.
- Commit source, tests, benchmark suites, and methodology documentation; keep reports, transfers, logs, evaluations, and captures local and ignored.
- Do not overwrite the dirty original checkout until every source delta has been compared with the recovery branch and the user approves final normalization.
- Never stage all files blindly; inspect staged paths before every recovery commit or PR.

## Current Evidence

- Original checkout: `main` at `cfdf237`, one commit ahead of the locally recorded `origin/main`, with modified tracked files and untracked source/data.
- Recovery worktree: `.worktrees/hygiene-recovery` was clean at `cf45d4b`, eight commits ahead of the locally recorded `origin/main`, before this planning document was added. This document is its only new uncommitted path.
- The two branches share the recorded `origin/main` base; the only main-only commit is `cfdf237` (`chore: ignore local worktrees`).
- The external recovery archive exists and all four entries in `SHA256SUMS` verified successfully.
- No managed run was active in the most recent inspection. The archived run has before/after status records and its run directory.
- `reports/` contains generated report output and `transfers/` contains local checkpoint files. Preserve both; neither belongs in source history.
- The original checkout still has local changes. It has not been cleaned or overwritten.

## Review Focus

- A tracked or untracked source delta exists only in the dirty checkout and was not recovered: compare paths and contents against the verified patch/archive before normalization.
- `.gitignore` reconciliation drops an existing rule: retain the local-worktree rule and the generated-artifact rules together.
- A current or upstream `main` commit appeared after the last fetch: fetch and inspect both sides before merging.
- Generated artifacts are accidentally staged or moved: verify report/checkpoint paths and ignore status while leaving files in place.
- A backend, CLI, dashboard, or frontend regression is missed: run the full Python tests, Ruff, and the Vite production build before opening the PR.

---

### Task 1: Revalidate the preserved recovery evidence

**Files:**
- Read: `/root/ascento-recovery/20260921-dirty-checkout/SHA256SUMS`
- Read: `/root/ascento-recovery/20260921-dirty-checkout/run-status-after-stop.json`
- Read: current Git status in both checkouts

**Interfaces:** Produces a verified source/run archive and a current branch/upstream inventory. Do not copy or restore files in this task.

- [x] **Step 1: Recheck archive integrity**

Run: `cd /root/ascento-recovery/20260921-dirty-checkout && sha256sum -c SHA256SUMS`

Expected: `tracked.patch`, `tracked-from-origin-main.patch`, `local-head.bundle`, and `untracked.tar` all report `OK`.

- [x] **Step 2: Recheck run and worktree state**

Run: `ascento run list --active --json`

Run: `git -C /root/Ascento_MuJoCo_Playground/.worktrees/hygiene-recovery status --short --branch`

Expected: no active runs; in the recovery worktree, this closeout plan may be the only uncommitted path. If any other path differs, stop and investigate before changing refs.

- [x] **Step 3: Refresh upstream refs and record divergence**

Run: `git -C /root/Ascento_MuJoCo_Playground fetch origin`

Run: `git -C /root/Ascento_MuJoCo_Playground log --oneline --left-right main...origin/main`

Run: `git -C /root/Ascento_MuJoCo_Playground log --oneline --left-right origin/main...hygiene/recovery`

Expected: the updated commit lists make clear whether upstream advanced and which local commits must be preserved. Do not infer current remote state from the older local `origin/main` ref.

### Task 2: Reconcile commit history on the recovery branch

**Files:**
- Modify only if needed: `.gitignore` in the recovery worktree
- Preserve: all existing recovery commits and the local-main-only commit

**Interfaces:** Produces one review branch containing the current upstream tip, the `cfdf237` local-main commit, and the recovered source commits without rewriting existing history.

- [x] **Step 1: Merge the refreshed upstream tip into the recovery branch**

Run from `/root/Ascento_MuJoCo_Playground/.worktrees/hygiene-recovery`: `git merge --no-ff origin/main`

Expected: merge succeeds or stops on explicit conflicts. If `.gitignore` conflicts, retain `.worktrees/`, `reports/`, `transfers/`, `.codex/recovery/`, and all unrelated pre-existing patterns.

- [x] **Step 2: Merge the local-main-only commit**

Run from the recovery worktree: `git merge --no-ff main`

Expected: `cfdf237` is now an ancestor of `hygiene/recovery`. Resolve any `.gitignore` conflict by preserving the union of rules; do not drop any recovered source files.

- [x] **Step 3: Inspect the combined tree before testing**

Run: `git status --short --branch`

Run: `git diff --check origin/main...HEAD`

Run: `git diff --stat origin/main...HEAD`

Expected: the recovery worktree contains only intentional source/docs/tests/policy changes (including this closeout plan); `reports/`, `transfers/`, run data, and captures are not staged or tracked. Commit the reviewed closeout plan as a documentation-only commit before opening the PR.

### Task 3: Prove the recovered change set is complete and valid

**Files:**
- Validate all files changed by the recovery branch
- Do not modify generated artifact directories

**Interfaces:** Produces test, lint, and frontend-build results attached to the PR review.

- [x] **Step 1: Compare the archived dirty source with the integrated branch**

Compare each path in `source-status-porcelain.txt`, `untracked-paths.zlist`, and `untracked-archive-members.txt` with the integrated recovery branch. Confirm each source/test/doc/suite change is either represented in the branch or explicitly classified as local generated data. If any source change is missing or differs, recover that exact path from the archive into the recovery worktree, test it, and commit it before proceeding.

- [x] **Step 2: Run the full Python test suite**

Run from the recovery worktree: `uv run --frozen --extra cpu --extra dashboard pytest -q tests_mjlab tests_dashboard`

Expected: all tests pass. Investigate and fix failures on the recovery branch; do not weaken assertions to obtain a pass.

- [x] **Step 3: Run lint and whitespace checks**

Run: `uv run --frozen --extra cpu --extra dashboard ruff check .`

Run: `git diff --check`

Expected: both pass with no unreviewed generated files in the diff.

- [x] **Step 4: Build the dashboard frontend**

Run from `dashboard/frontend`: `npm ci && npm run build`

Expected: Vite exits successfully and creates only its normal ignored build output.

### Task 4: Review and integrate through the remote

**Files:** all recovered source changes, tests, documentation, CI, and ignore policy

**Interfaces:** Produces an approved PR from `hygiene/recovery` to `main`; no remote branch is rewritten.

- [ ] **Step 1: Review the complete PR diff**

Check the file list and diff against the refreshed `origin/main`. Confirm generated reports, checkpoints, transfers, logs, evaluations, and captures are absent from version control. Confirm the run-provenance and dirty-tree guard paths are tested at CLI, API/MCP, and launcher boundaries.

- [ ] **Step 2: Open the recovery PR and wait for review/CI**

Use the `hygiene/recovery` branch as the PR source. Do not merge until required tests and review pass. Prefer a merge method that preserves the local-main commit as ancestry; do not force-push or delete either branch.

- [ ] **Step 3: Confirm the merged remote revision**

Run: `git fetch origin`

Record `git rev-parse origin/main` and confirm the recovered source, policies, tests, and local-main-only commit are present in the remote tree.

### Task 5: Normalize the original checkout without losing local data

**Files:**
- Original checkout at `/root/Ascento_MuJoCo_Playground`
- Preserve local artifact directories and the external recovery archive

**Interfaces:** Produces a clean local `main` whose commit equals the fetched `origin/main`, with existing ignored local artifacts retained on disk.

- [ ] **Step 1: Reconcile remaining root-checkout differences against the merged tree**

Record `git status --porcelain=v1 --untracked-files=all`, compare every modified tracked path and nonignored untracked source path with the merged `origin/main`, and compare hashes with the verified external archive. If any change is not represented remotely, stop and preserve/import it before cleanup.

- [ ] **Step 2: Preserve local data while clearing only reconciled source deltas**

Keep `reports/`, `transfers/`, `logs/`, `evaluations/`, and `captures/` in place. Confirm the merged `.gitignore` ignores them. For each remaining source path that conflicts with a newly tracked path, make a reversible copy outside the checkout, then update only the enumerated tracked paths from the verified merged revision. Do not run `git clean` or use a broad reset.

- [ ] **Step 3: Fast-forward the local main and verify the result**

Run: `git -C /root/Ascento_MuJoCo_Playground switch main`

Run: `git -C /root/Ascento_MuJoCo_Playground merge --ff-only origin/main`

Run: `git -C /root/Ascento_MuJoCo_Playground status --short --branch`

Run: `git -C /root/Ascento_MuJoCo_Playground rev-parse HEAD origin/main`

Expected: `main` is clean and `HEAD` equals `origin/main`; ignored local artifacts still exist. If ancestry prevents fast-forwarding, stop rather than rewriting the branch; retain the clean recovery worktree and request a merge-strategy decision.

- [ ] **Step 4: Retain recovery assets until user confirmation**

Keep the external archive, recovery branch, and worktree until the user confirms the clean checkout and local artifacts are intact. Only then discuss whether to remove the extra worktree or archive; do not delete either as part of this plan.

## Completion Criteria

- Recovery PR is merged and fetched; all intended source changes are on `origin/main`.
- Full Python tests, Ruff, and frontend production build pass on the integrated revision.
- Local `main` has no tracked or nonignored untracked changes and `HEAD` equals `origin/main`.
- Reports, transfer checkpoints, training/evaluation data, and the external recovery archive remain present and are not tracked.
