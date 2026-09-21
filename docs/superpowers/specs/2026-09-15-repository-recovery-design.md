# Repository Recovery Design

## Goal

Restore a clean, reviewable, reproducible development workflow without
discarding the current local implementation or experiment evidence.

## Scope and invariants

- Preserve existing logs, checkpoints, evaluator artifacts, and the current
  dirty checkout before any cleanup.
- Do not train, evaluate, or capture from an uncommitted source tree after the
  recovery is installed.
- Keep source, tests, task definitions, benchmark suites, and baseline
  documents under version control.
- Keep generated reports, checkpoints, transfers, logs, captures, and
  evaluations out of version control while retaining clear local locations.
- Use one clean branch/worktree for recovery integration; do not rewrite or
  reset the original checkout.

## Recovery sequence

1. Gracefully stop the active dirty-tree run and archive its provenance:
   run metadata, resolved environment/agent configurations, checkpoints, logs,
   and a complete archive of both tracked and untracked source material.
2. Classify the archived material.  Commit implementation and test changes in
   reviewable slices; retain generated data as managed artifacts only.
3. Add repository policies that prevent recurrence: an ignored local-worktree
   directory, generated-artifact rules, tracked hook installation, and a
   managed-run dirty-tree guard.
4. Repair managed-run provenance.  A clean run records its exact Git revision;
   an explicitly allowed dirty run records a complete bundle of tracked diff,
   untracked source files, and a manifest with hashes.  A malformed/truncated
   diff cannot be accepted as provenance.
5. Align automated validation with the supported Python 3.12 environment and
   provide an explicit, separately dispatched GPU smoke workflow for hosts
   carrying the required GPU runner labels.

## Artifact classification

| Material | Policy | Canonical location |
| --- | --- | --- |
| Python/JS source, tests, benchmark suites, baseline methodology docs | Commit | repository source tree |
| Training logs, checkpoints, evaluations, captures | Ignore | `logs/`, `evaluations/`, `captures/` |
| Actor-transfer checkpoints and manifests | Ignore, retain locally | `logs/transfers/` |
| Generated review reports and frontend build products | Ignore | `reports/`, frontend build directories |
| Process-only design/plan notes | Ignore | `.codex/recovery/` |

## Managed-run contract

`ascento run start` must fail when the working tree has tracked or untracked
source changes unless the caller supplies an explicit dirty-run override.  The
override is accepted only after the launcher creates a complete provenance
bundle in the run directory.  The normal path records the Git commit SHA and
requires a clean working tree.

## Verification

- Unit tests cover clean-run admission, dirty-run rejection, and dirty-run
  bundle completeness.
- The runner rejects an invalid/truncated provenance patch.
- CI uses Python 3.12 for its standard CPU suite.
- A workflow-dispatch GPU smoke job is available for a labeled self-hosted
  runner; it is not represented as hosted-GitHub GPU coverage.
- Full repository tests and lint pass from the clean recovery worktree.

## Non-goals

- Recovering or promoting the active dirty-tree 10k run.
- Deleting historical logs or evaluator artifacts.
- Rewriting `main` history or force-pushing remote branches.
