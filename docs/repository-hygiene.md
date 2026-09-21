# Repository hygiene and reproducible runs

Develop from a named branch or an isolated worktree. Keep the original
checkout intact when recovering a dirty experiment; never reset or clean it as
a shortcut.

Install the repository hook once per shared Git directory:

```bash
bash scripts/install_git_hooks.sh
```

The installer preserves a different existing pre-commit hook and exits with
instructions rather than replacing it. Combine or back up that hook before
re-running the installer.

The hook rejects whitespace errors and runs Ruff. Source, tests, task
definitions, benchmark suites, and baseline methodology are committed.
Generated logs, checkpoints, captures, evaluations, reports, and transfer
artifacts remain local. Store transfer artifacts below `logs/transfers/`.

Managed training requires a clean working tree. A normal run records its Git
commit. The maintained Docker image excludes `.git`; it therefore records the
commit, branch, and clean-build status in image build metadata. Unknown or dirty
image provenance blocks managed runs. If a dirty exploratory run is genuinely required, make the exception
visible and reproducible:

```bash
ascento run start --allow-dirty-provenance --task Ascento-Balance-Flat
```

That override writes the tracked patch, an archive of all nonignored untracked
files, and SHA-256 manifests below the run before the trainer starts. It is not
a replacement for committing source before a benchmark or promotion decision.

Standard CI uses Python 3.12 and CPU dependencies. GPU validation is a manual
workflow-dispatch job restricted to the labeled self-hosted GPU runner; it is
not represented as hosted-GitHub GPU coverage.
