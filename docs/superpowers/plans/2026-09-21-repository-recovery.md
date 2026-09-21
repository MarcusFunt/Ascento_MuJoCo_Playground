# Repository Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover local source into a clean branch and prevent non-reproducible managed training runs.

**Architecture:** Preserve the original checkout and its run outside the repository, apply source-only changes in this isolated worktree as focused commits, and add a shared provenance guard at the managed-launch boundary.

**Tech Stack:** Git worktrees, Python 3.12, pytest, Ruff, Bash, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-15-repository-recovery-design.md`

## Global Constraints

- Never reset, clean, delete, or force-push the original checkout.
- Preserve the active run and a valid full source archive before stopping it.
- Commit code, tests, suites, and benchmark methodology; ignore checkpoints, logs, evaluations, transfers, and generated reports.
- A dirty run requires an explicit override and a complete provenance bundle.
- CI uses Python 3.12; GPU validation is a manually dispatched self-hosted job.

## Review Focus

- A tracked dirty file rejects a run before a directory or subprocess exists (Task 4).
- An untracked source file rejects a run by default (Task 4).
- An override bundles a valid patch and hashes every nonignored untracked path (Task 3).
- A failed bundle aborts launch rather than records ambiguous provenance (Task 4).
- Generated data is ignored while recovered source files are committed (Task 2).

---

### Task 1: Archive and stop the dirty-tree run

**Files:**
- Create outside repo: `/root/ascento-recovery/20260921-dirty-checkout/`
- Modify: managed run status for `02c0f463da7a`

**Interfaces:** Produces `tracked.patch`, `untracked.tar`, a hash manifest, and a copy of the managed run directory outside Git.

- [ ] **Step 1: Capture run status and source state**

```bash
ascento run status 02c0f463da7a --json > /root/ascento-recovery/20260921-dirty-checkout/run-status-before-stop.json
git diff --binary HEAD > /root/ascento-recovery/20260921-dirty-checkout/tracked.patch
git ls-files --others --exclude-standard -z > /root/ascento-recovery/20260921-dirty-checkout/untracked-paths.zlist
tar --null --files-from=/root/ascento-recovery/20260921-dirty-checkout/untracked-paths.zlist -cf /root/ascento-recovery/20260921-dirty-checkout/untracked.tar
sha256sum /root/ascento-recovery/20260921-dirty-checkout/tracked.patch /root/ascento-recovery/20260921-dirty-checkout/untracked.tar
```

- [ ] **Step 2: Validate the tracked patch in a clean temporary worktree**

```bash
git -C /root/Ascento_MuJoCo_Playground/.worktrees/hygiene-recovery apply --check /root/ascento-recovery/20260921-dirty-checkout/tracked.patch
```

Expected: exit status 0; the replacement archive is valid even though mjlab's prior patch is corrupt.

- [ ] **Step 3: Stop only after archive validation**

```bash
ascento run stop 02c0f463da7a --reason repository_recovery_dirty_source
ascento run status 02c0f463da7a --json
```

- [ ] **Step 4: Copy complete run evidence outside the repo**

```bash
cp -a logs/rsl_rl/20260915_134448_locomotion-10k-settle-push-step-from-79-999-retr_02c0f463 /root/ascento-recovery/20260921-dirty-checkout/run-02c0f463
```

- [ ] **Step 5: Commit**

No Git commit; this is forensic preservation only.

### Task 2: Import source and establish artifact policy

**Files:**
- Modify: `.gitignore`, `docs/training.md`, `docs/cli-reference.md`, `docs/cli-reference.json`
- Create: recovered source, tests, suites, and methodology docs
- Create: `scripts/install_git_hooks.sh`, `scripts/git-hooks/pre-commit`
- Test: `tests_dashboard/test_repository_hygiene.py`

**Interfaces:** Source under `benchmarks/`, `src/`, `tests_*/`, and product docs is tracked. `reports/`, `transfers/`, `.worktrees/`, and `.codex/recovery/` are ignored.

- [ ] **Step 1: Write failing artifact-policy tests**

```python
def test_generated_artifact_directories_are_ignored(repo_root):
    assert check_ignore(repo_root, "reports")
    assert check_ignore(repo_root, "transfers")
    assert check_ignore(repo_root, ".worktrees")

def test_recovered_locomotion_source_is_tracked(repo_root):
    assert tracked(repo_root, "benchmarks/suites/locomotion_sequence_gate_v1.toml")
    assert tracked(repo_root, "src/ascento_mjlab/tasks/locomotion/env_cfg.py")
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests_dashboard/test_repository_hygiene.py`

Expected: FAIL because recovered source and ignore rules do not yet exist.

- [ ] **Step 3: Mechanically apply the archive and classify paths**

Apply `tracked.patch`; extract only source/test/docs/suite paths from `untracked.tar`. Do not import `reports/` or model checkpoints. Add:

```gitignore
.worktrees/
reports/
transfers/
.codex/recovery/
```

Move retained transfer checkpoints to `logs/transfers/`; use checkpoint placeholders in docs rather than repository-root `transfers/` paths.

- [ ] **Step 4: Add tracked hook installation**

`scripts/install_git_hooks.sh` copies `scripts/git-hooks/pre-commit` to `.git/hooks/pre-commit`. The hook runs:

```bash
git diff --cached --check
uv run --frozen --extra cpu --extra dashboard ruff check .
```

- [ ] **Step 5: Verify and commit**

Run: `pytest -q tests_dashboard/test_repository_hygiene.py && bash -n scripts/install_git_hooks.sh scripts/git-hooks/pre-commit && git diff --check`

```bash
git add .gitignore benchmarks src tests_mjlab tests_dashboard docs scripts
git commit -m "feat: recover source and establish artifact policy"
```

### Task 3: Implement complete dirty-source bundles

**Files:**
- Create: `dashboard/provenance.py`
- Test: `tests_dashboard/test_provenance.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class WorkingTreeState:
    commit: str | None
    branch: str | None
    tracked_paths: tuple[str, ...]
    untracked_paths: tuple[str, ...]

def working_tree_state(repo_root: Path) -> WorkingTreeState: ...
def write_dirty_source_bundle(repo_root: Path, destination: Path) -> dict[str, Any]: ...
```

- [ ] **Step 1: Write failing bundle tests**

```python
def test_state_reports_tracked_and_untracked_paths(initialized_repo):
    state = working_tree_state(initialized_repo)
    assert state.tracked_paths == ("tracked.py",)
    assert state.untracked_paths == ("new.py",)

def test_bundle_has_valid_patch_and_hashed_untracked_file(initialized_repo, tmp_path):
    bundle = write_dirty_source_bundle(initialized_repo, tmp_path / "bundle")
    assert (tmp_path / "bundle" / "tracked.patch").is_file()
    assert bundle["untracked"]["new.py"]["sha256"]
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests_dashboard/test_provenance.py`

Expected: FAIL because `dashboard.provenance` is absent.

- [ ] **Step 3: Implement and verify GREEN**

Use `git status --porcelain=v1 --untracked-files=all` and `git diff --binary HEAD`. Write an atomic manifest with SHA-256 hashes and archive each nonignored untracked path. Reject failed subprocesses and invalid patches.

Run: `pytest -q tests_dashboard/test_provenance.py`

- [ ] **Step 4: Commit**

```bash
git add dashboard/provenance.py tests_dashboard/test_provenance.py
git commit -m "feat: capture complete dirty-source provenance"
```

### Task 4: Guard every managed run entry point

**Files:**
- Modify: `dashboard/launch.py`, `dashboard/run_service.py`, `dashboard/app.py`, `src/ascento_mjlab/cli.py`, `src/ascento_mjlab/mcp_server.py`
- Modify: `docs/cli-reference.md`, `docs/cli-reference.json`, `docs/training.md`
- Test: `tests_dashboard/test_config_launch.py`, `tests_dashboard/test_run_service.py`, `tests_dashboard/test_cli.py`, `tests_dashboard/test_run_api.py`

**Interfaces:** Adds `--allow-dirty-provenance`; API/MCP requests contain `allow_dirty_provenance: bool = False`; manifests contain `source_provenance.mode` of `clean_commit` or `dirty_bundle`.

- [ ] **Step 1: Write failing admission tests**

```python
def test_launcher_rejects_dirty_tree_without_override(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "working_tree_state", lambda _: dirty_state())
    with pytest.raises(ValueError, match="allow-dirty-provenance"):
        launch.validate_source_provenance(tmp_path, allow_dirty_provenance=False)

def test_launcher_writes_bundle_only_when_explicitly_allowed(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "working_tree_state", lambda _: dirty_state())
    assert launch.validate_source_provenance(tmp_path, allow_dirty_provenance=True)["mode"] == "dirty_bundle"
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests_dashboard/test_config_launch.py -k provenance`

Expected: FAIL because the guard and override do not exist.

- [ ] **Step 3: Implement the guard**

Call `validate_source_provenance` before a run directory or subprocess is created. Thread the override through CLI, dashboard request, RunService command, MCP request, and launcher parser. A clean tree records its commit; a dirty override writes Task 3's bundle below the run directory.

- [ ] **Step 4: Verify and commit**

Run: `pytest -q tests_dashboard/test_config_launch.py tests_dashboard/test_run_service.py tests_dashboard/test_cli.py tests_dashboard/test_run_api.py`

```bash
git add dashboard src/ascento_mjlab/cli.py src/ascento_mjlab/mcp_server.py tests_dashboard docs
git commit -m "feat: reject dirty-source managed training by default"
```

### Task 5: Align CI and document the recovery workflow

**Files:**
- Modify: `.github/workflows/dashboard-ci.yml`
- Create: `.github/workflows/gpu-smoke.yml`, `tests_dashboard/test_ci_workflows.py`, `docs/repository-hygiene.md`

**Interfaces:** Standard CI uses Python 3.12. GPU smoke is `workflow_dispatch` on `[self-hosted, linux, gpu]` only.

- [ ] **Step 1: Write failing workflow tests**

```python
def test_ci_uses_python_312():
    assert workflow_text("dashboard-ci.yml").count("python-version: '3.12'") == 3

def test_gpu_smoke_is_manual_and_labeled():
    text = workflow_text("gpu-smoke.yml")
    assert "workflow_dispatch:" in text
    assert "[self-hosted, linux, gpu]" in text
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests_dashboard/test_ci_workflows.py`

Expected: FAIL because CI uses Python 3.11 and no GPU workflow exists.

- [ ] **Step 3: Implement workflows and guide**

Change all standard CI Python versions to 3.12. Add the manual self-hosted GPU smoke workflow using `--extra cu128 --extra dashboard`. Document worktree setup, hook installation, clean-run admission, dirty-run override, and artifact locations.

- [ ] **Step 4: Verify and commit**

Run: `pytest -q tests_dashboard/test_ci_workflows.py && git diff --check`

```bash
git add .github docs tests_dashboard
git commit -m "ci: align Python and document GPU smoke"
```

### Task 6: Final verification and remote readiness

**Files:** all recovered changes

- [ ] **Step 1: Confirm policy state**

```bash
git status --short
git check-ignore reports transfers .worktrees
git ls-files benchmarks/suites src/ascento_mjlab/tasks/locomotion
```

Expected: only intentional tracked changes; artifacts ignored; task/suite files tracked.

- [ ] **Step 2: Run full validation**

```bash
uv run --frozen --extra cpu --extra dashboard pytest -q tests_mjlab tests_dashboard
uv run --frozen --extra cpu --extra dashboard ruff check .
git diff --check
```

- [ ] **Step 3: Commit final guide and request review**

```bash
git add docs/repository-hygiene.md
git commit -m "docs: record reproducible repository workflow"
```

Do not delete remote branches or change branch protection without a separate user decision. Request review and open a PR from `hygiene/recovery`.
