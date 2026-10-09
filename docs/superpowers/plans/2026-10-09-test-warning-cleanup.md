# Test Warning Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the test-suite warnings and the test-client collection blocker in the verified canonical checkout without hiding diagnostics or changing application and simulator behavior.

**Architecture:** Resolve project-owned warnings directly in their source and configuration. Replace deprecated FastAPI startup/shutdown hooks with one lifespan manager, then address the Starlette test-client and mjlab/Torch warnings through compatible dependency updates or a narrowly maintained upstream fix. Finish with a warning-as-error full-suite run and the existing runtime smoke checks.

**Tech Stack:** Python 3.11–3.13, pytest, FastAPI 0.141.1, Starlette 1.6.0, Alembic 1.20.0, mjlab 1.6.0, PyTorch, uv.

**Spec:** Prior full test-suite report of 183 warnings: 140 FastAPI lifecycle deprecations, Torch JIT deprecations emitted through mjlab, 6 Alembic `path_separator` warnings, 1 invalid-escape `SyntaxWarning` in `tests_mjlab/test_waypoints.py`, and a Starlette TestClient warning recommending `httpx2`.

The first implementation task reconciles the prior report with the exact checkout and dependency environment being changed. In canonical WSL `origin/main` at `6f27b7f`, the suite currently fails collection in `tests_dashboard/test_backend.py` and `tests_dashboard/test_run_api.py`: Starlette 1.6.0 requires `httpx2`, which is absent from the project dependencies. The baseline also confirms the waypoint invalid-escape warning. The Windows OneDrive checkout is behind and dirty, so it is not the remediation target. Execute Task 3 first because the missing dependency prevents the full baseline and every dashboard test from collecting; after it passes, rerun Task 0 and continue with Tasks 1, 2, and 4.

## Global Constraints

- Keep the dashboard dependency constraint `fastapi==0.141.1` unless the TestClient compatibility investigation proves a coordinated version change is required.
- Preserve `mjlab==1.6.0` behavior and the supported Python range `>=3.11,<3.14` unless a compatible, verified upstream release is selected.
- Keep CPU and CUDA 12.8 (`cu128`) dependency resolutions reproducible in `uv.lock`.
- Do not add global warning filters or broad `DeprecationWarning`/`SyntaxWarning` suppressions.
- Startup must initialize the dashboard database and preserve the existing fallback warning behavior; shutdown must stop managed viewers and dispose the database.
- The final full test run must pass with warnings treated as errors; no warning is considered resolved merely because its output was filtered.

## Review Focus

- Lifespan startup/shutdown must run exactly once for each application lifespan and always dispose resources when startup or request handling fails; cover both normal shutdown and startup failure.
- TestClient response, streaming, and lifespan behavior must remain the same after changing its HTTP client backend; cover the existing dashboard API and streaming tests.
- Any `mjlab` or PyTorch update/patch must preserve CPU and CUDA execution; cover import/initialization, a CPU simulation smoke, and the existing CUDA smoke where hardware is available.
- Alembic must still resolve migration paths consistently on Windows and Linux; cover a migration configuration/upgrade invocation on both supported path styles in CI.
- The corrected waypoint regex must continue matching the same validation message; run the waypoint tests with `SyntaxWarning` promoted to an error.

---

### Task 0: Reproduce and bind the warning report to the target checkout

**Files:**
- No source changes; record the canonical checkout path, branch, commit, Python version, and resolved dependency versions in the implementation notes.

**Interfaces:**
- Consumes: the recorded 183-warning report and the checkout selected for remediation.
- Produces: a fresh warning inventory with stack traces, grouped by source and count, for the exact checkout that will be edited.

- [x] Identify the authoritative checkout for the test run and record its path, branch, commit, working-tree state, Python version, and `uv.lock` resolution; do not infer that the Windows checkout and WSL checkout are interchangeable.
- [x] Run `uv run --extra cpu --extra dashboard --group dev python -m pytest -q` without warning filters and capture the complete warning summary and first stack trace for each distinct warning. On the starting commit, record the expected TestClient collection failure due to absent `httpx2`; rerun this command after Task 3 to obtain the full inventory.
- [x] Confirm whether the report still contains all five recorded warning groups and whether the regex and TestClient warnings still map to the files named below. Update this plan’s paths/counts before implementation if the checkout or warning inventory differs.
- [x] If the full run does not reproduce a warning group, identify whether it is already fixed, belongs to another checkout/environment, or was caused by a transient dependency; do not add speculative source edits for it.

### Task 1: Remove the test regex and Alembic configuration warnings

**Files:**
- Modify: `tests_mjlab/test_waypoints.py` at the invalid-escape regex reported by the warning output.
- Modify: `alembic.ini` in the `[alembic]` section.

**Interfaces:**
- Consumes: the existing waypoint validation assertion and current Alembic configuration.
- Produces: a raw-string regex with identical matching semantics and an explicit platform-native Alembic path separator.

- [x] Change the test regex to a raw string, preserving the escaped square brackets and the expected text `within [0, 0.4]`.
- [x] Add `path_separator = os` under `[alembic]`; do not change migration paths or database URLs.
- [x] Run `uv run --extra dashboard --extra cpu --group dev python -W error::SyntaxWarning -m pytest tests_mjlab/test_waypoints.py -q`.
- [x] Run `uv run --extra cpu --extra dashboard --group dev python -m pytest tests_dashboard/test_database.py -q -W error::DeprecationWarning`; confirm Alembic emits no path-separator warning. Run full dashboard tests after Task 2 removes FastAPI warnings.

### Task 2: Replace FastAPI event hooks with a lifespan manager

**Files:**
- Modify: `dashboard/app.py` where `app = FastAPI(...)` and the startup/shutdown event handlers are defined near the bottom of the module.
- Create or modify: `tests_dashboard/test_app_lifecycle.py` (or the established lifecycle test module if one exists).

**Interfaces:**
- Consumes: module-level `DATABASE`, `VIEWER_SERVICE`, and `STARTUP_WARNINGS`.
- Produces: `lifespan(app: FastAPI)` as the FastAPI lifespan context manager passed to `FastAPI(..., lifespan=lifespan)`.

- [x] Add a failing test that enters the app lifespan, asserts `DATABASE.initialize()` runs and the unavailable-database message is appended once, and asserts `VIEWER_SERVICE.stop_all()` plus `DATABASE.dispose()` run on exit.
- [x] Add a failing startup-error test proving resource cleanup still runs if initialization raises after partially opening resources; keep behavior aligned with the current initialize/dispose contract.
- [x] Replace both `@app.on_event` handlers with an `@asynccontextmanager` lifespan function. Preserve the existing database fallback warning de-duplication and cleanup order.
- [x] Run `uv run --extra cpu --extra dashboard --group dev python -m pytest tests_dashboard -q -W error::DeprecationWarning`; confirm lifecycle, API, streaming, and TestClient behavior pass with no FastAPI event-hook warnings.

### Task 3: Move Starlette TestClient to its supported HTTP client

**Files:**
- Modify: `pyproject.toml` development/test dependency declaration.
- Modify: `uv.lock`.
- Create: `tests_dashboard/test_testclient_compat.py` to exercise TestClient import and a real request.

**Interfaces:**
- Consumes: current Starlette 1.6.0 TestClient import failure from missing `httpx2` and the prior warning report.
- Produces: one centrally configured, supported TestClient backend with a lockfile-pinned compatible dependency.

- [x] Add `tests_dashboard/test_testclient_compat.py` with a test that imports `fastapi.testclient.TestClient` inside the test, calls a simple FastAPI route, and asserts HTTP 200. Run it before dependency changes and verify it fails in the test body because Starlette requires the missing `httpx2` package.
- [x] Confirm the installed Starlette 1.6.0 TestClient API uses `httpx2`; add a compatible `httpx2` development dependency to `pyproject.toml` and regenerate `uv.lock` without changing unrelated package selections.
- [x] Keep the existing `fastapi.testclient.TestClient` imports and add only the missing test dependency; the current tests import TestClient directly and have no shared fixture. Preserve request, response, streaming, and lifespan behavior.
- [x] Run `uv run --extra cpu --extra dashboard --group dev python -m pytest tests_dashboard/test_testclient_compat.py -q -W error::DeprecationWarning`; confirm there is no TestClient warning and the request still returns HTTP 200.

### Task 4: Resolve mjlab Warp and PyTorch deprecations at their source

**Files:**
- Modify: pyproject.toml and uv.lock to cap CPU and CUDA Torch constraints below 2.10; the lock resolves both to Torch 2.9.1.
- Add: src/mjlab/__init__.py as a minimal compatibility overlay for the pinned upstream mjlab==1.6.0, only if upstream still assigns the deprecated wp.config.quiet; preserve upstream initialization, entry-point loading, and the installed asset root.
- Add: tests_mjlab/test_dependency_warnings.py with subprocess imports under warning-as-error settings.

**Interfaces:**
- Consumes: the full traces showing Torch 2.14's @torch.jit.script FutureWarning in mjlab.utils.lab_api.math and Warp's wp.config.quiet DeprecationWarning in mjlab.__init__.
- Produces: a repeatable CPU/CUDA dependency resolution and an upstream-compatible mjlab initialization path with no warning filters.

- [x] Add warning-as-error subprocess tests for import mjlab (DeprecationWarning) and import mjlab.utils.lab_api.math (all warnings as errors); run them before the fix and confirm they fail on the respective dependency warnings.
- [x] Check upstream mjlab release/source and Warp's documented replacement. As of 2026-10-09, upstream mjlab==1.6.0 still uses wp.config.quiet = quiet; Warp documents wp.config.log_level = wp.LOG_WARNING as the replacement and schedules removal of quiet in 1.18. Record the upstream references in the compatibility code.
- [x] Replace the deprecated Warp setting at the dependency boundary with the documented public log_level setting, without filtering warnings or changing the default verbosity.
- [x] Torch 2.11.0 still emits a DeprecationWarning from TorchScript decorators when the project task entry point loads; PyTorch 2.9.0 source does not emit that runtime warning. Cap CPU/CUDA optional constraints at <2.10, resolve uv.lock, and confirm both resolve to the validated 2.9 line. Do not replace TorchScript decorators with torch.compile unless CPU/CUDA behavior and numerical tolerances are demonstrated equivalent.
- [x] Preserve MJLAB_SRC_PATH as the installed dependency asset root and preserve mjlab entry-point task registration; verify the standard asset files resolve and the project's registered tasks load.
- [x] Run focused simulator tests and a CPU simulation smoke. Run the existing CUDA smoke on the project GPU environment if available; record when hardware prevents that check.
- [x] Run the focused warning tests with -W error::DeprecationWarning and -W error::FutureWarning; confirm both dependency warnings are gone without suppression.

### Task 5: Enforce a clean warning baseline

**Files:**
- Modify: `.github/workflows/dashboard-ci.yml`.
- Modify: `pyproject.toml` only if adding a narrowly scoped pytest warning policy is needed after the sources are fixed.

**Interfaces:**
- Consumes: all source and dependency fixes from Tasks 1–4.
- Produces: a reproducible warning-as-error verification gate for the complete test suite.

- [ ] Run `uv lock --check` and `uv run --extra dashboard --extra cpu --group dev pytest -q -W error`; expect all tests in the reconciled baseline to pass with zero warnings.
- [ ] Run `uv run --extra dashboard --extra cpu --group dev ruff check .` and the dashboard migration/configuration test command from Task 1.
- [ ] Run the existing simulator CPU smoke and GPU smoke where CUDA is available; record the selected dependency versions and test environment in the change summary.
- [ ] Add `-W error` to the backend and dashboard-backend pytest commands in `.github/workflows/dashboard-ci.yml` so CI keeps warning regressions from returning.
- [ ] Review the final warning summary and verify no broad warning filter, ignored warning category, or untracked environment-only patch was used.

## Completion Criteria

- The full suite reports zero warnings and passes with warnings promoted to errors.
- FastAPI lifespan behavior, dashboard API behavior, Alembic migrations, and simulator CPU/CUDA behavior remain covered and pass.
- `pyproject.toml` and `uv.lock` agree, and any third-party compatibility change is reproducible from a clean environment.


**Canonical WSL baseline (Task 0, after Task 3):** At 6f27b7f plus the TestClient dependency fix, the full suite reports 529 passed, 5 skipped, and 147 warnings in 60.47 seconds. The visible warning groups are 140 FastAPI lifecycle emissions, 6 Alembic path-separator emissions, and 1 Torch JIT FutureWarning. The invalid-escape warning was confirmed during the initial uncached collection, then hidden by Python bytecode cache on the later full run; it remains a source fix. The earlier 183 count and a separate TestClient warning do not reproduce in this canonical environment. Task 3 resolved a hard collection error, not merely a warning.


**Task 2 verification note:** The dashboard suite passes with deprecations treated as errors after Task 4 (187 passed); the earlier strict collection error was the independent Warp deprecation that Task 4 fixed.


**Task 4 result:** The warning tests first failed for the upstream Warp setter and TorchScript decorators. Upstream mjlab 1.6.0 still uses wp.config.quiet; Warp's public replacement is config.log_level. Torch 2.11.0 emits a DeprecationWarning at @torch.jit.script, while the 2.9.1 CPU and cu128 builds pass the warning-as-error import tests. The lock therefore caps CPU/CUDA Torch at <2.10, and the local mjlab initializer overlay is pinned to mjlab==1.6.0, keeps the installed asset root and entry-point load, and can be removed when upstream ships the equivalent Warp fix.
