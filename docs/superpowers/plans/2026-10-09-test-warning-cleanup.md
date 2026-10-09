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

- [ ] Add a failing test that enters the app lifespan, asserts `DATABASE.initialize()` runs and the unavailable-database message is appended once, and asserts `VIEWER_SERVICE.stop_all()` plus `DATABASE.dispose()` run on exit.
- [ ] Add a failing startup-error test proving resource cleanup still runs if initialization raises after partially opening resources; keep behavior aligned with the current initialize/dispose contract.
- [ ] Replace both `@app.on_event` handlers with an `@asynccontextmanager` lifespan function. Preserve the existing database fallback warning de-duplication and cleanup order.
- [ ] Run `uv run --extra cpu --extra dashboard --group dev python -m pytest tests_dashboard -q -W error::DeprecationWarning`; confirm lifecycle, API, streaming, and TestClient behavior pass with no FastAPI event-hook warnings.

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

### Task 4: Remove the mjlab/Torch JIT deprecation at its source

**Files:**
- Modify: `pyproject.toml` and `uv.lock` only if selecting a compatible upstream `mjlab` or PyTorch version.
- If upstream has no compatible fix: add the smallest maintained compatibility patch at the dependency boundary and document its upstream issue/reference; do not patch the installed environment manually.
- Add or modify: focused dependency compatibility tests in the established simulator test suite.

**Interfaces:**
- Consumes: the complete warning stack trace for `torch.jit.script` FutureWarnings and current CPU/CUDA dependency constraints.
- Produces: a dependency source that no longer emits the warning while preserving the simulator API and numerical behavior.

- [ ] Capture the full warning stack trace and identify the exact deprecated JIT call, its dependency owner, and all affected code paths.
- [ ] Check for the first upstream `mjlab` release that removes or safely replaces that call and supports the repository’s Python, Torch, and MuJoCo Warp matrix. If available, update the exact project constraint and lockfile to that release.
- [ ] If no compatible upstream release exists, prepare a minimal tracked patch or fork reference that replaces only the deprecated call while preserving the function signature and behavior; document why a Torch version pin alone is insufficient or sufficient based on the trace.
- [ ] Run focused simulator tests on CPU and the existing CUDA smoke test on the project GPU environment. Compare deterministic outputs within the existing tolerances before accepting the dependency change.
- [ ] Run the focused test with `-W error::FutureWarning` and confirm the JIT warning is gone without a warning filter.

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
