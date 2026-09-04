# Quantitative evaluation benchmarks

This directory contains immutable, versioned evaluation suites for trained
Ascento policies. Training rewards are diagnostics only; benchmark acceptance is
defined by physical/task metrics and explicit gates.

## Rules

1. A published suite is immutable. If a reset range, threshold, metric
   definition, event timing, or scenario generator changes, create a new suite
   version.
2. Official evaluation uses deterministic policy inference unless the suite
   explicitly declares stochastic evaluation.
3. Every generated scenario has an order-independent seed derived from the suite
   ID, root seed, family ID, and scenario index.
4. Evaluation is headless and vectorized. Full trajectories/videos are not
   recorded during the bulk pass.
5. Missing required capabilities produce `INCOMPLETE`; telemetry contradictions
   produce `INVALID`.
6. A checkpoint only advances when every hard gate passes.

Result states:

- `PASS`: evaluator was valid and every hard gate passed.
- `FAIL`: evaluator was valid but policy performance missed one or more gates.
- `INCOMPLETE`: the current task/evaluator cannot measure a required quantity.
- `INVALID`: the evaluation itself is inconsistent or corrupt.

## Current suites

- `balance_dev_v1`: smaller, faster development screen.
- `balance_gate_v1`: authoritative Gate D benchmark with nominal resets,
  expanded resets, deterministic corners, physical force disturbances, and
  60-second endurance.
- `velocity_gate_v1`: deterministic command-timeline tracking benchmark.
- `recovery_gate_v1`: wide-reset recovery benchmark using the canonical
  `RecoveryEnvelope` fields plus continuous stable duration.
- `jump_gate_v1`: intentionally fails closed on current main until
  heading-relative landing distance, true pre-impact landing metrics, and
  limiting-wheel clearance are implemented.

## Run

```bash
ascento-evaluate \
  --checkpoint logs/rsl_rl/.../model_1999.pt \
  --suite balance_gate_v1 \
  --batch-size 512 \
  --device cuda:0
```

Screen checkpoints:

```bash
ascento-evaluate-checkpoints 'logs/rsl_rl/.../model_*.pt' \
  --suite balance_dev_v1 --top 3
```

Compare two completed evaluations using paired scenario IDs:

```bash
ascento-compare-evaluations evaluations/base evaluations/candidate
```

Replay an exact stored scenario:

```bash
ascento-replay-eval evaluations/<evaluation-id> \
  --scenario balance_gate_v1/disturbance/000421 \
  --viewer native
```

## Artifacts

Each evaluation creates:

```text
evaluations/<evaluation-id>/
├── manifest.json
├── suite.json
├── resolved_scenarios.jsonl
├── results.sqlite
├── summary.json
├── consistency.json
├── gate.json
├── failures.json
└── report.html
```

`results.sqlite` stores one row per scenario outcome plus long-form metrics.
`resolved_scenarios.jsonl` is the exact materialized benchmark input and is
hashed into the manifest.

## Balance Gate D v1

Families:

- 1024 nominal 20 s episodes using the training reset envelope.
- 1024 expanded 20 s episodes outside the nominal balance reset basin.
- 256 deterministic corner cases with explicit yaw strata.
- 1024 20 s physical push scenarios. Pushes are parameterized by equivalent
  velocity impulse and resolved to a force pulse using the evaluated model mass.
- 256 nominal 60 s endurance episodes.

The disturbance recovery predicate requires, continuously for 0.5 s:

- tilt <= 0.08 rad,
- planar speed <= 0.10 m/s,
- height error <= 0.05 m,
- roll/pitch angular speed <= 0.25 rad/s,
- both wheels supported.

The gate includes Wilson lower confidence bounds for reliability plus P95
motion/control constraints. See the TOML itself for authoritative values.

## Consistency audit

Every completed evaluation checks invariants including:

- episode time versus integer control steps,
- result/scenario count and ID agreement,
- displacement <= path length,
- mean absolute effort <= RMS effort <= peak effort,
- requested physical effort <= configured physical authority,
- fractions remain in [0, 1],
- successful episodes reach their scenario horizon.

A failed invariant changes the evaluation result to `INVALID` regardless of
policy performance.
