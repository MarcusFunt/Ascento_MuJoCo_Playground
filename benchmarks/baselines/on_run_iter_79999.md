# Frozen balance baseline: `on_run_iter_79999`

Behavioral reference checkpoint:

```text
logs/rsl_rl/20260912_222540_on-run_1f40b484/ascento_balance/2026-09-12_22-26-00/model_79999.pt
sha256: 10515b0ccb52f37ddc1851b03ba0ab5e03e25fab4ca6b31e616d43eec6a0fc3b
```

This checkpoint is the comparison baseline. `model_best_long_horizon.pt` is a
negative oscillation fixture, not a replacement baseline. Do not overwrite or
resume it into a changed reward/task ABI; use actor-only transfer instead.

## Evidence

- Broad acceptance: `evaluations/balance_79999_baseline_v5/20260914T153815Z_balance_gate_v5_model_79999`.
  The fixed checkpoint is valid; its sole strict-v5 miss is
  `expanded_survival_lcb`, so it remains the behavioral reference rather than
  a final long-horizon release candidate.
- Stationary-quality regression: `evaluations/balance_quality_regression/20260914T155726Z_balance_quality_regression_v1_model_79999` (`PASS`, 256 fixed scenarios).
- Three deterministic captures (seeds 0, 1, 2):
  `captures/balance_79999_baseline/take_000.npz` through `take_002.npz`.

## Stationary-quality reference values

All values are the nominal-family p95 except contact quality, which is p05.

| Metric | Value |
| --- | ---: |
| Tilt RMS (rad) | 0.015079 |
| Planar-speed RMS (m/s) | 0.004754 |
| Heading-error RMS (rad) | 0.006595 |
| Effort RMS | 2.224555 |
| Action-rate RMS | 0.000358 |
| Action second-difference RMS | 0.000201 |
| High-frequency action-power ratio | 0.030768 |
| Nyquist action-power ratio | 0.000004013 |
| Body-rocking RMS (rad/s) | 0.008152 |
| Roll/pitch reversal rate (Hz) | 0.000000 |
| Post-settle angular-velocity RMS (rad/s) | 0.008152 |
| Two-wheel-support fraction | 1.000000 |
| Net displacement (m) | 0.024489 |
| Settling time (s) | 1.122500 |

Use the versioned quality suite for an automatic hard-gate rejection and
paired comparison. A continuation may be promoted only when it does not
produce a `WORSE` baseline verdict and its paired deltas provide positive
evidence beyond survival alone.
