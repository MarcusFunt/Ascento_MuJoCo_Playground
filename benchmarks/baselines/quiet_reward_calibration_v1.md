# Quiet-balance reward calibration v1

This calibration uses the fixed 256-scenario stationary-quality measurements,
not PPO return. The proposed penalties are evaluated only through
`settled_balance`, so these p95 RMS values are conservative proxies: the
actual per-step term is additionally multiplied by a score in `[0, 1]` and
vanishes during a genuine recovery.

| Quantity (nominal p95) | 79,999 reference | Long-horizon oscillation fixture |
| --- | ---: | ---: |
| Action second-difference RMS | 0.00020127 | 0.12370267 |
| High-frequency action-power ratio | 0.030768 | 0.512328 |
| Nyquist action-power ratio | 0.00000401 | 0.090712 |
| Roll/pitch reversal rate (Hz) | 0.000000 | 3.817682 |
| Body-rocking RMS (rad/s) | 0.00815200 | 0.00701678 |

Source artifacts:

- reference: `evaluations/balance_quality_regression/20260914T155726Z_balance_quality_regression_v1_model_79999`;
- oscillation fixture: `evaluations/balance_quality_regression/20260914T160041Z_balance_quality_regression_v1_model_best_long_horizon`.

The selected `Ascento-Balance-Quiet-Flat` weights are `-0.50` for the settled
action-second-difference energy and `-0.10` for settled roll/pitch angular
energy. Squaring the p95 second-difference proxy gives an unweighted separation
of about 377,500×: before the state gate and timestep reward scaling, the
weighted action penalty is approximately `-2.0e-8` for 79,999 and `-7.65e-3`
for the oscillation fixture. It is therefore negligible for the reference but
substantial for the A-B-A-B behavior.

The fixture's body-rocking RMS is not elevated, so the physical term was not
tuned to reject that fixture on its own. It protects the separate failure mode
where a controller makes the chassis rock while actions remain superficially
smooth; the regression gate measures both channels independently.
