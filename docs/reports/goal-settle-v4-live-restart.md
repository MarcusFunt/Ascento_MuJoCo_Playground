# GOAL-SETTLE-v4: critic-safe completion objective and fresh-lineage handoff

## Decision basis (13 October 2026, Copenhagen time)

The matched from-scratch 50-update CPU v2/v3 pilot (seed 20261013, 64 envs, 24 rollout steps, 15% precision and 5% recovery cohorts, fixed stage zero) finished successfully.

- v2: mean episode length 28.50 steps, short target arrival LCB 0.0453, critic loss 0.1283, **zero settled completions**.
- v3: mean episode length 96.84 steps, short target arrival LCB 0.1210, critic loss **856.28**, **zero settled completions**.
- Raw rewards and unnormalized critic losses across these differently scaled objectives are *not directly comparable*, but v3 critic-gradient scale is a serious concern.

**Newer evidence supersedes the early pilot**: the independent fresh-from-random v2 lineage at ~3150 iterations had already reached short-arrival LCB **0.5530** and nonzero one-shot settled-stop reward **0.8667**. This is actual growing skill, not just historical data. Preserve this checkpoint for diagnosis and comparison. **User chose to start v4 randomly initialized**, not to transfer actor, critic, observation normalizer or optimizer weights from v2 or the old ~80k lineage; this trades near-term v2 skills for a cleaner reward+curriculum experiment.

## V4 contract

New task: `Ascento-Generalist-Locomotion-Goal-Settle-V4-Flat`.

V4 inherits all v3 completion-first incentives, reduces the overly large catastrophic critic target, and **unblocks the navigation curriculum**:

- V3 failure termination impulse **?200** ? V4 **?30**, paid once on genuine non-timeout termination. It is still deliberately larger than the upper bound on the discounted outstanding per-attempt time cost under gamma 0.998: `3.0 reward/s * 0.01 s / (1?0.998) ? 15`. This is a heuristic safety comparison, *not* a proof against falling-to-reset exploits.
- Goal incomplete cost stays **?1.7/s rising to ?3.0/s** over 8 seconds and resets only on a new attempt.
- Ongoing passive balancing/heading income ? **+1.4/s**.
- Signed distance-closing progress **14**, proximity-gated braking **?3**.
- Capped settling-quality credit **6 per attempt**, and settled-stop completion **30 once** after 0.5 s.
- **Curriculum:** Navigation goal-distance progression now uses waypoint success independently of push-recovery LCB, while recovery anchor/cohort and diagnostic metrics remain in place. Prior run at ~3300 updates had stage-window short-arrival LCB ~0.97 vs recovery LCB ~0.18, making the old 0.85 recovery promotion threshold a bottleneck. Original v1-v3 event defaults are unchanged. Navigation LCB promotion thresholds and demotion hysteresis remain; recovery does not demote navigation stage.
- Maintains v3 goal thresholds (10 cm, 0.2rad, 0.5s), 42-D actor, 6 actions and physics. Existing v1/v2/v3 task contracts and checkpoints remain untouched.

## Recommended controlled continuation

1. Preserve final source run checkpoint; hash it; do not delete run logs.
2. Launch v4 **from scratch**, without `--agent.resume`, `--parent-checkpoint`, or `--agent.load-run`. Use navigation-heavy fractions (15% precision, 5% recovery, 80% navigation) and no fixed goal-mix stage, so the new navigation gates can advance.
3. Verify the fresh trainer starts without loaded actor weights, has finite losses, and makes navigation progress. Keep the run managed, checkpointed and monitored; use initial checkpoints to judge learning quality.
4. Compare **actual successful attempts, settled stops, falls and curriculum promotion**, rather than v2/v4 mean reward.
5. Evaluate and revert to preserved source if v4 destroys the already acquired v2 ability.

Do not stop or restart a managed run without verifying its authoritative run status. When reloading checkpoints into a different reward task, a direct resume is invalid; use target-signed actor-only transfer.
