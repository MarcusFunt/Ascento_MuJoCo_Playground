# V6 terminal precision overnight pilot (2026-10-14)

## Motivation

Complete matched 836-scenario V5 evaluation, restarted checkpoint 20400,
SHA-256 e17ece617c41ea35ea46dee18a4038abfa7abc6a8245ee7a623a194256b96ae2:
46/56 gates, 100% disturbance recovery, but stopping at 0.10 m boundary.
Recovery/sequence usually finished at 0.1000.107 m, and long navigation
at 0.210.27 rad heading, exceeding 0.20 rad.

## Hypothesis and bounded change

Training stop/bonus at 0.10 m permits edge parking. V6 preserves V5's actor
and plant ABI, all locomotion/recovery rewards and curriculum structure, but:
- Tightens training retarget and one-shot completion radius from 0.10 to 0.08 m,
  keeping them identical and preserving event/bonus alignment.
- Adds attempt-capped GeneralistTargetSettledQuality weight 14, max credit
  one second, with 0.18 m shaping radius and 0.16 rad tilt tolerance.
- Preserves 0.50 s training stop hold, 0.16 rad V5 tilt and recovery mix.
- Does not loosen or modify independent benchmark thresholds.

Different reward ABI: use actor-only transfer, never resume V5 critic or optimizer.

## Overnight protocol

1. Preflight verified compatible source V5 20400 -> V6 actor-only transfer.
2. Let V5 trainer and existing guarded 244-scenario evaluations finish first.
3. Start independent V6 conservative PPO (actor/normalizer transferred,
   new critic/optimizer/iteration, learning rate 3e-5, desired KL 0.003).
4. Save checkpoints every 100 updates, monitor loss/reward/curriculum, and
   stop around the requested 12-hour wall clock end.
5. Transfer a V6 candidate actor back to V5 ABI, run matched 244-case
   independent evaluation after training, no automatic promotion.
6. Fail closed on missing source, incompatible actor ABI, or setup failure.

## Promotion gates

Evaluate 0.035 m arrival, 0.10 m strict FINAL stop, heading, recovery, and
tilt. Source V5 20400 is protected; choose candidates only on independent
matched evidence, not mean PPO reward or raw stop success alone.
