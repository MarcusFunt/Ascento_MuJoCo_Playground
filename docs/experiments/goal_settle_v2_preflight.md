# GOAL-SETTLE-v2 experiment preflight

**Status:** experimental implementation on branch `feat/goal-settle-v2`; not a promoted policy.

## Rationale and frozen baseline

The 5,200-update October 12 four-hour PPO run improved episode reward but finished with 0/320 settled targets on `generalist_reward_alignment_v1`, compared with 192/320 for the selected transfer. In the short cohorts the trained robot typically settled at 11?12 cm target error with near-zero speed, just outside the immutable 10 cm evaluation envelope. The preserved source checkpoint is:

`transfers/ascento_generalist_locomotion_flat/reward_alignment_reward_fix/model_000000.pt`

Its actor state was verified tensor-for-tensor identical to the previously selected generalist transfer. Never resume optimizer/critic state from the failed `model_5199.pt`.

## Treatment and causal constraints

- **Control task:** `Ascento-Generalist-Locomotion-Flat` with existing rewards.
- **Treatment task:** `Ascento-Generalist-Locomotion-Goal-Settle-V2-Flat`.
- Preserve the 42-dimensional actor observations, six action outputs, 65 Nm plant, 20%/20%/60% fixed cohorts, stage-0 goal sampling, seed, batch size and PPO parameters.
- Treatment replaces near-goal progress attenuation with a 0.80 floor while stationary motion still earns zero signed progress; reduces final-yaw and settled-balance shaping away from the target; delays planar-speed braking penalty until inside approximately 14 cm; and pays an 8-point **one-shot, 0.5-second dwell** bonus inside 10 cm / 0.20 rad.
- The training event's target-arrival and retarget thresholds are changed **only for the v2 task** to 10 cm / 0.20 rad, with 0.5 s dwell. A shared metrics bug was corrected so it reads its configured distance threshold rather than the old hardcoded 3.5 cm value; v1 keeps the same 3.5 cm configuration.
- This treatment is a **bundle**: initial experiment determines whether the combined intervention can retain/improve behavior, *not* which individual reward change caused any improvement. If positive, follow with focused single-variable ablations.

## Preflight tests

`PYTHONPATH=. .venv/bin/python -m pytest -q tests_mjlab/test_goal_settle_v2.py tests_mjlab/test_reward_contract.py tests_mjlab/test_locomotion_curriculum.py tests_mjlab/test_locomotion_targets.py tests_mjlab/test_task_config.py`

Must establish task topology versioning, signed target progress at 11.5 cm, braking within the envelope, reduced stationary reward far away, one-time settled arrival, and complete-return ranking of scripted traces. Those *scripted* traces are reward unit tests, not claims that the traces are physically reachable. In addition, real mjlab CPU simulation must reconcile every returned step reward with its weighted per-term rates.

The existing actor-only transfer command is the safe way to produce the treatment initializer:

```bash
ASCENTO_GENERALIST_FIXED_GOAL_MIX_STAGE=0 PYTHONPATH=. \
  .venv/bin/python -m ascento_mjlab.tools.initialize_transfer \
  --source transfers/ascento_generalist_locomotion_flat/reward_alignment_reward_fix/model_000000.pt \
  --task Ascento-Generalist-Locomotion-Goal-Settle-V2-Flat \
  --output transfers/ascento_generalist_goal_settle_v2_flat/goal_settle_v2_stage0/model_000000.pt \
  --device cpu
```

Verify all 13 actor tensors are equal, optimizer state empty, and new task contract differs. Keep both source and treatment transfer immutable.

**Critical managed-launch detail:** The environment overrides used to construct the transfer must also be set when launching the trainer, because the checkpoint's signed event topology includes `ASCENTO_GENERALIST_FIXED_GOAL_MIX_STAGE`. The managed `--parent-checkpoint` argument is only lineage metadata; *it does not enable model loading on its own*. After merging to a clean synchronized `main`, explicitly pass trainer resume arguments **after** the `--` separator:

```bash
export ASCENTO_GENERALIST_FIXED_GOAL_MIX_STAGE=0
export ASCENTO_GENERALIST_OBSTACLE_MODE=0
scripts/ascento-gpu ascento run start \
  --task Ascento-Generalist-Locomotion-Goal-Settle-V2-Flat \
  --display-name goal-settle-v2-pilot --purpose goal-settle-v2 \
  --parent-checkpoint transfers/ascento_generalist_goal_settle_v2_flat/goal_settle_v2_stage0/model_000000.pt \
  --envs 512 --iterations 50 --seed 20261012 -- \
  --agent.resume True --agent.load-run _resume_parent \
  --agent.load-checkpoint model_000000.pt --agent.save-interval 50
```

A ten-update transfer-initialized smoke with this exact contract completed on Oct 12 with exit code 0: managed run `20261012_200832_goal-settle-v2-stage0-transfer-10-updates_cea7cdba`, checkpoints 0/5/9, a populated optimizer, and a validated parent link. The older smoke without the resume arguments is **invalid for behavioral comparison**, and a subsequent incompatible parent attempt was rejected before training. Do not silently count either as treatment evidence.

An initial four-scenario physical evaluation of model_9 exceeded a 180-second command timeout; it left only an incomplete scenario-materialization artifact, so it does **not** establish success or failure. No longer pilot should be promoted before a completed physical evaluation.

## Evaluation compatibility (IMPORTANT)

Existing generalist suite TOMLs target `Ascento-Generalist-Locomotion-Flat`. Do **not** silently edit their immutable task IDs, bypass task-contract checks or compare different evaluation rewards.

For v2 checkpoints, create an *evaluation-only actor transfer back into the original generalist task*, copying the actor and its normalizer unchanged and resetting the critic/optimizer. This puts the candidate in the exact same physical evaluator and scenario IDs as the archived selected policy:

```bash
PYTHONPATH=. .venv/bin/python -m ascento_mjlab.tools.initialize_transfer \
  --source <new-v2-checkpoint.pt> --task Ascento-Generalist-Locomotion-Flat \
  --output <new-immutable-evaluation-transfer.pt> --device cpu

scripts/ascento-gpu ascento evaluate run \
  --checkpoint <new-immutable-evaluation-transfer.pt> \
  --suite roadrunner_generalist_sequence_gate_v1 --batch-size 256 --device cuda:0
```

Record source actor SHA-256, evaluation-transfer SHA-256 and lineage so no evaluator runs an unrelated policy. For the original selected actor, retain the pre-existing reference evaluation. Compare identical suite/scenario hashes and physical metrics only. Only open the held-out promotion suite if a candidate passes the development gate.

## Bounded training plan (requires merge before GPU launch)

The canonical `scripts/ascento-gpu` guard requires a clean checkout with `main == origin/main`. It must **not** be bypassed to train this feature branch. Merge the isolated, tested PR through review/CI before attempting GPU training. Verify the source checkpoint and run provenance before each launch.

- Transfer both arms from the same frozen actor; no incompatible PPO optimizer resume.
- Use 512 environments, 24 rollout steps, seed `20261012`, fixed stage 0, regular optimizer. Do not simultaneously modify PPO learning rate.
- Two-arm pilot: 0?300 updates maximum per arm, with checkpoint screens at approximately 0, 50, 100, 150, 200 and 299, sequential GPU use, matched scenario hashes, and early stopping for retained-skill regressions.
- Initial screen: `roadrunner_driver_progression_v1` and `roadrunner_generalist_sequence_gate_v1`; expand with `generalist_reward_alignment_v1` and morphology after a physically valid candidate. A merely diagnostic `PASS` on the 320-case suite is not sufficient.
- Retention floors: the selected transfer's short-target stop reliability; all nine hard development gates; practical p95 final position <= 0.05 m; survival/recovery and action quality; no regression on 0.5 m straight travel. Require a positive **paired** physical improvement before longer training.

No GPU training or model promotion is implied by implementing these files.
