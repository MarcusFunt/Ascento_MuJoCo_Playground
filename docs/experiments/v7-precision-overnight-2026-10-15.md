# V7 precision training (Oct 15-16, 2026)

* Isolated worktree: `/root/ascento_v7_overnight`, branch `feat/v7-precision-overnight-20261015`.
* Source actor: protected V5 `model_20400.pt` (SHA256 `e17ece617c41ea35ea46dee18a4038abfa7abc6a8245ee7a623a194256b96ae2`). Never overwrite it.
* Task: `Ascento-Generalist-Locomotion-Goal-Settle-V7-Flat`. Actor and plant interfaces match V5, but the reward topology differs: **actor-only transfer, fresh critic/optimizer**.
* V7 changes over V6: training retarget/completion at 0.09 m rather than 0.08; weighted quality shaping distance radius 0.22 m; balanced rather than seven-way multiplicative quality; per-attempt credit budget remains capped and gated by physical stability. The benchmark thresholds are unchanged.
* Trainer: 512 environments, seed 20261015, conservative PPO (LR 3e-5), snapshots every 200 iterations, upper bound 50,000 iterations. A short 64-env MuJoCo/CUDA smoke test must succeed first.
* Supervisor service: `ascento-v7-overnight-20261015.service`; runner code in `scripts/overnight_v7_runner.py`. On-host runtime script in ignored `.maintenance/overnight_v7_20261015/runner.py`. Logs, status and evaluator output are in that ignored maintenance directory.
* Training cutoff: **Friday Oct 16, 09:30 Europe/Copenhagen**. Reserve until **Friday Oct 16, 12:00 Europe/Copenhagen** for actor transfer + matched 244-scenario V5 evaluation. These are absolute deadlines, independent of clock drift in iteration counts.
* On unexpected failure, a bounded checkpoint resume is permitted (up to 3 training attempts), only if the CUDA probe passes and a V7 checkpoint exists. No restart from scratch after a crash; no automatic V5/V6 replacement.
* Observation/monitoring: live frames in `logs/rsl_rl/overnight_v7_20261015`. Add `docker/compose.deep-live-v7.yaml` on top of the V6 overlay, then replace only dashboard container. The V7 mount is read-only.
* Pre-existing main-branch untracked `benchmarks/suites/goal_settle_v5_comprehensive_v1.toml` is intentionally untouched.

View status: `systemctl status ascento-v7-overnight-20261015.service`, `cat /root/ascento_v7_overnight/.maintenance/overnight_v7_20261015/status.json`.
No change to the NVIDIA driver, Tailscale, database or GPU container workers.
