#!/usr/bin/env python3
"""Deadline-bounded isolated PPO run. No automatic checkpoint promotion."""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/root/ascento_v7_overnight")
HOME = ROOT / ".maintenance/overnight_v7_20261015"
SOURCE = Path("/root/ascento_v6_overnight/.maintenance/overnight_v6_20261014/source/model_20400.pt")
EXPECTED_SOURCE_SHA = "e17ece617c41ea35ea46dee18a4038abfa7abc6a8245ee7a623a194256b96ae2"
PYTHON = "/root/.cache/ascento-mjlab/cu128/bin/python3"
TASK = "Ascento-Generalist-Locomotion-Goal-Settle-V7-Flat"
V5 = "Ascento-Generalist-Locomotion-Goal-Settle-V5-Flat"
OUTPUT = ROOT / "logs/rsl_rl/overnight_v7_20261015"
SUITE = Path("/root/Ascento_MuJoCo_Playground/.maintenance/overnight_v5_20261013/goal_settle_v5_overnight_20261014.toml")
ZONE = ZoneInfo("Europe/Copenhagen")
DEADLINE = dt.datetime(2026, 10, 16, 12, 0, tzinfo=ZONE)
TRAIN_CUTOFF = DEADLINE - dt.timedelta(hours=2, minutes=30)
HOME.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
STOP = False

def stamp() -> str:
    return dt.datetime.now(ZONE).isoformat(timespec="seconds")

def emit(msg: str) -> None:
    line = f"[{stamp()}] {msg}"
    print(line, flush=True)
    with (HOME / "supervisor.log").open("a") as file:
        file.write(line + "\n")

def status(**kwargs) -> None:
    path = HOME / "status.json"
    try:
        obj = json.loads(path.read_text())
    except (ValueError, OSError):
        obj = {}
    obj.update(kwargs)
    obj["updated_at"] = stamp()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def environment(initial: bool) -> dict:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "2",
        "ASCENTO_GENERALIST_OPTIMIZER_PROFILE": "conservative",
        "ASCENTO_GENERALIST_NORMALIZER_PROFILE": "adaptive",
        "ASCENTO_GENERALIST_OBSTACLE_MODE": "0",
        "ASCENTO_GENERALIST_PRECISION_ANCHOR_FRACTION": "0.15",
        "ASCENTO_GENERALIST_RECOVERY_RETARGET_ANCHOR_FRACTION": "0.05",
        "ASCENTO_GENERALIST_QUARTER_TURN_HEADING_FRACTION": "0.05",
        "ASCENTO_LIVE_MONITOR_ENVS": "0,1",
    })
    env.pop("ASCENTO_V5_INITIAL_ACTOR_CHECKPOINT", None)
    if initial:
        env["ASCENTO_V5_INITIAL_ACTOR_CHECKPOINT"] = str(SOURCE)
    return env

def gpu_busy() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True)
    jobs = []
    for line in result.stdout.splitlines():
        m = re.match(r"\s*(\d+)\s+(.*)", line)
        if not m or int(m[1]) == os.getpid():
            continue
        command = m[2]
        if "python" in command and (
            "mjlab.scripts.train" in command
            or "ascento_mjlab.cli evaluate run" in command
            or "ascento_mjlab.tools.capture_benchmark" in command
        ):
            jobs.append(line.strip()[:250])
    return jobs

def gpu_probe() -> bool:
    code = ("import torch; "
            "assert torch.cuda.is_available(); "
            "v=torch.ones(128,device='cuda:0'); "
            "assert (v*2).sum().item()==256")
    try:
        proc = subprocess.run([PYTHON, "-c", code], cwd=ROOT, env=environment(False),
                              capture_output=True, text=True, timeout=100)
        if proc.returncode:
            emit("CUDA preflight failed: " + (proc.stderr[-700:] or proc.stdout[-700:]))
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError) as exc:
        emit("CUDA preflight failure " + repr(exc))
        return False

def checkpoints() -> list[Path]:
    return sorted(OUTPUT.rglob("model_*.pt"),
                  key=lambda p: (p.stat().st_mtime_ns, p.name))

def progress(log: Path) -> dict:
    try:
        with log.open("rb") as file:
            file.seek(max(0, log.stat().st_size - 2_000_000))
            data = file.read().decode(errors="replace")
    except OSError:
        return {}
    its = re.findall(r"Learning iteration\s+(\d+)/\d+", data)
    rewards = re.findall(r"Mean reward:\s*([-+\d.]+)", data)
    return {
        "last_iteration": int(its[-1]) if its else None,
        "last_mean_reward": float(rewards[-1]) if rewards else None,
    }

def launch(command: list[str], env: dict, log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as file:
        file.write(f"\n===== {stamp()} =====\n")
    stream = log.open("ab", buffering=0)
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        stream.close()
    return process

def stop_process(proc: subprocess.Popen, reason: str) -> None:
    if proc.poll() is not None:
        return
    emit(reason + f"; asking PID {proc.pid} to stop")
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=160)
    except subprocess.TimeoutExpired:
        emit("Trainer did not stop after SIGINT; SIGTERM process group")
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=65)
        except subprocess.TimeoutExpired:
            emit("Trainer ignored SIGTERM; SIGKILL process group")
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=20)

def run_bounded(command: list[str], logfile: Path, seconds: float, env: dict) -> int:
    proc = launch(command, env, logfile)
    status(evaluation_pid=proc.pid)
    while proc.poll() is None and dt.datetime.now(ZONE) < DEADLINE:
        if time.monotonic() - start_monotonic > seconds:
            stop_process(proc, "Evaluation time limit reached")
            return 124
        time.sleep(10)
    if proc.poll() is None:
        stop_process(proc, "Noon deadline reached")
        return 124
    return int(proc.returncode)

def main() -> None:
    lock = (HOME / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert SOURCE.is_file() and sha256(SOURCE) == EXPECTED_SOURCE_SHA, "V5 source has changed"
    assert SUITE.is_file(), "Matched evaluation suite missing"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    now = dt.datetime.now(ZONE)
    if now >= TRAIN_CUTOFF:
        raise RuntimeError("Training cutoff already elapsed")
    status(phase="preflight", deadline_local=DEADLINE.isoformat(),
           training_cutoff_local=TRAIN_CUTOFF.isoformat(),
           source_checkpoint=str(SOURCE), source_sha256=EXPECTED_SOURCE_SHA,
           worktree_commit=commit, no_auto_promotion=True,
           started_at=stamp())
    emit(f"V7 isolated training prepared; train cutoff {TRAIN_CUTOFF.isoformat()}, final deadline {DEADLINE.isoformat()}")
    while gpu_busy() and dt.datetime.now(ZONE) < now + dt.timedelta(hours=1):
        status(phase="waiting_for_gpu", conflicting_gpu_jobs=gpu_busy())
        time.sleep(60)
    if gpu_busy():
        status(phase="blocked_gpu_busy", conflicting_gpu_jobs=gpu_busy())
        return
    if not gpu_probe():
        status(phase="blocked_cuda_preflight")
        return
    command_base = [
        PYTHON, "-u", "-m", "mjlab.scripts.train", TASK,
        "--agent.save-interval", "200",
        "--env.scene.num-envs", "512",
        "--agent.max-iterations", "50000",
        "--agent.seed", "20261015",
        "--gpu-ids", "[0]",
        "--log-root", str(OUTPUT),
    ]
    status(phase="smoke_test")
    smoke = [
        PYTHON, "-u", "-m", "mjlab.scripts.train", TASK,
        "--agent.save-interval", "10",
        "--env.scene.num-envs", "64",
        "--agent.max-iterations", "20",
        "--agent.seed", "20261015",
        "--gpu-ids", "[0]",
        "--log-root", str(ROOT / "logs/rsl_rl/v7_preflight_smoke"),
    ]
    emit("Running short real CUDA/MuJoCo PPO smoke test")
    with (HOME / "smoke.log").open("wb") as f:
        try:
            check = subprocess.run(smoke, cwd=ROOT, env=environment(True),
                                   stdin=subprocess.DEVNULL, stdout=f,
                                   stderr=subprocess.STDOUT, timeout=900)
            smoke_ok = check.returncode == 0
        except subprocess.TimeoutExpired:
            smoke_ok = False
    if not smoke_ok:
        emit("Real PPO smoke failed; refusing long training")
        status(phase="blocked_smoke_failed")
        return
    emit("Smoke succeeded; starting full 512-environment PPO")
    status(phase="training", smoke_passed=True, training_attempt=0,
           training_command=command_base)
    attempts = 0
    log = HOME / "trainer.log"
    while dt.datetime.now(ZONE) < TRAIN_CUTOFF:
        existing = checkpoints()
        resume = bool(existing)
        command = list(command_base)
        env = environment(not resume)
        if resume:
            recent = existing[-1]
            command += ["--agent.resume", "--agent.load-run", recent.parent.name,
                        "--agent.load-checkpoint", recent.name]
            emit(f"Resuming V7 optimizer from saved checkpoint {recent}")
        elif attempts:
            emit("No recovery checkpoint available; not restarting from scratch")
            status(phase="failed_no_checkpoint")
            return
        attempts += 1
        proc = launch(command, env, log)
        status(phase="training", trainer_pid=proc.pid,
               training_attempt=attempts, resumed=resume)
        while proc.poll() is None and dt.datetime.now(ZONE) < TRAIN_CUTOFF:
            status(phase="training", trainer_pid=proc.pid,
                   training_attempt=attempts, checkpoint_count=len(checkpoints()),
                   **progress(log))
            time.sleep(50)
        if proc.poll() is None:
            stop_process(proc, "Training cutoff reached; reserve time for evaluation")
            break
        emit(f"Trainer exited with code {proc.returncode}")
        status(trainer_exit_code=proc.returncode, checkpoint_count=len(checkpoints()),
               **progress(log))
        if proc.returncode == 0:
            break
        if attempts >= 3 or not checkpoints():
            emit("No further automatic retries; preserving artifacts")
            status(phase="training_failed")
            return
        if (TRAIN_CUTOFF - dt.datetime.now(ZONE)).total_seconds() < 900:
            break
        emit("Unexpected trainer exit; waiting before bounded checkpoint recovery")
        time.sleep(180)
        if not gpu_probe():
            status(phase="blocked_cuda_recovery")
            return
    available = checkpoints()
    if not available:
        status(phase="failed_no_checkpoint")
        return
    selected = available[-1]
    status(phase="training_complete", selected_checkpoint=str(selected),
           selected_checkpoint_sha256=sha256(selected),
           checkpoint_count=len(available), **progress(log))
    emit(f"Selected latest saved V7 checkpoint for evaluation: {selected.name}")
    if dt.datetime.now(ZONE) >= DEADLINE - dt.timedelta(minutes=25):
        status(phase="evaluation_deferred_deadline")
        return
    if not gpu_probe() or gpu_busy():
        status(phase="evaluation_deferred_gpu")
        return
    converted = HOME / "candidate_v7_as_v5.pt"
    transfer = [PYTHON, "-m", "ascento_mjlab.tools.initialize_transfer",
                "--source", str(selected), "--task", V5,
                "--output", str(converted), "--device", "cuda:0"]
    try:
        with (HOME / "transfer_to_v5.log").open("wb") as file:
            ret = subprocess.run(transfer, cwd=ROOT, env=environment(False),
                                 stdout=file, stderr=subprocess.STDOUT,
                                 timeout=900).returncode
    except subprocess.TimeoutExpired:
        ret = 124
    if ret or not converted.is_file():
        emit(f"Transfer to V5 evaluator failed ({ret})")
        status(phase="evaluation_transfer_failed", transfer_returncode=ret)
        return
    status(phase="evaluating", converted_checkpoint=str(converted))
    emit("Running matched V5 benchmark on V7 actor, without changing benchmark thresholds")
    evaluate = [
        PYTHON, "-m", "ascento_mjlab.cli", "evaluate", "run",
        "--checkpoint", str(converted), "--suite", str(SUITE),
        "--batch-size", "128", "--device", "cuda:0",
        "--output-root", str(HOME / "evaluations"), "--json",
    ]
    # Evaluate until 12:00, or a bounded two-hour wall budget.
    global start_monotonic
    start_monotonic = time.monotonic()
    ret = run_bounded(evaluate, HOME / "candidate_evaluation.log",
                      max(0.0, min(7200.0, (DEADLINE - dt.datetime.now(ZONE)).total_seconds() - 30)),
                      environment(False))
    status(phase="finished" if ret != 124 else "evaluation_timeout",
           evaluation_returncode=ret, finished_at=stamp(),
           no_auto_promotion=True)
    emit(f"Finished V7 run: evaluation returncode={ret}; no automatic promotion")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("FATAL " + repr(exc))
        status(phase="failed", error=repr(exc))
        raise
