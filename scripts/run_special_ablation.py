"""Run exactly four isolated authorized SFT arms, export, evaluate, then stop."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/special_ablation_v1"
LOGS = ROOT / "logs/special_ablation_v1"
ARMS = [("A_keep_eos", False, False), ("B_reset_eos", True, False),
        ("C_keep_noeos", False, True), ("D_reset_noeos", True, True)]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_state(state):
    state["updated_utc"] = utc_now()
    target = REPORT / "status.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def identities(paths):
    return {str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in paths}


def command(module, *args):
    return [sys.executable, "-X", "utf8", "-m", module, *map(str, args)]


def run_child(state, phase, arm, argv):
    if (REPORT / "STOP_REQUESTED").exists():
        raise RuntimeError("Stop requested before next phase; no further jobs started")
    stem = f"{arm}.{phase}"
    stdout_path, stderr_path = LOGS / f"{stem}.stdout.log", LOGS / f"{stem}.stderr.log"
    with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout, stderr_path.open("x", encoding="utf-8", newline="\n") as stderr:
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=stdout, stderr=stderr,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        started = perf_counter()
        try:
            state.update({"phase": phase, "arm": arm, "child_pid": proc.pid, "command": argv,
                          "phase_elapsed_seconds": 0, "stdout": str(stdout_path), "stderr": str(stderr_path)})
            save_state(state)
            while True:
                if (REPORT / "STOP_REQUESTED").exists():
                    raise RuntimeError("User stop requested during the active phase")
                try:
                    code = proc.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    state["phase_elapsed_seconds"] = perf_counter() - started
                    save_state(state)
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=15)
            state["child_pid"] = None
            raise
        state["child_pid"] = None
        state["phase_elapsed_seconds"] = perf_counter() - started
        state["completed_phases"].append({"phase": phase, "arm": arm, "exit_code": code,
                                          "seconds": state["phase_elapsed_seconds"]})
        save_state(state)
        if code != 0:
            raise RuntimeError(f"{stem} exited {code}; see {stderr_path}")


def main():
    approval = json.loads((REPORT / "launch_approved.json").read_text(encoding="utf-8"))
    if approval.get("approved") is not True or approval.get("arms") != [arm for arm, _, _ in ARMS]:
        raise ValueError("Missing reviewed launch approval")
    if (REPORT / "status.json").exists():
        raise FileExistsError("Refusing to restart or overwrite this completed/partial experiment")
    for arm, _, _ in ARMS:
        for directory in (ROOT / "runs/special_ablation_v1" / arm, ROOT / "artifacts/special_ablation_v1" / arm,
                          REPORT / arm):
            if directory.exists():
                raise FileExistsError(f"Existing output: {directory}")
    if not (REPORT / "frozen_evaluation.json").is_file():
        raise FileNotFoundError("Evaluation must be frozen before training")
    with (REPORT / "launch_started.json").open("x", encoding="utf-8") as marker:
        json.dump({"driver_pid": os.getpid(), "started_utc": utc_now()}, marker)
    protected = [ROOT / "data/stage1_v1/tokenizer.json", ROOT / "runs/pretrain_100m_v1/step_00003052.pt"]
    protected.extend(ROOT / "artifacts" / name / "model.safetensors" for name in
                     ("base_100m_v1", "base_clean_100m_v2", "chat_stage1_v1", "chat_oldbase_v2", "chat_cleanbase_v2"))
    before = identities(protected)
    LOGS.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "driver_pid": os.getpid(), "started_utc": utc_now(),
             "completed_phases": [], "protected_files_before": before, "child_pid": None,
             "pretraining_started": False, "downloads_started": False}
    save_state(state)
    try:
        for arm, reset, no_eos in ARMS:
            data = ROOT / ("data/sft_special_ablation_v1_noeos" if no_eos else "data/sft_verified_v2")
            run_dir = ROOT / "runs/special_ablation_v1" / arm
            checkpoint = run_dir / "step_00000313.pt"
            argv = command("tinychat.train_stage1", "--stage", "sft", "--config", ROOT / "configs/pilot.json",
                           "--data-dir", data, "--run-dir", run_dir, "--micro-batch", 8, "--grad-accum", 8,
                           "--checkpoint-every", 250, "--eval-every", 250, "--seed", 42,
                           "--init-checkpoint", ROOT / "runs/pretrain_100m_v1/step_00003052.pt",
                           "--special-ablation-diagnostics", "--reset-special-seed", 1729)
            if reset:
                argv.append("--reset-unseen-specials")
            run_child(state, "train", arm, argv)
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            if (metrics["completed_updates"] != 313 or metrics["cumulative_examples"] != 20000
                    or metrics.get("error") or metrics.get("optimizer_state_unsafe")):
                raise RuntimeError(f"Incomplete authorized SFT epoch: {arm}")
            run_child(state, "export", arm, command("tinychat.export_stage1", "--stage", "sft",
                      "--checkpoint", checkpoint, "--data-dir", data,
                      "--output", ROOT / "artifacts/special_ablation_v1" / arm))
        for arm, _, _ in ARMS:
            run_child(state, "evaluate", arm, command("scripts.evaluate_special_ablation", "model", "--arm", arm))
        run_child(state, "summarize", "all", command("scripts.evaluate_special_ablation", "summarize"))
        after = identities(protected)
        state["protected_files_after"] = after
        if before != after:
            raise RuntimeError("A protected original file changed")
        state.update({"status": "awaiting_independent_results_review", "phase": "stopped",
                      "arm": None, "child_pid": None, "completed_utc": utc_now()})
    except BaseException as exc:
        state.update({"status": "failed_stopped", "error": f"{type(exc).__name__}: {exc}"})
        save_state(state)
        raise
    save_state(state)
    print(json.dumps({"status": state["status"], "completed_phases": len(state["completed_phases"])}), flush=True)


if __name__ == "__main__":
    main()
