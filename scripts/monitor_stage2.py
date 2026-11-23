"""Sample whole-device GPU telemetry until a Stage 2 invocation writes its stop metrics."""

import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import subprocess
import time


COLUMNS = ("memory_used_MiB", "memory_total_MiB", "gpu_utilization_percent", "temperature_C", "power_W")
STOP_REASONS = ("complete", "paused", "paused_disk", "interrupted", "failure")
NOTE = "nvidia-smi device-wide telemetry includes other GPU users; trainer allocator counters are separate"


def metrics_mtime(path):
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def read_stop(path, baseline):
    mtime = metrics_mtime(path)
    if mtime is None or mtime == baseline:
        return None
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(metrics, dict) and metrics.get("stop_reason") in STOP_REASONS:
        return metrics, mtime
    return None


def query_gpu(timeout):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("GPU query failed or returned no rows")
    cells = next(csv.reader(result.stdout.splitlines()))
    if len(cells) != len(COLUMNS):
        raise RuntimeError("Unexpected GPU telemetry columns")
    values = []
    for cell in cells:
        try:
            value = float(cell.strip())
        except ValueError:
            value = None
        values.append(value if value is not None and math.isfinite(value) else None)
    return values


def monitor(args):
    metrics_path = args.run_dir / "metrics.json"
    baseline = None if args.accept_existing_stop else metrics_mtime(metrics_path)
    next_sample = time.monotonic()
    deadline = next_sample + args.max_seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    peaks = dict.fromkeys(COLUMNS)
    samples, failures = 0, 0
    stop = None
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utc", *COLUMNS])
        while True:
            stop = read_stop(metrics_path, baseline)
            now = time.monotonic()
            remaining = deadline - now
            if stop is not None or remaining <= 0:
                break
            if now < next_sample:
                time.sleep(min(next_sample - now, remaining))
                continue
            try:
                values = query_gpu(min(15, remaining))
            except (subprocess.TimeoutExpired, OSError, RuntimeError, StopIteration):
                failures += 1
            else:
                writer.writerow([datetime.datetime.now(datetime.timezone.utc).isoformat(), *values])
                handle.flush()
                samples += 1
                for name, value in zip(COLUMNS, values):
                    if value is not None:
                        peaks[name] = value if peaks[name] is None else max(peaks[name], value)
            next_sample = time.monotonic() + args.interval

    metrics, mtime = stop if stop is not None else ({}, None)
    summary = {"pid": args.pid, "process_create_time": None, "samples": samples, "query_failures": failures,
               "peaks": peaks, "note": NOTE, "run_dir": str(args.run_dir.resolve()),
               "stop_reason": metrics.get("stop_reason", "timeout"), "metrics_mtime_ns": mtime,
               "checkpoint_step": metrics.get("checkpoint_step")}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--accept-existing-stop", action="store_true")
    parser.add_argument("--pid", type=int, help="Optional summary label")
    args = parser.parse_args()
    if not 1 <= args.interval <= 60 or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--interval must be 1..60 and --max-seconds must be finite and positive")
    print(json.dumps(monitor(args)), flush=True)


if __name__ == "__main__":
    main()
