"""Record GPU telemetry until a particular owned training PID exits."""
import argparse
import csv
import datetime
import json
import subprocess
import time
from pathlib import Path

import psutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interval", type=float, default=10)
    args = p.parse_args()
    if not 1 <= args.interval <= 60:
        raise ValueError("interval must be 1..60 seconds")
    process = psutil.Process(args.pid)
    started = process.create_time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = ["memory_used_MiB", "memory_total_MiB", "gpu_utilization_percent", "temperature_C", "power_W"]
    count = 0
    peaks = {key: None for key in names}
    failures = 0
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utc", *names])
        while True:
            try:
                if not process.is_running() or process.create_time() != started:
                    break
            except psutil.NoSuchProcess:
                break
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
                lines = result.stdout.splitlines()
                if result.returncode != 0 or not lines:
                    raise RuntimeError("GPU query failed or returned no rows")
                cells = next(csv.reader(lines))
                if len(cells) != len(names):
                    raise RuntimeError("Unexpected GPU telemetry columns")
                values = []
                for cell in cells:
                    try:
                        values.append(float(cell.strip()))
                    except ValueError:
                        values.append(None)
                writer.writerow([datetime.datetime.now(datetime.timezone.utc).isoformat(), *values])
                handle.flush()
                for name, value in zip(names, values):
                    if value is not None:
                        peaks[name] = value if peaks[name] is None else max(peaks[name], value)
                count += 1
            except (subprocess.TimeoutExpired, OSError, RuntimeError, StopIteration):
                failures += 1
            time.sleep(args.interval)
    summary = {"pid": args.pid, "process_create_time": started, "samples": count, "query_failures": failures, "peaks": peaks, "note": "nvidia-smi device-wide telemetry includes other GPU users; trainer allocator counters are separate"}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
