"""Record finite-duration nvidia-smi telemetry during the pilot."""
import argparse
import csv
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1200:
        raise ValueError("Telemetry is limited to 1200 seconds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    end = time.monotonic() + args.seconds
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "memory_used_mib", "utilization_percent", "power_watts", "temperature_c"])
        while time.monotonic() < end:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                check=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
            for row in csv.reader(result.stdout.strip().splitlines()):
                writer.writerow([cell.strip() for cell in row])
            handle.flush()
            time.sleep(min(5, max(0, end-time.monotonic())))


if __name__ == "__main__":
    main()
