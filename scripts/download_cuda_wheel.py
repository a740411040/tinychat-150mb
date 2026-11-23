"""Bounded ranged download for an official wheel when the proxy buffers full GETs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".cache" / "wheels"
NAME = "torch-2.12.1+cu130-cp312-cp312-win_amd64.whl"
INDEX = "https://download.pytorch.org/whl/cu130/torch/"
URL = "https://download.pytorch.org/whl/cu130/torch-2.12.1%2Bcu130-cp312-cp312-win_amd64.whl"


def curl(*args, timeout=50):
    return subprocess.run(["curl.exe", "--silent", "--show-error", "--fail", "--location",
                           "--max-time", str(timeout), *args], check=True,
                          capture_output=True, text=True, encoding="utf-8",
                          creationflags=subprocess.CREATE_NO_WINDOW).stdout


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    parts_dir = DEST / "torch_parts"
    parts_dir.mkdir(exist_ok=True)
    manifest_path = DEST / "torch_download.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["url"] != URL or manifest["index"] != INDEX:
            raise RuntimeError("Existing download manifest targets a different source")
        total, expected_sha = manifest["bytes"], manifest["sha256"]
    else:
        parser = Links()
        parser.feed(curl(INDEX))
        href = next(link for link in parser.links if unquote(urlparse(link).path).endswith("/" + NAME))
        expected_sha = urlparse(href).fragment.removeprefix("sha256=")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise RuntimeError("Official index SHA256 is missing")
        header = curl("--head", URL)
        total = int(re.findall(r"(?im)^content-length:\s*(\d+)", header)[-1])
        if not 1_000_000_000 < total < 4_000_000_000:
            raise RuntimeError(f"Unexpected official wheel size: {total}")
        manifest = {"url": URL, "index": INDEX, "bytes": total, "sha256": expected_sha,
                    "chunk_bytes": 8*2**20, "workers": 4}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    start_time = time.monotonic()
    chunk = manifest["chunk_bytes"]

    def fetch(start):
        if time.monotonic() - start_time >= 1800:
            raise TimeoutError("Bounded wheel download exceeded 30 minutes")
        end = min(total, start+chunk)-1
        expected_range = f"bytes {start}-{end}/{total}"
        for attempt in range(3):
            suffix = "" if attempt == 0 else f".retry{attempt}"
            path = parts_dir / f"{start:012d}{suffix}.part"
            headers = parts_dir / f"{start:012d}{suffix}.headers"
            if not path.exists():
                break
            if headers.exists():
                ranges = re.findall(r"(?im)^content-range:\s*(.+)", headers.read_text(encoding="utf-8"))
                if path.stat().st_size == end-start+1 and ranges and ranges[-1].strip() == expected_range:
                    return start, path, end-start+1
        else:
            raise RuntimeError(f"Part {start} has exhausted its bounded attempts")
        try:
            status = curl("--range", f"{start}-{end}", "--dump-header", str(headers),
                          "--speed-limit", "16384", "--speed-time", "30",
                          "--output", str(path), "--write-out", "%{http_code}", URL, timeout=300)
        except subprocess.CalledProcessError as error:
            path.with_suffix(".error.txt").write_text(error.stderr, encoding="utf-8")
            raise
        received = headers.read_text(encoding="utf-8")
        ranges = re.findall(r"(?im)^content-range:\s*(.+)", received)
        if status.strip() != "206" or not ranges or ranges[-1].strip() != expected_range:
            raise RuntimeError(f"Invalid content range for {start}: {status}, {ranges}")
        if path.stat().st_size != end-start+1:
            raise RuntimeError(f"Incomplete range at {start}")
        return start, path, end-start+1

    def fetch_with_transient_retry(start):
        # Only retry transport failures once; HTTP authorization/errors are not retried.
        for attempt in range(2):
            try:
                return fetch(start)
            except subprocess.CalledProcessError as error:
                if attempt or error.returncode not in (18, 28, 35, 52, 56):
                    raise
                print(json.dumps({"transient_range_failure": start,
                                  "curl_exit": error.returncode,
                                  "bounded_retry": 1}), flush=True)
                time.sleep(2)

    completed = 0
    chosen_parts = {}
    last_log = 0.0
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        pending = [pool.submit(fetch_with_transient_retry, offset) for offset in range(0, total, chunk)]
        for future in as_completed(pending, timeout=1800):
            offset, part, size = future.result()
            chosen_parts[offset] = part
            completed += size
            elapsed = time.monotonic() - start_time
            if elapsed-last_log >= 5 or completed == total:
                print(json.dumps({"completed_MiB_including_reused": round(completed/2**20, 1),
                                  "total_MiB": round(total/2**20, 1),
                                  "elapsed_s": round(elapsed, 1)}), flush=True)
                last_log = elapsed
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    target = DEST / NAME
    digest = hashlib.sha256()
    with target.open("xb") as writer:
        for offset in range(0, total, chunk):
            with chosen_parts[offset].open("rb") as reader:
                while block := reader.read(8*2**20):
                    writer.write(block)
                    digest.update(block)
    if digest.hexdigest() != expected_sha:
        raise RuntimeError("Wheel differs from the official index digest; do not install")
    print(json.dumps({"completed": True, "path": str(target),
                      "seconds": round(time.monotonic()-start_time, 1)}), flush=True)


if __name__ == "__main__":
    main()
