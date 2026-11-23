"""Fetch only bounded Parquet text ranges from pinned Stage 2 sources."""

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path
from ssl import SSLEOFError
import time
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import psutil
import pyarrow.parquet as pq

from tinychat.stage1_data import file_identity, json_line, write_json
from tinychat.stage2_runtime import DiskBudgetError, ensure_disk_budget


MIB = 1024 * 1024
GIB = 1024 * MIB
MAX_TOTAL_DOWNLOAD = 8_000_000_000
HUB = "https://huggingface.co"


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    directories: tuple[str, ...]
    licenses: tuple[str, ...]
    license_terms: str


SOURCES = {
    "fineweb2_hq": Source(
        "epfml/FineWeb2-HQ", "c0c06e94fd3a44ae9e802b2b0fc533817601eb5e",
        ("cmn_Hani",), ("ODC-By-1.0", "Common Crawl Terms of Use"),
        "ODC-By attribution and Common Crawl terms apply.",
    ),
    "fineweb_edu": Source(
        "opencsg/chinese-fineweb-edu", "784f118864c0741ce32aa74d9588795bdf1a896b",
        ("IndustryCorpus", "Skypile", "cci2", "map", "tele"), ("Apache-2.0", "OpenCSG Community License"),
        "Dataset card requires the OpenCSG Community License and explicitly asks commercial users for permission.",
    ),
    "cosmopedia": Source(
        "opencsg/chinese-cosmopedia", "b15fa10b9ec8ac30a719b46eead5ffbdaf644d50",
        ("data",), ("Apache-2.0", "OpenCSG Community License"),
        "Dataset card requires the OpenCSG Community License and explicitly asks commercial users for permission.",
    ),
}


class TransferBudgetExceeded(RuntimeError):
    pass


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, budget, timeout):
        self.url = url
        self.size = size
        self.budget = budget
        self.timeout = timeout
        self.position = 0
        self.bytes_read = 0
        self.requests = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        bases = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        position = bases[whence] + offset
        if position < 0:
            raise ValueError("Cannot seek to a negative offset")
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0:
            raise TransferBudgetExceeded("Unbounded remote reads are disabled")
        size = min(size, max(0, self.size - self.position))
        if size > self.budget - self.bytes_read:
            raise TransferBudgetExceeded(f"Range of {size:,} bytes exceeds the remaining transfer budget")
        if size >= 32 * MIB and psutil.virtual_memory().available < 2 * (GIB + size):
            raise MemoryError(f"A {size:,}-byte Parquet range cannot be read while preserving 2 GiB of RAM")
        return super().read(size)

    def open_range(self, request):
        for attempt in range(2):
            self.requests += 1
            try:
                return urlopen(request, timeout=self.timeout)
            except (URLError, TimeoutError) as exc:
                reason = exc.reason if isinstance(exc, URLError) else exc
                if attempt or not isinstance(reason, (SSLEOFError, TimeoutError)):
                    raise
                print("[stage2-acquire] retrying interrupted connection before reading the range", flush=True)
                time.sleep(1)

    def readinto(self, target):
        length = max(0, min(len(target), self.size - self.position))
        if length == 0:
            return 0
        if length > self.budget - self.bytes_read:
            raise TransferBudgetExceeded(
                f"Range of {length:,} bytes exceeds remaining {self.budget - self.bytes_read:,} bytes"
            )
        start = self.position
        end = start + length - 1
        if length == self.size:
            raise TransferBudgetExceeded("Full-shard transfers are disabled")
        request = Request(self.url, headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "tinychat-stage2-acquisition/1",
        })
        with self.open_range(request) as response:
            expected_range = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise OSError(f"Server did not honor the exact range {expected_range}")
            received = 0
            while received < length:
                chunk = response.read(min(MIB, length - received))
                if not chunk:
                    raise OSError(f"Short Parquet range read at offset {start}")
                target[received:received + len(chunk)] = chunk
                received += len(chunk)
                previous = self.bytes_read
                self.bytes_read += len(chunk)
                if self.bytes_read // (32 * MIB) != previous // (32 * MIB):
                    print(f"[stage2-acquire] {self.bytes_read / MIB:.1f} MiB read", flush=True)
        self.position += length
        return length


def text_column_groups(parquet):
    groups = []
    for index in range(parquet.num_row_groups):
        group = parquet.metadata.row_group(index)
        column = next(
            (group.column(i) for i in range(group.num_columns) if group.column(i).path_in_schema == "text"),
            None,
        )
        if column is None:
            raise ValueError("Parquet row group has no top-level text column")
        groups.append({
            "id": index, "rows": group.num_rows,
            "compressed_bytes": column.total_compressed_size,
            "uncompressed_bytes": column.total_uncompressed_size,
        })
    return groups


def choose_groups(groups, requested, remaining_bytes, max_memory_bytes, count=1):
    eligible = [
        group for group in groups
        if group["compressed_bytes"] <= remaining_bytes
        and group["uncompressed_bytes"] <= max_memory_bytes
    ]
    if requested:
        if len(set(requested)) != len(requested) or not set(requested) <= {g["id"] for g in eligible}:
            raise TransferBudgetExceeded("Requested row groups exceed the transfer or decoded-column bounds")
        selected = [g for g in eligible if g["id"] in requested]
    else:
        selected = eligible[:count]
    if not selected:
        raise TransferBudgetExceeded("No text row group fits the transfer and decoded-column bounds")
    if sum(g["compressed_bytes"] for g in selected) > remaining_bytes:
        raise TransferBudgetExceeded("Selected text columns cumulatively exceed the remaining transfer budget")
    return selected


def source_url(source, path):
    components = path.split("/")
    if (len(components) != 2 or components[0] not in source.directories
            or not components[1].endswith(".parquet") or ".." in components[1] or "\\" in path):
        raise ValueError(f"Path is outside the pinned source: {path}")
    return f"{HUB}/datasets/{source.repo}/resolve/{source.revision}/{quote(path)}"


def peak_working_set():
    memory = psutil.Process().memory_info()
    return getattr(memory, "peak_wset", memory.rss)


def write_text_batch(stream, path, texts, shard, group, first_row):
    payload = b"".join(json_line({
        "text": text, "origin": shard, "row_group": group, "row_in_group": index,
    }).encode("utf-8") for index, text in enumerate(texts, start=first_row))
    ensure_disk_budget(path, required_bytes=len(payload))
    stream.write(payload)


def recover_row_groups(previous, stream, path, selected, shard, limit):
    """Copy the validated complete prefix without writing into the prior attempt."""
    rows = 0
    copied_bytes = 0
    complete_groups = []
    with previous.open("rb") as saved:
        for group in selected:
            expected = min(group["rows"], limit - rows)
            if expected <= 0:
                break
            for row_index in range(expected):
                line = saved.readline()
                if not line.endswith(b"\n"):
                    break
                row = json.loads(line)
                if (not isinstance(row, dict) or row.get("origin") != shard or row.get("row_group") != group["id"]
                        or row.get("row_in_group") != row_index or "text" not in row
                        or (row["text"] is not None and not isinstance(row["text"], str))):
                    raise ValueError(f"Invalid recovery row {group['id']}:{row_index} in {previous}")
            else:
                rows += expected
                copied_bytes = saved.tell()
                complete_groups.append(group["id"])
                continue
            break
        else:
            if saved.read(1):
                raise ValueError(f"Recovery input exceeds the selected row groups: {previous}")
        if rows == limit and saved.read(1):
            raise ValueError(f"Recovery input exceeds the requested row count: {previous}")
        ensure_disk_budget(path, required_bytes=copied_bytes)
        saved.seek(0)
        remaining = copied_bytes
        while remaining:
            chunk = saved.read(min(MIB, remaining))
            if not chunk:
                raise OSError(f"Recovery input changed while being copied: {previous}")
            stream.write(chunk)
            remaining -= len(chunk)
    return {"rows": rows, "bytes": copied_bytes, "groups": complete_groups}


def acquire(args):
    source = SOURCES[args.source]
    started = time.monotonic()
    report = {
        "source": args.source, "repo": source.repo, "revision": source.revision,
        "license": list(source.licenses), "shard": args.shard, "shard_bytes": args.shard_bytes,
        "license_terms": source.license_terms, "dataset_card": f"{HUB}/datasets/{source.repo}/blob/{source.revision}/README.md",
        "rows": 0, "transferred_bytes": 0, "range_requests": 0, "metadata_only": args.output is None,
    }
    if args.output is not None:
        ensure_disk_budget(args.output, required_bytes=args.budget)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
    ensure_disk_budget(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    status = "failed"
    with RangeFile(source_url(source, args.shard), args.shard_bytes, args.budget, args.timeout) as remote:
        try:
            with pq.ParquetFile(remote, pre_buffer=False) as parquet:
                groups = text_column_groups(parquet)
                report["groups"] = groups
                if args.output is not None:
                    selected = choose_groups(groups, args.row_group, args.budget - remote.bytes_read,
                                             args.max_column_bytes, args.groups)
                    report["selected_groups"] = [group["id"] for group in selected]
                    partial = args.output.with_name(args.output.name + ".partial")
                    with partial.open("xb") as stream:
                        resume = getattr(args, "resume_from", None)
                        if resume is not None:
                            original = file_identity(resume)
                            recovered = recover_row_groups(
                                resume, stream, partial, selected, args.shard, args.limit_rows,
                            )
                            if file_identity(resume) != original:
                                raise ValueError(f"Recovery input changed while being read: {resume}")
                            report["resumed_from"] = {"path": str(resume.resolve()), "file": original, **recovered}
                            report["rows"] = recovered["rows"]
                            print(f"[stage2-acquire] recovered {report['rows']:,} rows from {resume.name}", flush=True)
                        else:
                            recovered = {"groups": []}
                        for group in selected[len(recovered["groups"]):]:
                            if report["rows"] >= args.limit_rows:
                                break
                            ensure_disk_budget(partial, required_bytes=group["uncompressed_bytes"])
                            row = 0
                            batches = parquet.iter_batches(
                                32, row_groups=[group["id"]], columns=["text"], use_threads=False,
                            )
                            for batch in batches:
                                if peak_working_set() > 2.5 * GIB or psutil.virtual_memory().available < 2 * GIB:
                                    raise MemoryError("Parquet reader exceeded the RSS bound or the 2 GiB system RAM reserve")
                                texts = batch.column(0).to_pylist()[:args.limit_rows - report["rows"]]
                                write_text_batch(stream, partial, texts, args.shard, group["id"], row)
                                row += len(texts)
                                report["rows"] += len(texts)
                                if report["rows"] == args.limit_rows:
                                    break
                    partial.replace(args.output)
                    report["output_file"] = {"path": str(args.output.resolve()), **file_identity(args.output)}
                status = "success"
        except (OSError, DiskBudgetError, TransferBudgetExceeded, ValueError, MemoryError) as exc:
            report["error"] = str(exc)
            raise
        finally:
            report.update(status=status, elapsed_seconds=round(time.monotonic() - started, 3),
                          transferred_bytes=remote.bytes_read, range_requests=remote.requests,
                          peak_rss_bytes=peak_working_set())
            write_json(args.report, report)
            summary = {k: v for k, v in report.items() if k != "groups"}
            print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--shard-bytes", required=True, type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-from", type=Path,
                        help="Read only complete row groups from a previous partial JSONL.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=96 * MIB)
    parser.add_argument("--max-column-bytes", type=int, default=128 * MIB)
    parser.add_argument("--row-group", type=int, action="append")
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--limit-rows", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()
    if not 0 < args.budget <= MAX_TOTAL_DOWNLOAD or min(
        args.shard_bytes, args.max_column_bytes, args.groups, args.limit_rows, args.timeout,
    ) <= 0:
        parser.error("Limits and shard size must be positive, with a transfer budget at most 8 GB")
    if args.resume_from is not None and (args.output is None or not args.resume_from.is_file()):
        parser.error("Recovery requires an existing partial JSONL and a new output path")
    acquire(args)


if __name__ == "__main__":
    main()
