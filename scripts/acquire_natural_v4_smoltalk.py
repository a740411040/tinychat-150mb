"""Fetch one pinned, small public dialogue file; never fetch the whole repository."""
import json
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq
from huggingface_hub.utils import get_session

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data/public_sources/natural_sft_v4/smoltalk-chinese"
REVISION = "5edf5d0d7b06794755cf78b274a150a4efe1c9d4"
NAME = "everyday.parquet"
EXPECTED_BYTES = 16_703_662
URL = f"https://huggingface.co/datasets/opencsg/smoltalk-chinese/resolve/{REVISION}/{NAME}"


def main():
    destination = TARGET / NAME
    partial = TARGET / (NAME + ".part")
    metadata = TARGET / (NAME + ".source.json")
    metadata_partial = TARGET / (NAME + ".source.json.part")
    if any(path.exists() for path in (destination, partial, metadata, metadata_partial)):
        raise FileExistsError("Source output exists; refusing overwrite or an implicit retry")
    TARGET.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    total = 0
    next_progress = 4 * 1024 * 1024
    with get_session().stream("GET", URL, headers={"Accept-Encoding": "identity"},
                              follow_redirects=True, timeout=30) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError(f"Expected full single-file response, received {response.status_code}")
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise ValueError("Encoded response would obscure the transfer byte limit")
        size_header = response.headers.get("content-length")
        if size_header is not None and int(size_header) != EXPECTED_BYTES:
            raise ValueError("Public source size differs from the pinned metadata")
        with partial.open("xb") as output:
            for chunk in response.iter_raw(chunk_size=65536):
                if total + len(chunk) > EXPECTED_BYTES:
                    raise ValueError("Download exceeds the single-file byte limit")
                output.write(chunk)
                total += len(chunk)
                if total >= next_progress:
                    print(json.dumps({"event": "download_progress", "bytes": total,
                                      "limit": EXPECTED_BYTES}), flush=True)
                    next_progress += 4 * 1024 * 1024
    if total != EXPECTED_BYTES:
        raise ValueError(f"Incomplete download: {total} of {EXPECTED_BYTES} bytes")
    parquet = pq.ParquetFile(partial)
    rows = parquet.metadata.num_rows
    columns = parquet.schema_arrow.names
    parquet.close()
    if rows <= 0:
        raise ValueError("Empty public source")
    record = {
        "repo": "opencsg/smoltalk-chinese", "revision": REVISION,
        "file": NAME, "url": URL, "bytes": total, "rows": rows, "columns": columns,
        "card_url": f"https://huggingface.co/datasets/opencsg/smoltalk-chinese/blob/{REVISION}/README.md",
        "license_metadata": "apache-2.0",
        "additional_source_terms": "Card also requires OpenCSG Community License compliance; commercial use requires provider permission.",
        "source_type": "Synthetic five-round everyday dialogue; source label is not a quality endorsement.",
        "purpose": "Bounded candidate pilot after local source inspection; not released for training.",
        "elapsed_seconds": perf_counter() - started,
    }
    with metadata_partial.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
        output.write("\n")
    metadata_partial.rename(metadata)
    partial.rename(destination)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
