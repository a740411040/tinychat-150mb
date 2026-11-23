"""Bounded read-only source metadata/schema probe for the authorized comparison."""
import json
import os
import sqlite3
from pathlib import Path
from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq

root = Path.cwd()
out = root / "planning/train_150mb/comparison_source_probe.json"
if out.exists():
    raise FileExistsError(out)
api = HfApi()
info = api.dataset_info("wikimedia/wikipedia")
files = [r for r in api.list_repo_tree("wikimedia/wikipedia", path_in_repo="20231101.zh", repo_type="dataset", revision=info.sha)
         if getattr(r, "path", "").endswith(".parquet")]
records = [{"path": row.path, "bytes": row.size} for row in files]
fs = HfFileSystem()
first = records[0]
with fs.open(f"datasets/wikimedia/wikipedia@{info.sha}/{first['path']}", "rb", block_size=1 << 20, cache_type="readahead") as file:
    parquet = pq.ParquetFile(file)
    schema = parquet.schema_arrow.names
    batch = next(parquet.iter_batches(batch_size=3, row_groups=[0], columns=["id", "url", "title", "text"]))
    examples = [{**row, "text": row["text"][:600]} for row in batch.to_pylist()]
    rows, row_groups = parquet.metadata.num_rows, parquet.num_row_groups
database = root / "data/sft_stage1_v1/acquisition.sqlite3"
with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
    tables = connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table'").fetchall()
result = {"wiki": {"repo": "wikimedia/wikipedia", "revision": info.sha, "config": "20231101.zh", "files": records,
                   "schema": schema, "first_file_rows": rows, "first_file_row_groups": row_groups, "examples": examples,
                   "usage_note": "Publisher card CC-BY-SA-3.0 and GFDL; retain page URL/title and snapshot attribution."},
          "sft_journal_tables": tables}
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"output": str(out), "revision": info.sha, "files": records, "schema": schema,
                  "first_file_rows": rows, "first_file_row_groups": row_groups, "sft_tables": [t[0] for t in tables]}, ensure_ascii=False), flush=True)
