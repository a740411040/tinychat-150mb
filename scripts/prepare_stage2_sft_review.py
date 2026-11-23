"""Package the fixed independent sample into bounded text-only review inputs."""
import json
import argparse
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=root / "reports/stage2_500m/sft_candidate_blind.jsonl")
    parser.add_argument("--output", type=Path, default=root / "reports/stage2_500m/sft_review")
    args = parser.parse_args()
    source, destination = args.sample.resolve(), args.output.resolve()
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) != 250 or len({row["blind_id"] for row in rows}) != 250:
        raise ValueError("Expected the fixed 250 distinct blind examples")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"sample": str(source), "count": len(rows), "review_calls_completed": 0, "batches": []}
    batches, batch, size = [], [], 0
    for row in rows:
        clean = {"blind_id": row["blind_id"], "messages": row["messages"]}
        line = json.dumps(clean, ensure_ascii=False, separators=(",", ":")) + "\n"
        count = len(line.encode("utf-8"))
        if count > 52_000:
            raise ValueError("One example exceeds the text-only batch allowance")
        if batch and size + count > 52_000:
            batches.append(batch)
            batch, size = [], 0
        batch.append((clean["blind_id"], line))
        size += count
    if batch:
        batches.append(batch)
    for index, batch in enumerate(batches, 1):
        path = destination / f"batch_{index:02}.txt"
        header = f"独立抽检批次 {index}/{len(batches)}，本批 {len(batch)} 条。只审核，不重写。\n"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(header + "".join(line for _, line in batch))
        manifest["batches"].append({"input": str(path), "count": len(batch), "bytes": path.stat().st_size,
                                    "ids": [identity for identity, _ in batch]})
    with (destination / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"count": len(rows), "batches": len(batches),
                      "sizes": [entry["bytes"] for entry in manifest["batches"]]}))


if __name__ == "__main__":
    main()
