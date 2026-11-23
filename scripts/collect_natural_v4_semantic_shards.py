"""Validate bounded Kimi selection results; these reports never release training data."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/natural_sft_v4"
PLANNING = ROOT / "planning/train_150mb"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def collect(batch, result_path):
    result = read(result_path)
    if not result.get("executionPass") or not result.get("taskCompleted"):
        raise ValueError("Incomplete external task is not a valid selection result")
    if result.get("unexpectedFiles") or result.get("changedFiles"):
        raise ValueError("External task violated its read-only protocol")
    reads = [event for event in result["events"] if event.get("method") == "fs/read_text_file"]
    if len(reads) != 1 or Path(reads[0]["params"]["path"]).name != "input.txt":
        raise ValueError("Expected one actual read of the isolated input file")
    if not any(event.get("id") == reads[0]["id"] and "result" in event for event in result["events"]):
        raise ValueError("The input read has no response")
    packet = read(PLANNING / f"natural_sft_v4_semantic_round2_shard{batch}_input.txt")
    inputs = {row["id"]: row for row in packet["records"]}
    value = json.loads(result["finalText"])
    rows = value["rows"]
    if len(rows) != 8 or {row["id"] for row in rows} != set(inputs):
        raise ValueError("Selection output IDs/count do not match the assigned batch")
    for row in rows:
        category, count = row["category"], row["prefix_exchanges"]
        if category not in {"natural", "multi", "other", "reject"} or type(count) is not int:
            raise ValueError("Invalid category/prefix count")
        if not 0 <= count <= len(inputs[row["id"]]["messages"]) // 2:
            raise ValueError("Selected prefix is not a complete original prefix")
        if (count == 0) != (category == "reject"):
            raise ValueError("Rejected and accepted prefix accounting disagree")
        if category == "multi":
            if count < 2 or row["multi_subtype"] not in {"daily", "document", "work"} or not row.get("dependency_evidence"):
                raise ValueError("Multi selection lacks a checkable dependency claim")
        elif row["multi_subtype"] != "none":
            raise ValueError("Non-multi row has a multi subtype")
    value["protocol"] = {"taskId": result["taskId"], "batch": batch, "executionPass": True,
                         "taskCompleted": True, "input_reads": 1, "structural_rows_valid": 8,
                         "semantics_approved": False, "training_released": False}
    write_new(REPORT / f"semantic_round2_shard{batch}_kimi.json", value)
    print(json.dumps({"batch": batch, "counts": dict(Counter(row["category"] for row in rows))}))


def merge():
    shards = [read(REPORT / f"semantic_round2_shard{batch}_kimi.json") for batch in range(1, 5)]
    packet = read(PLANNING / "natural_sft_v4_semantic_round2_input.txt")
    rows = [row for shard in shards for row in shard["rows"]]
    by_id = {row["id"]: row for row in rows}
    if len(rows) != 32 or len(by_id) != 32 or set(by_id) != {row["id"] for row in packet["records"]}:
        raise ValueError("Merged output is not exactly the original 32-record pilot")
    merged = {"rows": [by_id[row["id"]] for row in packet["records"]],
              "protocols": [shard["protocol"] for shard in shards],
              "strategies_by_batch": [shard.get("strategy") for shard in shards],
              "failed_prior_task": "2026-11-22T16-06-12-053Z-kimi-a6da6284: provider timeout180s; no final text; not used",
              "semantics_approved": False, "training_released": False}
    write_new(REPORT / "semantic_round2_kimi.json", merged)
    print(json.dumps({"records": 32, "counts": dict(Counter(row["category"] for row in rows))}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=range(1, 5))
    parser.add_argument("--result", type=Path)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        if args.batch is not None or args.result is not None:
            parser.error("Choose collection or merge, not both")
        merge()
    elif args.batch is not None and args.result is not None:
        collect(args.batch, args.result)
    else:
        parser.error("Supply --batch and --result, or --merge")


if __name__ == "__main__":
    main()
