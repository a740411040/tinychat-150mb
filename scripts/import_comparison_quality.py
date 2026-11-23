import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    inputs = json.loads(args.input.read_text(encoding="utf-8"))["cases"]
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if not result["executionPass"] or not result["taskCompleted"] or result.get("providerErrors"):
        raise ValueError("Quality reviewer did not complete")
    rows = [json.loads(line) for line in result["finalText"].strip().splitlines() if line.strip()]
    if len(rows) != len(inputs) or {r["id"] for r in rows} != {r["id"] for r in inputs}:
        raise ValueError("Missing, duplicated or additional sample IDs")
    if any(r["quality"] not in {"good", "mixed", "poor"} or not r["reason"].strip() for r in rows):
        raise ValueError("Invalid quality rubric")
    cases = {r["id"]: r for r in inputs}
    counts = {group: dict(Counter(r["quality"] for r in rows if r["id"].startswith(group)))
              for group in {r["id"].split("_")[0] for r in rows}}
    out = {"judge": "Kimi via approved read-only safe entry; not human review",
           "task_id": result["taskId"], "result_path": str(args.result), "input_path": str(args.input),
           "counts": counts, "scores": [{**r, "document_id": cases[r["id"]].get("document_id"),
                                           "source": cases[r["id"]].get("source")} for r in rows],
           "limitations": "Small stratified excerpt sample; no claim of whole-document or population factual correctness."}
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(out, file, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "counts": counts}))


if __name__ == "__main__":
    main()
