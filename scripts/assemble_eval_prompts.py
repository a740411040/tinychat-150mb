"""Preserve complete authoring records from one timeout and its bounded follow-up."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
planning = root / "planning/train_150mb"
partial = (planning / "eval_test_partial.jsonl").read_text(encoding="utf-8")
remainder = (planning / "eval_test_remainder.jsonl").read_text(encoding="utf-8")
rows = []
discarded_partial_lines = 0
for line in partial.splitlines():
    if not line.strip():
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        discarded_partial_lines += 1
assert len(rows) == 66 and discarded_partial_lines == 1
rows.extend(json.loads(line) for line in remainder.splitlines() if line.strip())
assert [row["id"] for row in rows] == [f"test_{i:03d}" for i in range(1, 101)]
for row in rows:
    if row["category"] in {"rewriting", "uncertainty", "multi_turn", "everyday_chat", "summarization"}:
        row["expected"] = None  # Rubrics, not exact-string matching, grade open answers.
path = planning / "eval_test_complete.jsonl"
with path.open("x", encoding="utf-8", newline="\n") as out:
    for row in rows:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"complete_records": len(rows), "discarded_truncated_lines": discarded_partial_lines, "output": str(path)}))
