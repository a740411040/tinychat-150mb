"""Write reviewed pilot artifacts only; never create a training manifest."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/natural_sft_v4_rewrite_pilot"
REPORT = ROOT / "reports/natural_sft_v4/rewrite_pilot"


def main():
    review = json.loads((REPORT / "quality_review_v1.json").read_text(encoding="utf-8"))
    candidates = [json.loads(line) for line in (DATA / "rewrite_candidates_v1.jsonl").read_text(encoding="utf-8").splitlines() if line]
    grades = {row["id"]: row for row in review["records"]}
    if len(candidates) != 64 or len(grades) != 64 or len(review["records"]) != 64:
        raise ValueError("Final pilot must have exactly 64 distinct reviewed records")
    if len({row["id"] for row in candidates}) != 64 or {row["id"] for row in candidates} != set(grades):
        raise ValueError("Review and candidate IDs disagree")
    counts = dict(Counter(row["grade"] for row in review["records"]))
    if review["status"] != "PASS" or counts.get("MAJOR", 0) or any(grade not in {"PASS", "MINOR"} for grade in counts):
        raise ValueError("Pilot quality gate is not satisfied")
    if any(row["structural_errors"] or row["input_tokens"] > 512 for row in candidates):
        raise ValueError("A candidate has unresolved structural errors")
    outputs = (DATA / "reviewed_pilot_v1.jsonl", REPORT / "pilot_result.json")
    if any(path.exists() for path in outputs):
        raise FileExistsError("Final pilot artifacts already exist")
    writer_ids = {}
    for batch in range(1, 9):
        value = json.loads((REPORT / f"writer_batch{batch:02d}.json").read_text(encoding="utf-8"))
        if not value["structural_passed"] or value["actual_input_reads"] != 1:
            raise ValueError("Incomplete writer protocol or structural validation")
        for row in value["rows"]:
            writer_ids[row["id"]] = value["task_id"]
    turns = sum(sum(message["role"] == "assistant" for message in row["messages"]) for row in candidates)
    target_tokens = sum(row["assistant_tokens"] for row in candidates)
    with outputs[0].open("x", encoding="utf-8", newline="\n") as output:
        for row in candidates:
            result = {key: value for key, value in row.items() if key not in {"writer_response", "semantic_passed"}}
            rating = grades[row["id"]]
            result.update({"pilot_review_grade": rating["grade"], "pilot_review_reason": rating["reason"],
                           "pilot_review_scope": "Independent local model review of this 64-item pilot only; minor issues retained.",
                           "writer_task_id": writer_ids[row["id"]], "training_released": False})
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {"status": "completed_stopped", "scope": "64-item local rewrite pilot only",
               "conversations": 64, "assistant_turns": turns,
               "conversation_grades": review["conversation_counts"],
               "assistant_turn_grades": review["assistant_turn_counts"],
               "category_grades": review["category_counts"],
               "repair_controls": 4, "new_selected_seed_groups": 60,
               "input_tokens_total": sum(row["input_tokens"] for row in candidates),
               "assistant_supervised_tokens_including_end_turn": target_tokens,
               "assistant_end_turn_labels": turns, "eos_labels": 0,
               "end_turn_label_share": turns / target_tokens,
               "max_context_tokens": max(row["input_tokens"] for row in candidates),
               "final_eos": False, "user_messages_preserved": True,
               "writer_tasks_completed": 8, "new_corpus_downloads": 0,
               "training_started": False, "model_comparison_performed": False,
               "expanded_to_20000": False, "training_released": False,
               "reviewed_artifact": str(outputs[0].relative_to(ROOT)),
               "quality_review": str((REPORT / "quality_review_v1.json").relative_to(ROOT)),
               "limitations": "Model-selected and model-reviewed seed sample, not human annotation or a random population estimate. Six minor issues remain. No claim about model capability or a full20k dataset."}
    with outputs[1].open("x", encoding="utf-8", newline="\n") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
