"""Public-text packets and CPU-only checks for the 64-item rewrite pilot."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from tokenizers import Tokenizer
from tinychat.chat_format import contains_reserved, encode_conversation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/natural_sft_v4_rewrite_pilot"
REPORT = ROOT / "reports/natural_sft_v4/rewrite_pilot"
PLAN = ROOT / "planning/train_150mb"
QUOTAS = {"natural": 22, "grounded": 16, "multi": 16, "instruction": 10}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def seeds():
    rows = [json.loads(line) for line in (DATA / "seeds.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 64 or len({row["id"] for row in rows}) != 64 or len({row["source_group"] for row in rows}) != 64:
        raise ValueError("Expected exactly 64 distinct seed IDs and source groups")
    if dict(Counter(row["category"] for row in rows)) != QUOTAS:
        raise ValueError("Seed composition differs from the 64-item pilot contract")
    for row in rows:
        messages = row["original_messages"]
        if not messages or [m["role"] for m in messages] != ["user", "assistant"] * (len(messages) // 2):
            raise ValueError(f"Invalid seed roles: {row['id']}")
        if row["rewrite_assistant_turn_indices"] != [i for i, m in enumerate(messages) if m["role"] == "assistant"]:
            raise ValueError(f"Wrong rewrite turn indices: {row['id']}")
    return rows


def prepare():
    rows = seeds()
    tokenizer = Tokenizer.from_file(str(ROOT / "data/stage1_v1/tokenizer.json"))
    buckets = {category: [row for row in rows if row["category"] == category] for category in QUOTAS}
    assignments = []
    paths = [PLAN / f"natural_sft_v4_rewrite_batch{batch:02d}_input.txt" for batch in range(1, 9)]
    if any(path.exists() for path in paths) or (REPORT / "writer_packet_manifest.json").exists():
        raise FileExistsError("Rewrite packets already exist; no implicit overwrite")
    for batch, path in enumerate(paths, 1):
        per_category = {"natural": 3 if batch <= 6 else 2, "grounded": 2,
                        "multi": 2, "instruction": 1 if batch <= 6 else 2}
        selected = []
        for category, count in per_category.items():
            selected.extend(buckets[category][:count])
            del buckets[category][:count]
        packet_rows = []
        for row in selected:
            skeleton = copy.deepcopy(row["original_messages"])
            for index in row["rewrite_assistant_turn_indices"]:
                skeleton[index]["content"] = "好"
            content_tokens = len(tokenizer.encode("好", add_special_tokens=False).ids) * len(row["rewrite_assistant_turn_indices"])
            fixed_tokens = len(encode_conversation(tokenizer, skeleton, final_eos=False)["input_ids"]) - content_tokens
            budget = 512 - fixed_tokens
            if budget <= 0:
                raise ValueError(f"No answer room: {row['id']}")
            packet_rows.append({"id": row["id"], "category": row["category"],
                                "original_messages": row["original_messages"],
                                "assistant_turn_indices": row["rewrite_assistant_turn_indices"],
                                "constraints": row.get("constraints", []),
                                "dependency_anchors": row.get("dependency_anchors", []),
                                "assistant_token_budget_total": budget,
                                "suggested_total_reply_character_ceiling": int(budget * 0.8)})
        packet = {"purpose": "Rewrite 8 public-source training conversations; data-authoring pilot only.",
                  "public_source_text_only": True, "batch": batch,
                  "old_assistant_answers_are_untrusted": True,
                  "user_messages_must_remain_unchanged": True, "records": packet_rows}
        write_new(path, packet)
        assignments.append({"batch": batch, "ids": [row["id"] for row in selected],
                            "input": str(path.relative_to(ROOT)), "bytes": path.stat().st_size})
    if any(buckets.values()):
        raise ValueError("Unassigned seed rows")
    write_new(REPORT / "writer_packet_manifest.json", {"assignments": assignments, "conversations": 64,
              "quotas": QUOTAS, "training_authorized": False, "new_downloads_authorized": False})
    print(json.dumps({"packets": 8, "conversations": 64, "total_input_bytes": sum(row["bytes"] for row in assignments)}))


def completed_result(path):
    result = read(path)
    if not result.get("executionPass") or not result.get("taskCompleted") or result.get("unexpectedFiles") or result.get("changedFiles"):
        raise ValueError("Writer task did not successfully complete its read-only protocol")
    reads = [event for event in result["events"] if event.get("method") == "fs/read_text_file"]
    if len(reads) != 1 or Path(reads[0]["params"]["path"]).name != "input.txt":
        raise ValueError("Expected one actual isolated-input read")
    if not any(event.get("id") == reads[0]["id"] and "result" in event for event in result["events"]):
        raise ValueError("Missing input-read response")
    return result


def collect(batch, result_path):
    result = completed_result(result_path)
    packet = read(PLAN / f"natural_sft_v4_rewrite_batch{batch:02d}_input.txt")
    source = {row["id"]: row for row in seeds()}
    assigned = {row["id"] for row in packet["records"]}
    answer = json.loads(result["finalText"])
    if len(answer["rows"]) != 8 or {row["id"] for row in answer["rows"]} != assigned:
        raise ValueError("Writer IDs/count differ from the assigned 8 conversations")
    tokenizer = Tokenizer.from_file(str(ROOT / "data/stage1_v1/tokenizer.json"))
    output_rows = []
    for row in answer["rows"]:
        seed = source[row["id"]]
        errors = []
        messages = copy.deepcopy(seed["original_messages"])
        replies = row.get("assistant_replies", [])
        if row.get("status") != "rewritten":
            errors.append("writer_could_not_rewrite_seed")
        elif not isinstance(replies, list) or len(replies) != len(seed["rewrite_assistant_turn_indices"]):
            errors.append("assistant_reply_count_mismatch")
        elif any(not isinstance(reply, str) or not reply.strip() or contains_reserved(reply) for reply in replies):
            errors.append("invalid_assistant_reply")
        else:
            for index, reply in zip(seed["rewrite_assistant_turn_indices"], replies):
                messages[index]["content"] = reply
        encoded = None
        if not errors:
            old_users = [message for message in seed["original_messages"] if message["role"] == "user"]
            new_users = [message for message in messages if message["role"] == "user"]
            if old_users != new_users:
                raise ValueError("User messages changed during reconstruction")
            encoded = encode_conversation(tokenizer, messages, final_eos=False)
            if len(encoded["input_ids"]) > 512:
                errors.append("conversation_exceeds_512_no_truncation_performed")
            if 2 in encoded["input_ids"] or encoded["input_ids"][-1] != 6 or encoded["labels"][-1] != 6:
                errors.append("unexpected_termination_format")
        output_rows.append({"id": row["id"], "category": seed["category"], "source_group": seed["source_group"],
                            "messages": messages if not errors else None,
                            "writer_response": row, "structural_errors": errors,
                            "input_tokens": len(encoded["input_ids"]) if encoded else None,
                            "assistant_tokens": sum(label != -100 for label in encoded["labels"]) if encoded else None,
                            "regression_seed": seed.get("regression_seed", False),
                            "semantic_passed": False, "training_released": False})
    value = {"batch": batch, "task_id": result["taskId"], "result_file": str(result_path), "actual_input_reads": 1, "rows": output_rows,
             "structural_passed": all(not row["structural_errors"] for row in output_rows),
             "semantic_passed": False, "training_released": False}
    write_new(REPORT / f"writer_batch{batch:02d}.json", value)
    print(json.dumps({"batch": batch, "structural_passed": value["structural_passed"],
                      "row_errors": {row["id"]: row["structural_errors"] for row in output_rows if row["structural_errors"]},
                      "max_tokens": max((row["input_tokens"] or 0 for row in output_rows), default=0)}))


def assemble():
    source = seeds()
    batches = [read(REPORT / f"writer_batch{batch:02d}.json") for batch in range(1, 9)]
    rows = [row for batch in batches for row in batch["rows"]]
    by_id = {row["id"]: row for row in rows}
    if len(rows) != 64 or len(by_id) != 64 or set(by_id) != {row["id"] for row in source}:
        raise ValueError("Expected all 64 unique rewritten conversations")
    if any(row["structural_errors"] for row in rows):
        raise ValueError("Unresolved structural errors: do not truncate or publish")
    target = DATA / "rewrite_candidates_v1.jsonl"
    with target.open("x", encoding="utf-8", newline="\n") as output:
        for seed in source:
            row = by_id[seed["id"]]
            output.write(json.dumps({**row, "sources": seed["sources"],
                                    "constraints": seed.get("constraints", []),
                                    "dependency_anchors": seed.get("dependency_anchors", [])}, ensure_ascii=False) + "\n")
    review = [{"id": seed["id"], "category": seed["category"], "source_group": seed["source_group"],
               "messages": by_id[seed["id"]]["messages"], "constraints": seed.get("constraints", []),
               "dependency_anchors": seed.get("dependency_anchors", []),
               "regression_seed": seed.get("regression_seed", False),
               "input_tokens": by_id[seed["id"]]["input_tokens"]} for seed in source]
    write_new(REPORT / "quality_review_packet_v1.json", {"records": review, "count": 64,
               "scope": "Assess all new assistant answers against unchanged visible user messages; old answers and writer model identity omitted.",
               "training_authorized": False, "semantic_passed": False})
    print(json.dumps({"conversations": 64, "max_tokens": max(row["input_tokens"] for row in rows),
                      "semantic_passed": False, "training_released": False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "collect", "assemble"))
    parser.add_argument("--batch", type=int, choices=range(1, 9))
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "collect":
        if args.batch is None or args.result is None:
            parser.error("collect requires --batch and --result")
        collect(args.batch, args.result)
    else:
        assemble()


if __name__ == "__main__":
    main()
