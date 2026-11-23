"""Repair the seven explicitly reviewed turns in six pilot conversations."""
import argparse
import copy
import json
from pathlib import Path

from tokenizers import Tokenizer
from scripts.natural_v4_rewrite_pilot import completed_result, write_new
from tinychat.chat_format import contains_reserved, encode_conversation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/natural_sft_v4_rewrite_pilot"
REPORT = ROOT / "reports/natural_sft_v4/scale20k"
PACKET = ROOT / "planning/train_150mb/natural_sft_v4_minor_repairs_input.txt"
FIXES = {
    "rewrite-006": ([1], "保留沪杭苏，不在出发和返程地点未知时声称该城市顺序能减少折返；用明确条件表达路线选择。"),
    "rewrite-007": ([1], "称量瓶盖中的调料前明确去皮或减去空盖重量；不猜统一克数、容量或不同调料的密度。后答已给实测方法，保持一致。"),
    "rewrite-014": ([1, 3], "只把用户实际说到的时段和温度当已知；晚间未提供则用条件句。分层穿衣同时说明冷时保暖、热时脱减，别以透气替代温差大时需要的保暖。"),
    "rewrite-023": ([1], "在只有旋律、没有原曲时也给可操作办法，例如自行录下哼唱再用支持哼唱识曲的工具，不要只等原曲再出现。不要声称本助手可以听录音；可请用户提供记得的歌词/节奏/场景。"),
    "rewrite-040": ([1], "保留散步、好天气、温暖阳光三项信息和并列关系；不可加因为或悠闲等原文没有的因果/状态。仍须是一句连贯的话，且至少一个子句用同义词改写。"),
    "rewrite-050": ([1], "保留野餐的实用食物搭配，简短补上易腐馅料需要保冷并及时食用，或换耐放食物的条件。不要添加无依据的精确安全时长或长篇警告。"),
}


def originals():
    return [json.loads(line) for line in (DATA / "reviewed_pilot_v1.jsonl").read_text(encoding="utf-8").splitlines() if line]


def prepare():
    rows = {row["id"]: row for row in originals()}
    packet = {"purpose": "User-authorized correction of six reviewed public-source dialogue derivatives; no training or downloads.",
              "records": [{"id": key, "messages": rows[key]["messages"], "fix_indices": indices,
                           "repair_note": note} for key, (indices, note) in FIXES.items()]}
    write_new(PACKET, packet)
    print(json.dumps({"conversations": 6, "turns_to_fix": 7, "input_bytes": PACKET.stat().st_size}))


def collect(result_path):
    raw = completed_result(result_path)
    result = json.loads(raw["finalText"])
    if len(result["rows"]) != 6 or {row["id"] for row in result["rows"]} != set(FIXES):
        raise ValueError("Expected exactly the six authorized repair IDs")
    changes = {row["id"]: row["replacements"] for row in result["rows"]}
    tokenizer = Tokenizer.from_file(str(ROOT / "data/stage1_v1/tokenizer.json"))
    revised = copy.deepcopy(originals())
    repairs = []
    for row in revised:
        if row["id"] not in changes:
            continue
        before = copy.deepcopy(row["messages"])
        replacements = changes[row["id"]]
        indices = [item["message_index"] for item in replacements]
        if sorted(indices) != sorted(FIXES[row["id"]][0]):
            raise ValueError("Writer attempted to change an unauthorized turn")
        for item in replacements:
            index, content = item["message_index"], item["content"]
            if row["messages"][index]["role"] != "assistant" or not isinstance(content, str) or not content.strip() or contains_reserved(content):
                raise ValueError("Invalid replacement content")
            row["messages"][index]["content"] = content
        if any(left != right for i, (left, right) in enumerate(zip(before, row["messages"])) if i not in indices):
            raise ValueError("Unchanged messages must remain byte-for-byte equal as text objects")
        encoded = encode_conversation(tokenizer, row["messages"], final_eos=False)
        if len(encoded["input_ids"]) > 512 or 2 in encoded["input_ids"] or encoded["labels"][-1] != 6:
            raise ValueError("Repair violates the unchanged format/context limit; no truncation")
        row["input_tokens"] = len(encoded["input_ids"])
        row["assistant_tokens"] = sum(label != -100 for label in encoded["labels"])
        row["previous_pilot_grade"] = row.pop("pilot_review_grade")
        row["previous_pilot_reason"] = row.pop("pilot_review_reason")
        row["pilot_review_grade"] = "PENDING"
        row["pilot_review_reason"] = "Awaiting independent review of the explicit repairs."
        row["repair_writer_task_id"] = raw["taskId"]
        repairs.append({"id": row["id"], "category": row["category"], "before_messages": before,
                        "messages": row["messages"], "changed_assistant_indices": indices,
                        "repair_note": FIXES[row["id"]][1], "input_tokens": row["input_tokens"],
                        "structural_passed": True})
    target = DATA / "rewrite_candidates_v2.jsonl"
    if target.exists() or (REPORT / "minor_repairs_v2.json").exists():
        raise FileExistsError("Repair output exists; refusing overwrite")
    with target.open("x", encoding="utf-8", newline="\n") as output:
        for row in revised:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_new(REPORT / "minor_repairs_v2.json", {"task_id": raw["taskId"], "actual_input_reads": 1,
              "rows": repairs, "unchanged_conversations": 58, "training_released": False,
              "quality_passed": False, "result_file": str(result_path)})
    print(json.dumps({"repaired_conversations": len(repairs), "changed_turns": 7,
                      "max_repaired_tokens": max(row["input_tokens"] for row in repairs), "quality_passed": False}))


def publish():
    review = json.loads((REPORT / "minor_repairs_review_v2.json").read_text(encoding="utf-8"))
    records = {row["id"]: row for row in review["records"]}
    if review["status"] != "PASS" or set(records) != set(FIXES) or any(
        row["grade"] != "PASS" or not row["original_minor_resolved"] for row in records.values()
    ):
        raise ValueError("The independent six-row repair review has not passed")
    rows = [json.loads(line) for line in (DATA / "rewrite_candidates_v2.jsonl").read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if row["id"] in records:
            row["pilot_review_grade"] = "PASS"
            row["pilot_review_reason"] = "；".join(turn["reason"] for turn in records[row["id"]]["assistant_turns"])
            row["repair_review_file"] = "reports/natural_sft_v4/scale20k/minor_repairs_review_v2.json"
    if len(rows) != 64 or any(row["pilot_review_grade"] != "PASS" for row in rows):
        raise ValueError("Expected 64 PASS pilot rows, with the original 58 inherited")
    with (DATA / "reviewed_pilot_v2.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_new(REPORT / "minor_repairs_release_v2.json", {
        "status": "PASS", "pilot_conversations": 64, "newly_reviewed_conversations": 6,
        "newly_reviewed_modified_turns": 7, "inherited_pass_conversations": 58,
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "assistant_tokens": sum(row["assistant_tokens"] for row in rows),
        "max_input_tokens": max(row["input_tokens"] for row in rows),
        "data_file": "data/natural_sft_v4_rewrite_pilot/reviewed_pilot_v2.jsonl",
        "training_released": False, "scale20k_ready": False,
    })
    print(json.dumps({"pilot_conversations": 64, "PASS": 64, "training_released": False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "collect", "publish"))
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "publish":
        publish()
    else:
        if args.result is None:
            parser.error("collect requires --result")
        collect(args.result)


if __name__ == "__main__":
    main()
