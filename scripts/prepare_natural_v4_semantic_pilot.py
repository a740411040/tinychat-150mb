"""Prepare a small blind packet of public-source text for the authorized Kimi reader."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/natural_sft_v4/local_capacity_probe.json"
TARGET = ROOT / "planning/train_150mb/natural_sft_v4_semantic_pilot_input.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round2", action="store_true")
    parser.add_argument("--split-round2", action="store_true")
    args = parser.parse_args()
    if args.split_round2:
        source_path = TARGET.with_name("natural_sft_v4_semantic_round2_input.txt")
        packet = json.loads(source_path.read_text(encoding="utf-8"))
        assert len(packet["records"]) == 32
        prompt_path = TARGET.with_name("natural_sft_v4_semantic_round2_prompt.txt")
        prompt = prompt_path.read_text(encoding="utf-8").replace("new 32 records", "new 8 records").replace("Exactly 32 distinct input IDs", "Exactly 8 distinct input IDs")
        short_prompt = TARGET.with_name("natural_sft_v4_semantic_round2_shard_prompt.txt")
        with short_prompt.open("x", encoding="utf-8", newline="\n") as output:
            output.write(prompt)
        for index in range(4):
            part = {**packet, "records": packet["records"][index * 8:(index + 1) * 8],
                    "batch": index + 1, "batches": 4}
            path = TARGET.with_name(f"natural_sft_v4_semantic_round2_shard{index + 1}_input.txt")
            with path.open("x", encoding="utf-8", newline="\n") as output:
                json.dump(part, output, ensure_ascii=False, indent=2)
                output.write("\n")
            print(json.dumps({"batch": index + 1, "records": len(part["records"]),
                              "bytes": path.stat().st_size, "packet": str(path)}, ensure_ascii=False))
        return
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = source["records"]
    selected = rows[18:30] + rows[48:] if args.round2 else rows[:18] + rows[30:48]
    records = [
        {"id": row["sample_id"], "source": row["source"],
         "source_row_id": row["source_row_id"],
         "messages": row["messages_at_max_512_prefix"]}
        for row in selected
    ]
    target = TARGET
    if args.round2:
        target = TARGET.with_name("natural_sft_v4_semantic_round2_input.txt")
        supplemental = json.loads((ROOT / "reports/natural_sft_v4/smoltalk_source_probe.json").read_text(encoding="utf-8"))
        for row in supplemental["records"][:12]:
            count = row["max_complete_exchanges_at_512"]
            records.append({"id": row["sample_id"], "source": "opencsg/smoltalk-chinese",
                            "source_row_id": row["source_row_index_0_based"],
                            "messages": row["original_messages"][:2 * count]})
        prior = json.loads(TARGET.read_text(encoding="utf-8"))
        assert not ({row["id"] for row in records} & {row["id"] for row in prior["records"]})
    assert len(records) == (32 if args.round2 else 36)
    assert all(row["messages"] and len(row["messages"]) % 2 == 0 for row in records)
    packet = {
        "purpose": "Blind semantic selection feasibility pilot, not training release or benchmark evaluation.",
        "data_origin": "Only publicly released MOSS/BELLE and, in round2, SmolTalk training conversations; no private project code, weights, user data, benchmark questions, or generated model evaluations.",
        "constraints": "Assess the complete original prefix starting at the first user message. Each provided prefix is within 512 tokens. Never repair answers, invent facts, skip a bad early turn, or count multiple overlapping prefixes as separate examples.",
        "categories": {
            "natural": "Ordinary situated dialogue, concrete low-risk everyday advice, empathy or clarification; exclude workbook task wrappers, generic service offers and unsupported factual claims.",
            "multi": "At least two exchanges, with the last answer materially depending on actual prior content or updated constraints; all preceding assistant answers must also be acceptable. Generic continuation alone is insufficient. Grounded text transformations may qualify but label their subtype.",
            "other": "Acceptable instruction or visible-evidence task but no qualifying natural/multi prefix.",
            "reject": "No reliably acceptable prefix; wrong answer, contradiction, failure to fulfill the request, unsupported factual or personal claims, or broken text. When uncertain, reject rather than guess.",
        },
        "records": records,
    }
    with target.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(packet, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"records": len(records), "bytes": target.stat().st_size,
                      "packet": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
