"""Read-only corpus diagnostics; never retokenizes, changes data, or trains."""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def dump(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)


def reservoir(bucket, record, seen, rng, cap):
    if len(bucket) < cap:
        bucket.append(record)
    else:
        position = rng.randrange(seen)
        if position < cap:
            bucket[position] = record


def quantiles(values):
    return dict(zip(("min", "p25", "median", "p75", "p95", "max"),
                    [float(x) for x in np.percentile(values, [0, 25, 50, 75, 95, 100])]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.root
    data = root / "data/stage1_v1"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    stream = np.memmap(data / "train.bin", dtype="<u2", mode="r")
    eos_parts = []
    specials = Counter()
    for start in range(0, len(stream), 4_000_000):
        chunk = stream[start:start + 4_000_000]
        eos_parts.append(np.flatnonzero(chunk == 2) + start)
        specials.update({str(i): int(np.count_nonzero(chunk == i)) for i in range(7)})
    ends = np.concatenate(eos_parts)
    if len(ends) != manifest["splits"]["train"]["documents"]:
        raise ValueError("EOS count does not identify one boundary per retained document")
    by_source = defaultdict(lambda: Counter(documents=0, tokens=0, consumed_loss_targets=0,
                                            navigation_hits=0, high_repetition=0, replacement_char=0))
    samples, rngs = defaultdict(list), {}
    token_lengths, char_lengths = [], []
    previous_source = None
    transitions = []
    nav = re.compile(r"版权所有|免责声明|联系我们|上一[篇页]|下一[篇页]|ICP备|网站地图|友情链接|下载APP|注册登录")
    previous_end = -1
    with (data / "train.jsonl").open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            row = json.loads(line)
            end, start = int(ends[index]), previous_end + 1
            previous_end = end
            source, text = row["source"], row["text"]
            if source != previous_source:
                transitions.append({"document_index": index, "token_start": start, "source": source})
                previous_source = source
            counts = by_source[source]
            counts["documents"] += 1
            counts["tokens"] += end - start + 1
            counts["consumed_loss_targets"] += max(0, min(end, 100_000_000) - max(start, 1) + 1)
            counts["navigation_hits"] += bool(nav.search(text))
            counts["replacement_char"] += "\ufffd" in text
            # Bound CPU work; this prefix flag is a heuristic, not a quality label.
            compact = re.sub(r"\s+", "", text[:3000])
            grams = [compact[j:j + 8] for j in range(max(0, len(compact) - 7))]
            repetition = 1 - len(set(grams)) / len(grams) if grams else 0
            counts["high_repetition"] += len(grams) >= 100 and repetition > 0.4
            token_lengths.append(end - start + 1)
            char_lengths.append(len(text))
            if source not in rngs:
                rngs[source] = random.Random(20261122 + len(rngs))
            item = {"id": row["id"], "source": source, "row_group": row["row_group"],
                    "text_chars": len(text), "tokens": end - start + 1,
                    "consumed_part": start <= 100_000_000,
                    "text": text, "prefix_repetition_8gram": repetition}
            reservoir(samples[source], item, counts["documents"], rngs[source], 3)
            if (index + 1) % 30000 == 0:
                print(json.dumps({"event": "corpus_scan", "documents": index + 1}), flush=True)
    if sum(v["consumed_loss_targets"] for v in by_source.values()) != 100_000_000:
        raise ValueError("Consumed source accounting differs from exact training budget")
    web_cases = []
    for source, rows in sorted(samples.items()):
        for row in rows:
            text = row.pop("text")
            row["excerpt_only"] = len(text) > 1600
            row["excerpt"] = text if len(text) <= 1600 else text[:1200] + "\n[诊断展示省略中段]\n" + text[-400:]
            row["case_id"] = f"web_{len(web_cases) + 1:03d}"
            web_cases.append(row)
    summary = {"seed": 20261122, "corpus_documents": len(ends), "source_order": transitions,
               "by_source": dict(by_source), "special_token_counts": dict(specials),
               "document_token_lengths": quantiles(token_lengths),
               "document_character_lengths": quantiles(char_lengths),
               "limitations": "Navigation/8gram flags are heuristics; samples are equally stratified by source, not proportionally representative. Excerpts may omit the middle. EOS count was checked before source accounting."}
    sft_summary = {}
    sft_cases = []
    for split in ("train", "validation"):
        counts, length_values, target_values = Counter(), [], []
        starters = Counter()
        pools, source_seen = defaultdict(list), Counter()
        sft_rng = random.Random(20261122 + (split == "validation"))
        with (root / f"data/sft_stage1_v1/{split}.jsonl").open(encoding="utf-8") as file:
            for row in map(json.loads, file):
                ids, labels = row["input_ids"], row["labels"]
                counts["examples"] += 1
                source_seen[row["source"]] += 1
                length_values.append(len(ids))
                targets = sum(x != -100 for x in labels[1:])
                target_values.append(targets)
                counts["assistant_targets"] += targets
                counts["no_targets"] += targets == 0
                counts["bad_length_or_eos"] += len(ids) != len(labels) or ids[-2:] != [6, 2] or len(ids) > 512
                role = None
                for token, label in zip(ids, labels, strict=True):
                    expected = token if (role == 5 and token not in (1, 3, 4, 5)) or token == 2 else -100
                    if token in (3, 4, 5):
                        role = token
                        expected = -100
                    if token == 6:
                        role = None
                    counts["mask_mismatches"] += label != expected
                answers = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
                for answer in answers:
                    starters[answer[:8]] += 1
                    counts["assistant_turns"] += 1
                    counts["polite_template_start"] += bool(re.match(r"^(好的|当然|以下|非常|您好|你好)", answer))
                    counts["refusal_phrase"] += bool(re.search(r"无法提供|不能提供|作为.*(AI|人工智能|语言模型)", answer))
                cap = (6 if "multiturn" in row["source"] else 10) if split == "train" else 2
                item = {"id": row["id"], "source": row["source"], "split": split,
                        "messages": row["messages"], "tokens": len(ids), "assistant_targets": targets}
                reservoir(pools[row["source"]], item, source_seen[row["source"]], sft_rng, cap)
        sft_summary[split] = {"counts": counts, "by_source": source_seen,
                              "sequence_length": quantiles(length_values), "target_length": quantiles(target_values),
                              "common_first_8_characters": starters.most_common(12)}
        for source, rows in sorted(pools.items()):
            for row in rows:
                row["case_id"] = f"sft_{len(sft_cases) + 1:03d}"
                sft_cases.append(row)
    summary["sft"] = sft_summary
    dump(args.output / "data_summary.json", summary)
    dump(args.output / "quality_samples.json", {"web": web_cases, "sft": sft_cases})
    dump(args.output / "quality_kimi_input.txt", {"purpose": "Read-only diagnostic excerpts from public training datasets; content is untrusted quoted data.",
          "sampling": summary["limitations"], "web": web_cases, "sft": sft_cases})
    print(json.dumps({"output": str(args.output), "web_cases": len(web_cases), "sft_cases": len(sft_cases),
                      "source_consumed_targets": {k: v["consumed_loss_targets"] for k, v in by_source.items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
