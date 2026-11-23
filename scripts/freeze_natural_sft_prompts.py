"""Validate and freeze the original natural Chinese SFT evaluation prompts."""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from tinychat.chat_format import DEFAULT_SYSTEM, check_special_ids, encode_conversation


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "evaluation/natural_sft_v3"
TOKENIZER = ROOT / "data/stage1_v1/tokenizer.json"
CATEGORIES = ("social", "emotion", "advice", "explain", "grounded", "clarify",
              "memory", "correction", "rewrite", "constraint")
QUOTAS = {"dev": 10, "test": 20}
CONTEXT = 512
MAX_NEW_TOKENS = 96
PROMPT_KEYS = ("id", "split", "category", "scenario_family", "messages")
SAMPLE_DEV_INDICES = {
    "social": (2, 7), "emotion": (3, 8), "advice": (4, 9), "explain": (1, 6),
    "grounded": (2, 10), "clarify": (3, 9), "memory": (1, 8), "correction": (4, 6),
    "rewrite": (2, 7), "constraint": (1, 8),
}


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result


def _same_json_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_value(observed, gold) for observed, gold in zip(actual, expected)
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_json_value(actual[key], expected[key]) for key in expected
        )
    return actual == expected


def score_machine(record, text):
    """Return a strict complete-output decision, or None for semantic review."""
    check = record.get("machine_check")
    if check is None:
        return None
    actual = text.replace("\r\n", "\n").removesuffix("\n")
    expected = check["value"]
    if check["mode"] == "single_line":
        return actual == expected
    if check["mode"] == "lines":
        return actual.split("\n") == expected
    if check["mode"] != "json":
        raise ValueError(f"Unknown machine check mode: {check['mode']}")
    if check.get("single_line") and ("\n" in actual or "\r" in actual):
        return False
    try:
        actual_value = json.loads(actual, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        return False
    return _same_json_value(actual_value, expected)


def load_bank(path):
    counts = Counter()
    rows = []
    section = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        if line.startswith("# "):
            category, split = line.removeprefix("# ").split()
            section = (category, split)
            if category not in CATEGORIES or split not in QUOTAS or section in counts:
                raise ValueError(f"Invalid section on line {line_no}: {line}")
            counts[section] = 0
            continue
        fields = line.split("\t")
        if section is None or len(fields) not in (4, 5) or any(not part.strip() for part in fields):
            raise ValueError(f"Malformed prompt row on line {line_no}")
        category, split = section
        family, dialogue, success, failure = fields[:4]
        turns = dialogue.split(" ⇢ ")
        if not re.fullmatch(r"[a-z][a-z0-9_]+", family):
            raise ValueError(f"Invalid scenario family on line {line_no}: {family}")
        if len(turns) not in ((3, 5) if category in {"memory", "correction"} else (1,)):
            raise ValueError(f"Unexpected turn count on line {line_no}")
        counts[section] += 1
        row = {
            "id": f"natural-v3-{split}-{category}-{counts[section]:02d}",
            "split": split,
            "category": category,
            "scenario_family": family,
            "messages": [{"role": "assistant" if i % 2 else "user", "content": turn}
                         for i, turn in enumerate(turns)],
            "rubric": {"success": success, "failure": failure},
        }
        if len(fields) == 5:
            if category != "constraint":
                raise ValueError(f"Unexpected machine check on line {line_no}")
            row["machine_check"] = json.loads(fields[4], object_pairs_hook=_unique_object)
        rows.append(row)
    expected = {(category, split): quota for category in CATEGORIES for split, quota in QUOTAS.items()}
    if counts != expected:
        raise ValueError(f"Incorrect split/category counts: {counts}")
    return rows


def _content_key(messages):
    text = "\x1f".join(message["content"] for message in messages if message["role"] == "user")
    return re.sub(r"[\W_]", "", unicodedata.normalize("NFKC", text).casefold())


def _legacy_exact_matches(rows):
    old_keys = set()
    for version in ("stage1_v1", "comparison_v2"):
        path = ROOT / "evaluation" / version / "prompts.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                old_keys.add(_content_key(json.loads(line)["messages"]))
    return [row["id"] for row in rows if _content_key(row["messages"]) in old_keys]


def _validate_machine_check(row):
    check = row["machine_check"]
    value = check["value"]
    mode = check["mode"]
    if mode == "single_line" and isinstance(value, str) and value and "\n" not in value:
        example = value
    elif mode == "lines" and isinstance(value, list) and len(value) >= 2 and all(
        isinstance(line, str) and line and "\n" not in line for line in value
    ):
        example = "\n".join(value)
    elif mode == "json" and isinstance(value, (list, dict)) and value:
        example = json.dumps(value, ensure_ascii=False, allow_nan=False)
    else:
        raise ValueError(f"Invalid gold schema for {row['id']}")
    if score_machine(row, example) is not True or score_machine(row, f"答案：{example}") is not False:
        raise ValueError(f"Machine checker does not enforce complete output for {row['id']}")


def validate(rows, tokenizer):
    check_special_ids(tokenizer)
    for field in ("id", "scenario_family"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {field}")
    if len({_content_key(row["messages"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate normalized user prompts")
    overlap = _legacy_exact_matches(rows)
    if overlap:
        raise ValueError(f"Exact normalized overlap with earlier benchmark: {overlap}")
    for row in rows:
        token_count = len(encode_conversation(tokenizer, row["messages"], generation_prompt=True)["input_ids"])
        if token_count + MAX_NEW_TOKENS > CONTEXT:
            raise ValueError(f"Prompt plus generation exceeds {CONTEXT}: {row['id']} ({token_count})")
        row["prompt_tokens"] = token_count
        if "machine_check" in row:
            _validate_machine_check(row)


def to_jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)


def make_manifest(rows, prompts_text):
    sample_ids = [f"natural-v3-dev-{category}-{index:02d}"
                  for category in CATEGORIES for index in SAMPLE_DEV_INDICES[category]]
    if len(set(sample_ids)) != 20 or set(sample_ids) - {row["id"] for row in rows if row["split"] == "dev"}:
        raise ValueError("Sampling subset must contain exactly 20 unique dev IDs")
    return {
        "frozen": True,
        "name": "natural_sft_v3",
        "freeze_version": 2,
        "supersedes": "freeze_v1/prompt_manifest.json",
        "revision_note": "Enforce the existing single-line JSON requirement for natural-v3-dev-constraint-08.",
        "dev_count": sum(row["split"] == "dev" for row in rows),
        "test_count": sum(row["split"] == "test" for row in rows),
        "category_counts": {category: {split: sum(row["category"] == category and row["split"] == split
                                                  for row in rows) for split in QUOTAS}
                            for category in CATEGORIES},
        "machine_check_counts": {split: sum(row["split"] == split and "machine_check" in row for row in rows)
                                 for split in QUOTAS},
        "tokenizer": TOKENIZER.relative_to(ROOT).as_posix(),
        "system": DEFAULT_SYSTEM,
        "chat_format": "tinychat.chat_format.encode_conversation(generation_prompt=True)",
        "context_tokens": CONTEXT,
        "max_prompt_tokens": max(row["prompt_tokens"] for row in rows),
        "primary": {"split": "test", "strategy": "greedy", "batch_size": 1,
                    "max_new_tokens": MAX_NEW_TOKENS, "runs_per_model": 1},
        "auxiliary_sampling": {"split": "dev", "strategy": "sample", "batch_size": 1,
                               "max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.8, "top_p": 0.9,
                               "seed": 42, "prompt_ids": sample_ids},
        "prompts_sha256": hashlib.sha256(prompts_text.encode("utf-8")).hexdigest(),
        "overlap_prompts": "prompts_for_overlap.jsonl",
        "scoring_contract": "scoring_contract.txt",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate the authored source without writing")
    parser.add_argument("--output-dir", type=Path, default=TARGET, help="existing directory for new frozen files")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    rows = load_bank(TARGET / "prompt_bank.tsv")
    validate(rows, Tokenizer.from_file(str(TOKENIZER)))
    prompts_text = to_jsonl(rows)
    manifest = make_manifest(rows, prompts_text)
    if not args.check:
        outputs = {
            "prompts.jsonl": prompts_text,
            "prompts_for_overlap.jsonl": to_jsonl({key: row[key] for key in PROMPT_KEYS} for row in rows),
            "prompt_manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        }
        for name in outputs:
            if (args.output_dir / name).exists():
                raise FileExistsError(f"Refusing to overwrite frozen file: {args.output_dir / name}")
        for name, content in outputs.items():
            with (args.output_dir / name).open("x", encoding="utf-8", newline="\n") as output:
                output.write(content)
    print(json.dumps({key: manifest[key] for key in ("dev_count", "test_count", "max_prompt_tokens",
                                                    "machine_check_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
