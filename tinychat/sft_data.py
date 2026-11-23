"""Build research-only Stage 1 SFT examples against the frozen base tokenizer."""

import argparse
from collections import Counter
from contextlib import closing
import io
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from tokenizers import Tokenizer

from .chat_format import IGNORE_LABEL, SPECIAL_IDS, check_special_ids, contains_reserved, encode_conversation
from .stage1_data import (
    CONTROL, DEFAULT_SOURCES, HAN, MIB, REPEATED_CHARACTER, SPLITS, build_lock, eval_matches,
    eval_prompt_keys, file_identity, heartbeat, json_line, normalized_content, sha256, source_file,
    stable_bucket, write_json,
)


QUOTAS = {"single": {"train": 18_000, "validation": 450, "test": 450},
          "multi": {"train": 2_000, "validation": 50, "test": 50}}
SOURCE_KINDS = {"BelleGroup/train_0.5M_CN": "single", "BelleGroup/multiturn_chat_0.8M": "multi"}
SEED = 20261121
MAX_TOKENS = 512
MAX_JSON_RECORD = 4 * MIB
ROLE_LINE = re.compile(r"(?m)^[ \t]*(Human|Assistant)[ \t]*[:：][ \t]*")
REFUSAL = re.compile(
    r"^(?:很抱歉|抱歉|对不起|不好意思)[\s，,。:：]*(?:作为|我(?:不能|无法|没有|不支持|不被允许))"
    r"|^(?:作为(?:一个|一名)?(?:人工智能|AI|语言模型)|as an ai\b)",
    re.IGNORECASE,
)
WRONG_IDENTITY = re.compile(
    r"(?:我(?:是|叫|的名字是)[^\n。！？]{0,32}(?:ChatGPT|OpenAI|BELLE|百度|文心一言|微软小冰)"
    r"|我(?:是)?由[^\n。！？]{0,32}(?:开发|训练|创建))", re.IGNORECASE,
)
REPEATED_PHRASE = re.compile(r"(.{8,40}?)\1{2,}")


def iter_json_array(stream):
    decoder = json.JSONDecoder()
    buffer, offset = "", 0

    def refill():
        nonlocal buffer, offset
        chunk = stream.read(MIB)
        buffer = buffer[offset:] + chunk
        offset = 0
        return bool(chunk)

    def next_char():
        nonlocal offset
        while True:
            while offset < len(buffer) and buffer[offset].isspace():
                offset += 1
            if offset < len(buffer):
                return buffer[offset]
            if not refill():
                return ""

    if next_char() == "]":
        offset += 1
        if next_char():
            raise ValueError("Unexpected data after source JSON array")
        return
    while True:
        if not next_char():
            raise ValueError("Unterminated source JSON array")
        while True:
            try:
                row, end = decoder.raw_decode(buffer, offset)
                break
            except json.JSONDecodeError:
                if len(buffer) - offset >= MAX_JSON_RECORD or not refill():
                    raise
        yield row
        offset = end
        marker = next_char()
        if marker == "]":
            offset += 1
            if next_char():
                raise ValueError("Unexpected data after source JSON array")
            return
        if marker != ",":
            raise ValueError("Expected comma or closing bracket in source JSON array")
        offset += 1


def read_json_records(stream):
    """Read JSON Lines or a JSON array from a bounded streaming reader."""
    with io.TextIOWrapper(stream, encoding="utf-8-sig") as text:
        first = text.read(1)
        while first and first.isspace():
            first = text.read(1)
        if first == "[":
            yield from iter_json_array(text)
            return
        while first:
            line = first + text.readline(MAX_JSON_RECORD + 1)
            if len(line) > MAX_JSON_RECORD:
                raise ValueError("Oversize source JSON record")
            if line.strip():
                yield json.loads(line)
            first = text.read(1)


def clean_turn(text):
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return CONTROL.sub("", text).strip()


def parse_conversation(row, kind):
    if not isinstance(row, dict):
        return None
    instruction, context, answer = (row.get(field, "") for field in ("instruction", "input", "output"))
    if not all(isinstance(text, str) for text in (instruction, context, answer)) or not answer.strip():
        return None
    if kind == "single":
        if ROLE_LINE.search(instruction):
            return None
        user = "\n\n".join(text.strip() for text in (instruction, context) if text.strip())
        parts = [("user", user), ("assistant", answer)]
    else:
        instruction = instruction.replace("\r\n", "\n").replace("\r", "\n")
        marks = list(ROLE_LINE.finditer(instruction))
        if len(marks) < 4 or len(marks) % 2 or instruction[:marks[0].start()].strip() or context.strip():
            return None
        if [mark.group(1) for mark in marks] != ["Human", "Assistant"] * (len(marks) // 2):
            return None
        if instruction[marks[-1].end():].strip():
            return None
        parts = [("user" if mark.group(1) == "Human" else "assistant", instruction[mark.end():nxt.start()])
                 for mark, nxt in zip(marks, marks[1:])]
        parts.append(("assistant", answer))
    messages = [{"role": role, "content": clean_turn(text)} for role, text in parts]
    if any(not item["content"] or contains_reserved(item["content"]) for item in messages):
        return None
    return messages


def conversation_key(messages):
    normalized = tuple((item["role"], normalized_content(item["content"])) for item in messages)
    return sha256(json_line(normalized).encode("utf-8"))


def conversation_split(messages, seed=SEED):
    # BELLE may include several prefixes of the same history without a shared ID.
    group = conversation_key(messages[:2])
    bucket = stable_bucket(group, seed)
    return SPLITS[(bucket >= 9000) + (bucket >= 9500)], group


def quality_reason(messages):
    seen = set()
    for item in messages:
        text, role = item["content"], item["role"]
        normalized = normalized_content(text)
        han = len(HAN.findall(text))
        if han < (2 if role == "assistant" else 1) or han < 0.35 * len(normalized):
            return "insufficient_chinese"
        if REPEATED_CHARACTER.search(text) or REPEATED_PHRASE.search(normalized):
            return "repeated_or_junk"
        identity = (role, normalized)
        if len(normalized) >= 4 and identity in seen:
            return "repeated_or_junk"
        seen.add(identity)
        if role == "assistant" and REFUSAL.search(text):
            return "assistant_refusal_boilerplate"
        if role == "assistant" and WRONG_IDENTITY.search(text):
            return "wrong_assistant_identity"
    return None


def select_window(tokenizer, messages, kind, max_tokens=MAX_TOKENS):
    lengths = (2,) if kind == "single" else (6, 4)
    reason = "no_complete_window"
    for length in lengths:
        if len(messages) < length:
            continue
        window = messages[-length:]
        reason = quality_reason(window)
        if reason:
            continue
        encoded = encode_conversation(tokenizer, window)
        if len(encoded["input_ids"]) <= max_tokens:
            return (window, encoded), None
        reason = "over_context_length"
    return None, reason


def init_journal(db, request):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS request (payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS progress (
            source TEXT NOT NULL, file TEXT NOT NULL, rows INTEGER NOT NULL, counts TEXT NOT NULL,
            PRIMARY KEY (source, file)
        );
        CREATE TABLE IF NOT EXISTS examples (
            ordinal INTEGER PRIMARY KEY, original_key TEXT NOT NULL UNIQUE, window_key TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL, split TEXT NOT NULL, payload TEXT NOT NULL,
            input_tokens INTEGER NOT NULL, assistant_tokens INTEGER NOT NULL
        );
    """)
    payload = json.dumps(request, ensure_ascii=False, sort_keys=True)
    existing = db.execute("SELECT payload FROM request").fetchone()
    if existing and existing[0] != payload:
        raise ValueError("The SFT journal has different sources, tokenizer, screening or settings; use a new output directory")
    if not existing:
        with db:
            db.execute("INSERT INTO request VALUES (?)", (payload,))


def acquire_source(db, source, kind, quota, tokenizer, matches_eval, report, local_root):
    kept = dict(db.execute("SELECT split, count(*) FROM examples WHERE kind = ? GROUP BY split", (kind,)))
    remaining = {split: target - kept.get(split, 0) for split, target in quota.items()}
    source_name = source["repo"]
    for entry in source["files"]:
        if not any(remaining.values()):
            return
        entry_path = entry["path"]
        previous = db.execute("SELECT rows, counts FROM progress WHERE source = ? AND file = ?",
                              (source_name, entry_path)).fetchone()
        skip_rows, counts = previous if previous else (0, "{}")
        counts = Counter(json.loads(counts))
        counts.setdefault("evaluation_exact_overlap", 0)
        counts.setdefault("evaluation_substring_overlap", 0)
        report(f"[SFT acquire] {source_name}/{entry_path}; resuming after row {skip_rows:,}; remaining {remaining}")
        with source_file(source, entry, local_root=local_root) as stream, closing(read_json_records(stream)) as records:
            for ordinal, row in enumerate(records, start=1):
                if ordinal <= skip_rows:
                    continue
                reason = keep_example(db, tokenizer, kind, source_name, row, remaining, matches_eval)
                counts[reason] += 1
                if ordinal % 256 == 0 or not any(remaining.values()):
                    db.execute("INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?)",
                               (source_name, entry_path, ordinal, json.dumps(counts)))
                    db.commit()
                if ordinal % 8192 == 0:
                    report(f"[SFT acquire] {source_name} scanned {ordinal:,}; remaining {remaining}")
                if not any(remaining.values()):
                    report(f"[SFT acquire] {source_name} complete after {ordinal:,} rows")
                    return
            db.execute("INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?)",
                       (source_name, entry_path, sum(counts.values()), json.dumps(counts)))
            db.commit()
    raise RuntimeError(f"{source_name} exhausted with quotas unfilled: {remaining}")


def keep_example(db, tokenizer, kind, source_name, row, remaining, matches_eval):
    messages = parse_conversation(row, kind)
    if messages is None:
        return "invalid_conversation"
    for item in messages:
        if item["role"] == "user" and (reason := matches_eval(normalized_content(item["content"]))):
            return reason
    split, group = conversation_split(messages)
    if not remaining[split]:
        return "split_quota_full"
    original = conversation_key(messages)
    if db.execute("SELECT 1 FROM examples WHERE original_key = ?", (original,)).fetchone():
        return "original_duplicate"
    selected, reason = select_window(tokenizer, messages, kind)
    if selected is None:
        return reason
    window, encoded = selected
    key = conversation_key(window)
    if db.execute("SELECT 1 FROM examples WHERE window_key = ?", (key,)).fetchone():
        return "window_duplicate"
    payload = {"id": original, "messages": window, **encoded, "source": source_name, "group_id": group}
    target_tokens = sum(label != IGNORE_LABEL for label in encoded["labels"])
    db.execute("""
        INSERT INTO examples(original_key, window_key, kind, split, payload, input_tokens, assistant_tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (original, key, kind, split, json_line(payload), len(encoded["input_ids"]), target_tokens))
    remaining[split] -= 1
    return "accepted"


def publish_splits(db, out):
    stats = {}
    for split in SPLITS:
        stat = {"documents": 0, "tokens": 0, "assistant_tokens": 0, "single": 0, "multi": 0}
        with (out / f"{split}.jsonl.partial").open("w", encoding="utf-8", newline="\n") as target:
            for kind, payload, tokens, assistant in db.execute("""
                SELECT kind, payload, input_tokens, assistant_tokens FROM examples WHERE split = ? ORDER BY ordinal
            """, (split,)):
                target.write(payload)
                stat[kind] += 1
                stat["documents"] += 1
                stat["tokens"] += tokens
                stat["assistant_tokens"] += assistant
        stats[split] = stat
    return stats


def build_sft(out, source_entries, pretrain_dir, eval_prompts, *, local_root=None, quotas=QUOTAS):
    base = json.loads((pretrain_dir / "manifest.json").read_text(encoding="utf-8"))
    tokenizer_path = Path(base["tokenizer_path"])
    tokenizer_identity = file_identity(tokenizer_path)
    if (base["schema_version"] != 1 or base["frozen"] is not True or base["corpus_id"] != "stage1_v1"
            or base["special_token_ids"] != SPECIAL_IDS or tokenizer_identity != base["files"]["tokenizer.json"]):
        raise ValueError("The Stage 1 tokenizer does not match its frozen pretraining manifest")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    check_special_ids(tokenizer)
    if tokenizer.get_vocab_size() != base["vocab_size"]:
        raise ValueError("The Stage 1 tokenizer vocabulary size changed")

    keys, eval_identity = eval_prompt_keys(eval_prompts)
    base_screen = base.get("evaluation_screen")
    if base_screen and base_screen["sha256"] != eval_identity["sha256"]:
        raise ValueError("Pretraining and SFT evaluation prompts differ")
    screening = eval_matches(keys)
    with build_lock(out), heartbeat("[SFT] opening acquisition journal") as report, \
            closing(sqlite3.connect(out / "acquisition.sqlite3")) as db:
        request = {"sources": source_entries, "quotas": quotas, "evaluation_screen": eval_identity,
                   "tokenizer_sha256": base["tokenizer_sha256"], "seed": SEED, "max_tokens": MAX_TOKENS}
        init_journal(db, request)
        for source in source_entries:
            kind = SOURCE_KINDS[source["repo"]]
            acquire_source(db, source, kind, quotas[kind], tokenizer, screening, report, local_root)
        stats = publish_splits(db, out)
        if any(stats[split][kind] != quotas[kind][split] for kind in quotas for split in SPLITS):
            raise RuntimeError("SFT split counts differ from the requested quotas")

        names = [f"{split}.jsonl" for split in SPLITS]
        for name in names:
            (out / f"{name}.partial").replace(out / name)
        files = {name: file_identity(out / name) for name in names}
        files["tokenizer.json"] = tokenizer_identity
        rows = db.execute("SELECT source, file, rows, counts FROM progress ORDER BY source, file")
        manifest = {**request, "schema_version": 1, "corpus_id": "sft_stage1_v1", "frozen": True,
                    "tokenizer_path": str(tokenizer_path.resolve()), "vocab_size": base["vocab_size"],
                    "special_token_ids": SPECIAL_IDS, "splits": stats, "files": files,
                    "counts": {split: stats[split]["documents"] for split in SPLITS},
                    "filter_stats": [{"source": repo, "file": name, "rows_scanned": scanned,
                                      "counts": json.loads(counts)} for repo, name, scanned, counts in rows],
                    "usage": "Research only under the publisher's usage restrictions; generated data may contain errors.",
                    "split": "90/5/5 seeded hash of the normalized first user/assistant exchange before windowing; "
                             "BELLE has no conversation identifier, so later-only overlapping histories may remain."}
        write_json(out / "manifest.json", manifest)
        report(f"[SFT] froze {out / 'manifest.json'} with counts {manifest['counts']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--pretrain-dir", type=Path, default=Path("data/stage1_v1"))
    parser.add_argument("--eval-prompts", type=Path, required=True, help="Frozen evaluation prompts JSONL")
    parser.add_argument("--out-dir", type=Path, default=Path("data/sft_stage1_v1"))
    parser.add_argument("--local-root", type=Path, help="Read exact-size pinned files from this directory instead of the Hub")
    args = parser.parse_args()
    sources = json.loads(args.source_manifest.read_text(encoding="utf-8"))["sft"]
    if len(sources) != len(SOURCE_KINDS) or {source["repo"] for source in sources} != SOURCE_KINDS.keys():
        parser.error("Stage 1 SFT requires the two pinned BELLE sources")
    build_sft(args.out_dir, sources, args.pretrain_dir, args.eval_prompts, local_root=args.local_root)


if __name__ == "__main__":
    main()
