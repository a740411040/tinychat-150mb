"""Acquire, tokenize, and freeze the Stage 1 Chinese pretraining corpus."""

import argparse
from collections import Counter
from contextlib import closing, contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import threading
import unicodedata

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .chat_format import SPECIALS, SPECIAL_IDS, check_special_ids, contains_reserved


SPLITS = ("train", "validation", "test")
MIB = 1024 * 1024
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]")
NOT_ALNUM = re.compile(r"[\W_]+")
SPACES = re.compile(r"[^\S\n]+")
REPEATED_CHARACTER = re.compile(r"([^\s])\1{31,}")
EVAL_SUBSTRING_CHARS = 24
DEFAULT_SOURCES = Path("planning/train_150mb/stage1_sources.json")


@dataclass(frozen=True)
class CorpusSettings:
    shard_bytes: int = 64 * MIB
    tokenizer_bytes_per_shard: int = 16 * MIB
    vocab_size: int = 16384
    min_train_tokens: int = 100_000_001
    min_han: int = 48
    min_han_fraction: float = 0.35
    max_document_bytes: int = 128 * 1024
    batch_size: int = 128
    seed: int = 20261121


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_line(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_json(path, value):
    pending = path.with_name(path.name + ".partial")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    pending.replace(path)


def file_identity(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@contextmanager
def build_lock(out):
    """Keep a single writer; the OS releases the lock if the process exits."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".build.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process is building {out}") from exc
        if (out / "manifest.json").exists():
            raise RuntimeError(f"A frozen manifest already exists at {out / 'manifest.json'}")
        yield


@contextmanager
def heartbeat(initial_status):
    status = initial_status
    stop = threading.Event()

    def report(message):
        nonlocal status
        status = message
        print(message, flush=True)

    def pulse():
        while not stop.wait(20):
            print(f"[heartbeat] {status}", flush=True)

    report(status)
    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        yield report
    finally:
        stop.set()
        thread.join()


def clean_text(text):
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = CONTROL.sub("", text)
    return "\n".join(line for part in text.split("\n") if (line := SPACES.sub(" ", part).strip()))


def normalized_content(text):
    return NOT_ALNUM.sub("", unicodedata.normalize("NFKC", text).casefold())


def eval_prompt_keys(path):
    if path is None:
        return frozenset(), None
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    keys = frozenset(
        key for row in rows for message in row["messages"]
        if message["role"] == "user" and (key := normalized_content(message["content"]))
    )
    if not keys:
        raise ValueError(f"No normalized user prompts in {path}")
    return keys, {"path": str(path.resolve()), "sha256": sha256(data), "user_keys": len(keys),
                  "substring_min_chars": EVAL_SUBSTRING_CHARS}


def eval_matches(keys):
    longer_keys = tuple(sorted((key for key in keys if len(key) >= EVAL_SUBSTRING_CHARS), key=len, reverse=True))

    def match(text):
        if text in keys:
            return "evaluation_exact_overlap"
        if any(key in text for key in longer_keys):
            return "evaluation_substring_overlap"
        return None

    return match


def stable_bucket(key, seed):
    value = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(value, "big") % 10000


def document_split(key, seed):
    bucket = stable_bucket(key, seed)
    return SPLITS[(bucket >= 9800) + (bucket >= 9900)]


@contextmanager
def source_file(source, entry, *, local_root=None):
    if local_root is not None:
        path = local_root / entry["path"]
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Local source size differs from the pinned manifest for {path}")
        with path.open("rb") as stream:
            yield stream
        return

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=False)
    path = f'datasets/{source["repo"]}@{source["revision"]}/{entry["path"]}'
    with fs.open(path, "rb", block_size=MIB, cache_type="readahead") as stream:
        if stream.size != entry["bytes"]:
            raise ValueError(f"Remote source size differs from the pinned manifest for {path}")
        yield stream


def filter_document(raw, settings):
    if not isinstance(raw, str) or len(raw) > settings.max_document_bytes:
        return "invalid_or_oversize", None, 0
    text = clean_text(raw)
    size = len(text.encode("utf-8"))
    if size > settings.max_document_bytes:
        return "invalid_or_oversize", None, 0
    if contains_reserved(text):
        return "reserved_tokens", None, 0
    if REPEATED_CHARACTER.search(text):
        return "repeated_characters", None, 0
    han = len(HAN.findall(text))
    letters = len(NOT_ALNUM.sub("", text))
    if han < settings.min_han or han < settings.min_han_fraction * letters:
        return "insufficient_chinese", None, 0
    return None, text, size


def init_journal(db, request):
    db.execute("PRAGMA temp_store = FILE")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS request (payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY, exact TEXT NOT NULL, near TEXT NOT NULL UNIQUE,
            shard INTEGER NOT NULL, row_group INTEGER NOT NULL, row_in_group INTEGER NOT NULL,
            split TEXT NOT NULL, sample_rank BLOB NOT NULL, text TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS row_groups (
            shard INTEGER NOT NULL, ordinal INTEGER NOT NULL, row_group INTEGER NOT NULL,
            rows INTEGER NOT NULL, bytes INTEGER NOT NULL, finished INTEGER NOT NULL,
            counts TEXT NOT NULL, PRIMARY KEY (shard, ordinal)
        );
    """)
    payload = json.dumps(request, ensure_ascii=False, sort_keys=True)
    existing = db.execute("SELECT payload FROM request").fetchone()
    if existing and existing[0] != payload:
        raise ValueError("The acquisition journal has different sources, screening or settings; use a new output directory")
    if not existing:
        with db:
            db.execute("INSERT INTO request VALUES (?)", (payload,))


def ingest_batch(db, batch, shard_idx, group, rows, settings, matches_eval, remaining_bytes):
    counts = Counter()
    accepted_bytes, scanned = 0, 0
    for offset, raw in enumerate(batch):
        scanned += 1
        reason, text, size = filter_document(raw, settings)
        if reason:
            counts[reason] += 1
            continue
        canonical = normalized_content(text)
        if reason := matches_eval(canonical):
            counts[reason] += 1
            continue
        exact = sha256(text.encode("utf-8"))
        near = sha256(canonical.encode("utf-8"))
        previous = db.execute("SELECT exact FROM documents WHERE near = ?", (near,)).fetchone()
        if previous:
            counts["exact_duplicate" if previous[0] == exact else "normalized_duplicate"] += 1
            continue
        rank = hashlib.blake2b(f"sample:{settings.seed}:{near}".encode("utf-8"), digest_size=8).digest()
        db.execute("""
            INSERT INTO documents(exact, near, shard, row_group, row_in_group, split, sample_rank, text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (exact, near, shard_idx, group, rows + offset, document_split(near, settings.seed), rank, text))
        accepted_bytes += size
        counts["accepted"] += 1
        if accepted_bytes >= remaining_bytes:
            break
    return accepted_bytes, scanned, counts


def acquire_shard(db, source, entry, shard_idx, settings, matches_eval, report, local_root):
    import pyarrow.parquet as pq

    previous = db.execute("""
        SELECT ordinal, row_group, rows, bytes, finished, counts FROM row_groups WHERE shard = ? ORDER BY ordinal
    """, (shard_idx,)).fetchall()
    accepted = sum(row[3] for row in previous)
    if accepted >= settings.shard_bytes:
        report(f"[acquire] {entry['path']} already has {accepted:,} accepted bytes")
        return

    report(f"[acquire] opening {entry['path']} at {accepted:,} accepted bytes")
    with source_file(source, entry, local_root=local_root) as stream:
        parquet = pq.ParquetFile(stream, buffer_size=MIB, pre_buffer=False)
        groups = list(range(parquet.num_row_groups))
        random.Random(settings.seed + shard_idx).shuffle(groups)
        for ordinal, group in enumerate(groups):
            rows, group_bytes, counts = 0, 0, Counter()
            if ordinal < len(previous):
                _, old_group, rows, group_bytes, finished, old_counts = previous[ordinal]
                if old_group != group:
                    raise ValueError(f"Row-group order changed for {entry['path']}")
                if finished:
                    continue
                counts.update(json.loads(old_counts))
            rows_to_skip = rows
            total_rows = parquet.metadata.row_group(group).num_rows
            report(f"[acquire] {entry['path']} row group {group} at row {rows:,}/{total_rows:,}; {accepted:,} bytes total")
            for batch in parquet.iter_batches(row_groups=[group], columns=["text"], batch_size=settings.batch_size):
                if rows_to_skip >= len(batch):
                    rows_to_skip -= len(batch)
                    continue
                values = batch.column(0).slice(rows_to_skip).to_pylist()
                rows_to_skip = 0
                with db:
                    size, scanned, batch_counts = ingest_batch(
                        db, values, shard_idx, group, rows, settings, matches_eval, settings.shard_bytes - accepted
                    )
                    rows += scanned
                    group_bytes += size
                    counts.update(batch_counts)
                    db.execute("INSERT OR REPLACE INTO row_groups VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (shard_idx, ordinal, group, rows, group_bytes, rows == total_rows, json.dumps(counts)))
                accepted += size
                if accepted >= settings.shard_bytes:
                    report(f"[acquire] {entry['path']} complete at {accepted:,} accepted bytes")
                    return
            if rows != total_rows:
                raise RuntimeError(f"Incomplete Parquet row group {group} in {entry['path']}")
    raise RuntimeError(f"{entry['path']} exhausted at {accepted:,} bytes; target is {settings.shard_bytes:,}")


def tokenizer_sample(path, shard_count, per_shard_bytes, sample_counts):
    # Tokenizers pulls the iterator from its worker thread. This reader belongs to that iterator alone.
    with closing(sqlite3.connect(path, check_same_thread=False)) as db:
        db.execute("PRAGMA query_only = ON")
        for shard in range(shard_count):
            remaining = per_shard_bytes
            batch = []
            for (text,) in db.execute("""
                SELECT text FROM documents WHERE shard = ? AND split = 'train' ORDER BY sample_rank, id
            """, (shard,)):
                item = text.encode("utf-8")
                if len(item) >= remaining:
                    text = item[:remaining].decode("utf-8", errors="ignore")
                    sample_counts[shard] = per_shard_bytes - remaining + len(text.encode("utf-8"))
                    if text:
                        batch.append(text)
                    yield batch
                    break
                batch.append(text)
                remaining -= len(item)
                if len(batch) == 64:
                    yield batch
                    batch = []
            else:
                raise RuntimeError(f"Shard {shard} has insufficient TRAIN text for the tokenizer sample")


def train_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=list(SPECIALS),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    check_special_ids(tokenizer)
    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(f"Tokenizer generated {tokenizer.get_vocab_size()} tokens, expected {vocab_size}")
    probe = "中文标点、English!\n保留空格 🧪𠀀"
    if tokenizer.decode(tokenizer.encode(probe).ids) != probe:
        raise RuntimeError("ByteLevel round-trip failed")
    return tokenizer


def write_tokenized_splits(db, source_entries, out, tokenizer, report):
    stats = {}
    eos = SPECIAL_IDS["<|eos|>"]
    for split in SPLITS:
        docs, tokens = 0, 0
        report(f"[tokenize] writing {split}")
        cursor = db.execute("""
            SELECT exact, text, shard, row_group, row_in_group FROM documents WHERE split = ? ORDER BY id
        """, (split,))
        with (out / f"{split}.jsonl.partial").open("w", encoding="utf-8", newline="\n") as retained, \
                (out / f"{split}.bin.partial").open("wb") as binary:
            while batch := cursor.fetchmany(64):
                encoded = tokenizer.encode_batch([row[1] for row in batch], add_special_tokens=False)
                for (record_id, text, shard, group, row), encoding in zip(batch, encoded, strict=True):
                    retained.write(json_line({"id": record_id, "text": text, "source": source_entries[shard]["path"],
                                              "row_group": group, "row_in_group": row}))
                    binary.write(np.asarray([*encoding.ids, eos], dtype="<u2").tobytes())
                    tokens += len(encoding.ids) + 1
                    docs += 1
                if docs % 4096 == 0:
                    report(f"[tokenize] {split}: {docs:,} documents, {tokens:,} tokens")
        stats[split] = {"documents": docs, "tokens": tokens, "binary_bytes": tokens * 2}
        report(f"[tokenize] {split} complete: {docs:,} documents, {tokens:,} tokens")
    return stats


def source_stats(db, entries, sample_counts):
    for shard, entry in enumerate(entries):
        rows = db.execute("""
            SELECT row_group, rows, bytes, finished, counts FROM row_groups WHERE shard = ? ORDER BY ordinal
        """, (shard,)).fetchall()
        counts = Counter({"evaluation_exact_overlap": 0, "evaluation_substring_overlap": 0})
        groups = []
        for group, scanned, accepted, finished, row_counts in rows:
            counts.update(json.loads(row_counts))
            groups.append({"index": group, "rows_scanned": scanned, "cleaned_bytes": accepted, "finished": bool(finished)})
        yield {**entry, "row_groups": groups, "cleaned_bytes": sum(row[2] for row in rows),
               "rows_scanned": sum(row[1] for row in rows), "filter_counts": dict(counts),
               "tokenizer_sample_bytes": sample_counts[shard]}


def build_pretrain(out, source, *, eval_prompts=None, local_root=None, settings=CorpusSettings()):
    entries = source["selected"]
    eval_keys, eval_identity = eval_prompt_keys(eval_prompts)
    screening = eval_matches(eval_keys)
    settings_dict = vars(settings)
    journal_path = out / "acquisition.sqlite3"
    with build_lock(out), heartbeat("[stage1] opening corpus journal") as report, \
            closing(sqlite3.connect(journal_path)) as db:
        init_journal(db, {"source": source, "settings": settings_dict, "eval_prompts": eval_identity})
        for index, entry in enumerate(entries):
            acquire_shard(db, source, entry, index, settings, screening, report, local_root)

        with db:
            db.execute("CREATE INDEX IF NOT EXISTS sample_order ON documents(shard, split, sample_rank, id)")
        report("[tokenizer] fitting ByteLevel BPE on TRAIN samples")
        sample_counts = [0] * len(entries)
        tokenizer = train_tokenizer(
            tokenizer_sample(journal_path, len(entries), settings.tokenizer_bytes_per_shard, sample_counts), settings.vocab_size
        )
        tokenizer.save(str(out / "tokenizer.json.partial"))
        stats = write_tokenized_splits(db, entries, out, tokenizer, report)
        if any(stat["documents"] == 0 for stat in stats.values()) or stats["train"]["tokens"] < settings.min_train_tokens:
            write_json(out / "insufficient_corpus.json", {"splits": stats, "min_train_tokens": settings.min_train_tokens,
                                                         "sources": list(source_stats(db, entries, sample_counts))})
            raise RuntimeError(f"TRAIN has {stats['train']['tokens']:,} tokens; need {settings.min_train_tokens:,} and "
                               "nonempty held-out splits. Additional source volume requires a new source manifest.")

        names = ["tokenizer.json", *(f"{split}.{suffix}" for split in SPLITS for suffix in ("bin", "jsonl"))]
        for name in names:
            (out / f"{name}.partial").replace(out / name)
        manifest = {"schema_version": 1, "corpus_id": "stage1_v1", "frozen": True, "dtype": "<u2",
                    "tokenizer_sha256": sha256((out / "tokenizer.json").read_bytes()),
                    "tokenizer_path": str((out / "tokenizer.json").resolve()), "vocab_size": settings.vocab_size,
                    "special_token_ids": SPECIAL_IDS, "splits": stats,
                    "files": {name: file_identity(out / name) for name in names},
                    "sources": {**source, "selected": list(source_stats(db, entries, sample_counts))},
                    "filter_settings": settings_dict, "seed": settings.seed, "split_percentages": [98, 1, 1],
                    "tokenizer_sample_budget_bytes": settings.tokenizer_bytes_per_shard * len(entries),
                    "tokenizer_sample_bytes": sum(sample_counts), "evaluation_screen": eval_identity,
                    "deduplication": "exact cleaned text and NFKC casefolded alphanumeric content; no fuzzy edit-distance matching"}
        write_json(out / "manifest.json", manifest)
        report(f"[stage1] froze {out / 'manifest.json'}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--out-dir", type=Path, default=Path("data/stage1_v1"))
    parser.add_argument("--eval-prompts", type=Path, help="Frozen evaluation JSONL with message lists")
    parser.add_argument("--local-root", type=Path, help="Read exact-size pinned files from this directory instead of the Hub")
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))["pretrain"]
    if len(source["selected"]) != 8:
        parser.error("Stage 1 requires the eight pinned source shards")
    build_pretrain(args.out_dir, source, eval_prompts=args.eval_prompts, local_root=args.local_root)


if __name__ == "__main__":
    main()
