"""Build a frozen Stage 2 corpus from bounded, traceable Chinese text inputs."""

import argparse
from collections import Counter
from contextlib import ExitStack, closing
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from hashlib import sha256
from importlib.metadata import version
from itertools import islice
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time

import numpy as np
from opencc import OpenCC
import psutil
from tokenizers import Tokenizer

from .chat_format import SPECIAL_IDS, check_special_ids
from .comparison_pretrain_data import (
    QUALITY_RULES, TOKENIZER_SHA256, bottom_shingles, quality_filter, shingles, strip_navigation,
)
from .stage1_data import SPLITS, build_lock, clean_text, file_identity, json_line, normalized_content, write_json
from .stage2_runtime import ensure_disk_budget


CORPUS_ID = "pretrain_stage2_500m"
SOURCE_WEIGHTS = {"fineweb2_hq": 50, "fineweb_edu": 20, "cosmopedia": 20, "wiki": 10}
STAGE1_DIR = Path("data/stage1_v1")
WIKI_DIR = Path("data/pretrain_clean_v2_wiki")
EVALUATION_DIR = Path("evaluation")
OUTPUT_DIR = Path("data") / CORPUS_ID
REPORT_DIR = Path("reports/stage2_500m")
PREFIX_TOKENS = 500_000_000
PILOT_PREFIX_TOKENS = 5_013_504
TARGET_TRAIN_TOKENS = 550_000_000
TARGET_CANDIDATE_TOKENS = 620_000_000
MIN_CANDIDATE_TOKENS = 550_000_000
MAX_CANDIDATE_TOKENS = 650_000_000
SEED = 20261123
VOCAB_SIZE = 16_384
BATCH_SIZE = 64
EOS = "<|eos|>"
RESERVED_IDS = frozenset(SPECIAL_IDS.values())
REPETITION_RULES = {
    "min_normalized_chars": 240, "long_ngram_chars": 16, "max_long_ngram_repeat_fraction": 0.28,
    "min_definition_headings": 4, "min_title_bigram_containment": 0.6, "min_body_repeat_8gram_fraction": 0.05,
}
NUMBERED_HEADING = re.compile(r"^(?:#{1,5}\s+)?\s*(\d{1,2})[、.．]\s*(.{4,72})$")
DEFINITION_HEADING = re.compile(r"区别|什么是|是什么|什么意思|什么叫")
HEADING_FILLER = re.compile(r"分别是?什么意思|有什么区别|什么是|是什么|什么意思|什么叫|区别|分别|说详细点|谢谢|[与及和的跟]")


@dataclass(frozen=True)
class FreezeBudget:
    profile: str
    prefix: int
    train: int
    candidate_target: int
    candidate_min: int
    candidate_max: int


def freeze_budget(mode):
    if mode == "freeze-pilot":
        return FreezeBudget("pilot", PILOT_PREFIX_TOKENS, 10_000_000, 12_000_000, 9_000_000, 13_000_000)
    return FreezeBudget("full", PREFIX_TOKENS, TARGET_TRAIN_TOKENS, TARGET_CANDIDATE_TOKENS,
                        MIN_CANDIDATE_TOKENS, MAX_CANDIDATE_TOKENS)


@dataclass(frozen=True)
class Input:
    path: Path
    source: str
    protected: bool = False
    limit: int | None = None
    acquisition: Path | None = None
    sample_every: int = 1


@dataclass(frozen=True)
class Candidate:
    source: str
    record_id: str
    record: dict
    text: str
    normalized: str
    protected: bool


def stable_key(purpose, value):
    return sha256(f"{SEED}:{purpose}:{value}".encode("utf-8")).digest()


def split_for_cluster(record_id):
    bucket = int.from_bytes(stable_key("split", record_id)[:8], "big") % 10_000
    return SPLITS[(bucket >= 9800) + (bucket >= 9900)]


def jaccard_match(left, right, threshold=0.8):
    return len(left & right) >= threshold * len(left | right)


def repeated_ngrams(text, width):
    count = len(text) - width + 1
    return 1 - len({text[i:i + width] for i in range(count)}) / count


def repeated_definition_headings(text):
    topics = []
    seen = set()
    for line in text.splitlines():
        heading = NUMBERED_HEADING.fullmatch(line)
        if heading and int(heading[1]) not in seen and DEFINITION_HEADING.search(heading[2]):
            seen.add(int(heading[1]))
            topic = HEADING_FILLER.sub("", normalized_content(heading[2]))
            bigrams = {topic[i:i + 2] for i in range(len(topic) - 1)}
            if len(bigrams) >= 3:
                topics.append(bigrams)
    needed = REPETITION_RULES["min_definition_headings"]
    if len(topics) < needed:
        return False
    return any(sum(
        len(topic & other) >= REPETITION_RULES["min_title_bigram_containment"] * min(len(topic), len(other))
        for other in topics
    ) >= needed for topic in topics)


def document_repetition(text, normalized, source):
    if len(normalized) < REPETITION_RULES["min_normalized_chars"]:
        return None
    if (repeated_ngrams(normalized, REPETITION_RULES["long_ngram_chars"])
            >= REPETITION_RULES["max_long_ngram_repeat_fraction"]):
        return "repeated_document_phrases"
    if source in {"fineweb2_hq", "fineweb_edu"} and repeated_definition_headings(text):
        body = normalized_content("\n".join(line for line in text.splitlines() if not NUMBERED_HEADING.fullmatch(line)))
        if (len(body) >= REPETITION_RULES["min_normalized_chars"]
                and repeated_ngrams(body, 8) >= REPETITION_RULES["min_body_repeat_8gram_fraction"]):
            return "repeated_definition_sections"
    return None


def source_record(source, row, row_number):
    origin = row.get("origin", row.get("source", source.path.parent.name))
    row_group = row.get("row_group", 0)
    row_in_group = row.get("row_in_group", row_number)
    record_id = stable_key("document", json_line(
        [source.source, origin, row_group, row_in_group, row.get("id", "")],
    )).hex()
    record = {key: row[key] for key in ("article_id", "url", "title", "licenses") if key in row}
    return {**record, "id": record_id, "source": source.source, "origin": origin,
            "row_group": row_group, "row_in_group": row_in_group}


def sampled_lines(stream, stride):
    for line in iter(stream.readline, ""):
        for _ in range(stride - 1):
            if not stream.readline():
                break
        yield line


def sample_excerpt(text):
    if not isinstance(text, str):
        return {"text": None, "characters": 0, "excerpt": False}
    safe = text.encode("utf-8", errors="replace").decode("utf-8")
    if len(safe) <= 1_000:
        return {"text": safe, "characters": len(safe), "excerpt": False}
    return {"text": f"{safe[:650]}\n[中间省略]\n{safe[-350:]}",
            "characters": len(safe), "excerpt": True}


class EvaluationScreen:
    def __init__(self, paths, converter):
        self.keys = set()
        self.signatures = {}
        self.identities = []
        for path in paths:
            data = path.read_bytes()
            keys = {
                normalized_content(converter.convert(message["content"]))
                for row in data.decode("utf-8").splitlines() if row.strip()
                for message in json.loads(row)["messages"] if message["role"] == "user"
            } - {""}
            if not keys:
                raise ValueError(f"No evaluation prompts in {path}")
            self.keys.update(keys)
            self.identities.append({
                "path": str(path.resolve()), "sha256": sha256(data).hexdigest(), "user_keys": len(keys),
            })
        # Substring checks cover full prompts inside long articles. Short prompts
        # require a whole line or sentence match to avoid dropping common phrases.
        long_keys = sorted((key for key in self.keys if len(key) >= 24), key=lambda k: (-len(k), k))
        self.substrings = re.compile("|".join(map(re.escape, long_keys))) if long_keys else None
        for key in self.keys:
            if len(key) >= 10:
                _, signature = bottom_shingles(key)
                for value in signature:
                    self.signatures.setdefault(value, []).append(key)
        self.max_near_chars = int(max(map(len, self.keys), default=0) / 0.8)

    def conflict(self, text, normalized):
        if normalized in self.keys or (self.substrings is not None and self.substrings.search(normalized)):
            return True
        # Compare full documents and standalone lines. A separate sentence pass
        # catches short exact prompts and paraphrase-sized spans inside a line.
        for part in re.split(r"[\n。！？!?]+", text):
            if self.near(normalized_content(part)):
                return True
        return self.near(normalized) or any(self.near(normalized_content(line)) for line in text.splitlines())

    def near(self, normalized):
        if normalized in self.keys:
            return True
        if not 10 <= len(normalized) <= self.max_near_chars:
            return False
        _, signature = bottom_shingles(normalized)
        counts = Counter(key for value in signature for key in self.signatures.get(value, ()))
        relevant = [
            key for key, count in counts.items()
            if count >= 2 and min(len(key), len(normalized)) >= 0.8 * max(len(key), len(normalized))
        ]
        if not relevant:
            return False
        current = shingles(normalized)
        return any(jaccard_match(current, shingles(key))
                   or SequenceMatcher(None, normalized, key, autojunk=False).ratio() >= 0.9
                   for key in relevant)


class Corpus:
    def __init__(self, db, tokenizer, converter, screen, request, samples=False):
        self.db = db
        self.tokenizer = tokenizer
        self.converter = converter
        self.screen = screen
        self.samples = samples
        db.executescript("""
            PRAGMA cache_size = -32768;
            PRAGMA temp_store = FILE;
            CREATE TABLE IF NOT EXISTS request (spec TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY, record_id TEXT NOT NULL, exact BLOB UNIQUE NOT NULL,
                source TEXT NOT NULL, metadata TEXT NOT NULL, text TEXT NOT NULL, token_ids BLOB,
                tokens INTEGER NOT NULL, ngrams INTEGER NOT NULL, parent INTEGER,
                blocked INTEGER NOT NULL, shuffle BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS document_shuffle ON documents(shuffle);
            CREATE TABLE IF NOT EXISTS signatures (
                value BLOB NOT NULL, document INTEGER NOT NULL, PRIMARY KEY(value, document)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS inputs (
                path TEXT PRIMARY KEY, identity TEXT NOT NULL, position INTEGER NOT NULL, rows INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS counters (
                source TEXT NOT NULL, reason TEXT NOT NULL, total INTEGER NOT NULL,
                PRIMARY KEY(source, reason)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS rejected_samples (
                source TEXT NOT NULL, rank BLOB NOT NULL, record TEXT NOT NULL,
                PRIMARY KEY(source, rank)
            ) WITHOUT ROWID;
        """)
        specification = json.dumps(request, sort_keys=True)
        previous = db.execute("SELECT spec FROM request").fetchone()
        if previous is not None and previous[0] != specification:
            raise ValueError("Corpus build inputs or processing settings changed")
        if previous is None:
            db.execute("INSERT INTO request VALUES (?)", (specification,))
            db.commit()

    def prepare(self, source, row, row_number):
        if source.protected and source.source == "sft_cmrc":
            group = row.get("source_group")
            if (row.get("split") not in {"validation", "test"} or not isinstance(group, str) or not group.strip()
                    or not isinstance(row.get("context"), str) or not row["context"].strip()):
                raise ValueError(f"Invalid SFT held-out context at {source.path}:{row_number + 1}")
            raw = row["context"]
        else:
            raw = row.get("text")
        if not isinstance(raw, str):
            return "invalid_or_oversize", None
        if not source.protected and len(raw) > QUALITY_RULES["max_document_characters"]:
            return "invalid_or_oversize", None
        text = self.converter.convert(raw)
        if source.protected:
            cleaned = clean_text(text)
            lines, _, _ = strip_navigation(cleaned)
            text = "\n".join(lines) or cleaned
        else:
            reason, text, _, _ = quality_filter(text)
            if reason:
                return reason, None
        normalized = normalized_content(text)
        if not normalized:
            return "empty_normalized", None
        if not source.protected and (reason := document_repetition(text, normalized, source.source)):
            return reason, None
        record = source_record(source, row, row_number)
        return None, Candidate(source.source, record["id"], record, text, normalized, source.protected)

    def cluster(self, index):
        item = self.db.execute("SELECT parent, blocked FROM documents WHERE id = ?", (index,)).fetchone()
        path = []
        while item[0] is not None:
            path.append(index)
            index = item[0]
            item = self.db.execute("SELECT parent, blocked FROM documents WHERE id = ?", (index,)).fetchone()
        if len(path) > 1:
            self.db.executemany("UPDATE documents SET parent = ? WHERE id = ?", ((index, row) for row in path))
        return index, item[1]

    def add(self, candidate, encoding):
        exact = sha256(candidate.normalized.encode("utf-8")).digest()
        existing = self.db.execute("SELECT id FROM documents WHERE exact = ?", (exact,)).fetchone()
        blocked = candidate.protected or self.screen.conflict(candidate.text, candidate.normalized)
        if existing is not None:
            root, was_blocked = self.cluster(existing[0])
            if blocked and not was_blocked:
                self.db.execute("UPDATE documents SET blocked = 1 WHERE id = ?", (root,))
            return ("protected_duplicate" if blocked or was_blocked else "exact_duplicate"), 0

        ngrams, features = bottom_shingles(candidate.normalized)
        feature_keys = tuple(value.to_bytes(8, "big") for value in features)
        roots = {}
        current_shingles = None
        if len(feature_keys) >= 2:
            possible = self.db.execute(f"""
                SELECT d.id, d.text
                FROM (
                    SELECT document FROM signatures
                    WHERE value IN ({",".join("?" for _ in feature_keys)})
                    GROUP BY document HAVING COUNT(*) >= 2
                ) AS matches JOIN documents d ON d.id = matches.document
                WHERE d.ngrams BETWEEN ? AND ?
                """, (*feature_keys, 0.8 * ngrams, ngrams / 0.8))
            for index, text in possible:
                root, root_blocked = self.cluster(index)
                if root in roots:
                    continue
                if current_shingles is None:
                    current_shingles = shingles(candidate.normalized)
                if jaccard_match(current_shingles, shingles(normalized_content(text))):
                    roots[root] = root_blocked

        if encoding is None:
            tokens, token_ids = 0, None
        else:
            ids = encoding.ids
            if not ids or not RESERVED_IDS.isdisjoint(ids):
                return "invalid_tokenization", 0
            tokens = len(ids) + 1
            token_ids = np.asarray([*ids, SPECIAL_IDS[EOS]], dtype="<u2").tobytes()
        parent = min(roots, default=None)
        cluster_blocked = blocked or any(roots.values())
        cursor = self.db.execute("""
            INSERT INTO documents (record_id, exact, source, metadata, text, token_ids,
                                   tokens, ngrams, parent, blocked, shuffle)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (candidate.record_id, exact, candidate.source, json_line(candidate.record), candidate.text,
                  token_ids, tokens, ngrams, parent, cluster_blocked, stable_key("shuffle", candidate.record_id)))
        self.db.executemany("INSERT INTO signatures VALUES (?, ?)",
                            ((feature, cursor.lastrowid) for feature in feature_keys))
        if parent is not None:
            self.db.executemany("UPDATE documents SET parent = ? WHERE id = ?",
                                ((parent, root) for root in roots if root != parent))
            self.db.execute("UPDATE documents SET blocked = ? WHERE id = ?", (cluster_blocked, parent))
        if cluster_blocked:
            return ("protected_holdout" if candidate.protected else "protected_overlap"), 0
        if parent is not None:
            return "near_duplicate", 0
        return "kept_at_ingest", tokens

    def ingest(self, source, output, token_limit=None):
        path = str(source.path.resolve())
        identity = json_line({
            "file": file_identity(source.path), "source": source.source,
            "protected": source.protected, "limit": source.limit, "sample_every": source.sample_every,
            "acquisition": file_identity(source.acquisition) if source.acquisition else None,
        })
        snapshot = self.db.execute("SELECT identity, position, rows FROM inputs WHERE path = ?", (path,)).fetchone()
        if snapshot is not None and identity != snapshot[0]:
            raise ValueError(f"Previously ingested input changed: {path}")
        position, rows = snapshot[1:] if snapshot is not None else (0, 0)
        previous_rows = rows
        label = source_label(source)
        accumulated = self.db.execute("SELECT total FROM counters WHERE source = ? AND reason = 'tokens_at_ingest'",
                                      (label,)).fetchone()
        source_tokens = accumulated[0] if accumulated is not None else 0
        started = time.monotonic()
        with source.path.open(encoding="utf-8") as stream:
            stream.seek(position)
            incoming = sampled_lines(stream, source.sample_every)
            while source.limit is None or rows < source.limit:
                if token_limit is not None and source_tokens >= token_limit:
                    break
                batch_size = min(BATCH_SIZE, source.limit - rows) if source.limit is not None else BATCH_SIZE
                lines = list(islice(incoming, batch_size))
                if not lines:
                    break
                ensure_disk_budget(output, required_bytes=sum(map(len, lines)) * 16)
                counts = Counter(rows=len(lines))
                candidates = []
                rejected = []
                for index, line in enumerate(lines, start=rows):
                    row = json.loads(line)
                    reason, candidate = self.prepare(source, row, index)
                    if reason:
                        counts[reason] += 1
                        if self.samples and not source.protected:
                            record = source_record(source, row, index)
                            rank = stable_key("rejected_sample", record["id"])
                            rejected.append((source.source, rank, json_line({
                                **record, "decision": reason, **sample_excerpt(row.get("text")),
                            })))
                    else:
                        candidates.append(candidate)
                encodings = ([None] * len(candidates) if source.protected else self.tokenizer.encode_batch(
                    [item.text for item in candidates], add_special_tokens=False,
                ))
                with self.db:
                    for candidate, encoding in zip(candidates, encodings, strict=True):
                        reason, tokens = self.add(candidate, encoding)
                        counts[reason] += 1
                        counts["tokens_at_ingest"] += tokens
                    if rejected:
                        self.db.executemany("INSERT OR IGNORE INTO rejected_samples VALUES (?, ?, ?)", rejected)
                        self.db.execute("""
                            DELETE FROM rejected_samples WHERE source = ? AND rank NOT IN (
                                SELECT rank FROM rejected_samples WHERE source = ? ORDER BY rank LIMIT 4
                            )
                            """, (source.source, source.source))
                    source_tokens += counts["tokens_at_ingest"]
                    self.db.executemany("""
                        INSERT INTO counters VALUES (?, ?, ?)
                        ON CONFLICT(source, reason) DO UPDATE SET total = total + excluded.total
                        """, ((label, key, count) for key, count in counts.items()))
                    rows += len(lines)
                    self.db.execute("INSERT OR REPLACE INTO inputs VALUES (?, ?, ?, ?)",
                                    (path, identity, stream.tell(), rows))
                if rows % 2048 == 0:
                    print(f"[stage2-data] {label} {source.path.name}: {rows:,} rows / {source_tokens:,} tokens", flush=True)
        if json.loads(identity)["file"] != file_identity(source.path):
            raise ValueError(f"Input changed while it was being ingested: {path}")
        elapsed = time.monotonic() - started
        print(f"[stage2-data] {label} {source.path.name}: {rows - previous_rows:,} new rows in "
              f"{elapsed:.2f}s ({rows:,} cumulative)", flush=True)
        return {"source": label, "path": path, "new_rows": rows - previous_rows,
                "elapsed_seconds": round(elapsed, 3)}

    def candidate_counts(self):
        return {
            source: {"documents": documents, "tokens": tokens}
            for source, documents, tokens in self.db.execute("""
                SELECT source, count(*), sum(tokens) FROM documents
                WHERE parent IS NULL AND blocked = 0 AND tokens > 0 GROUP BY source
                """)
        }

    def counters(self):
        counts = {}
        for source, reason, total in self.db.execute("SELECT source, reason, total FROM counters"):
            counts.setdefault(source, {})[reason] = total
        for entries in counts.values():
            rows = entries.get("rows", 0)
            accepted = sum(entries.get(key, 0) for key in (
                "kept_at_ingest", "exact_duplicate", "near_duplicate", "protected_overlap", "protected_duplicate",
            ))
            entries["cleaning_pass_rate"] = accepted / rows if rows else 0.0
        return counts

    def quality_samples(self):
        result = []
        for source in SOURCE_WEIGHTS:
            discarded = [json.loads(record) for (record,) in self.db.execute(
                "SELECT record FROM rejected_samples WHERE source = ? ORDER BY rank", (source,),
            )]
            for metadata, text, tokens in self.db.execute("""
                SELECT metadata, text, tokens FROM documents
                WHERE source = ? AND parent IS NULL AND blocked = 0 AND tokens > 0 ORDER BY shuffle LIMIT ?
                """, (source, 10 - len(discarded))):
                result.append({**json.loads(metadata), "decision": "kept_after_cleaning_and_deduplication",
                               "tokens": tokens, **sample_excerpt(text)})
            result.extend(discarded)
        return result


def source_label(source):
    if source.protected and source.source == "sft_cmrc":
        return "sft_cmrc_holdout"
    return f"old_holdout_{source.source}" if source.protected else source.source


def read_tokenizer(stage1_dir):
    manifest = json.loads((stage1_dir / "manifest.json").read_text(encoding="utf-8"))
    tokenizer_path = (stage1_dir / "tokenizer.json").resolve()
    if sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256 or TOKENIZER_SHA256 != manifest["tokenizer_sha256"]:
        raise ValueError("Frozen Stage 1 tokenizer identity changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    check_special_ids(tokenizer)
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise ValueError("Frozen Stage 1 vocabulary changed")
    return tokenizer, tokenizer_path, manifest


def legacy_inputs(stage1_dir, wiki_dir, probe=False):
    sources = {"fineweb2_hq": stage1_dir, "wiki": wiki_dir}
    holdouts = [
        Input(directory / f"{split}.jsonl", source, protected=True, limit=125 if probe else None)
        for source, directory in sources.items() for split in ("validation", "test")
    ]
    train = [Input(directory / "train.jsonl", source, limit=250 if probe else None)
             for source, directory in sources.items()]
    return [*holdouts, *train]


def local_provenance(stage1_dir, wiki_dir, old_manifest):
    wiki_manifest_path = wiki_dir / "manifest.json"
    wiki_manifest = json.loads(wiki_manifest_path.read_text(encoding="utf-8"))
    result = {}
    for source, directory, manifest, spec in (
        ("fineweb2_hq", stage1_dir, old_manifest, old_manifest["sources"]),
        ("wiki", wiki_dir, wiki_manifest, wiki_manifest["sources"]["wiki"]),
    ):
        if manifest["frozen"] is not True:
            raise ValueError(f"Legacy source is not frozen: {directory}")
        for split in SPLITS:
            name = f"{split}.jsonl"
            if file_identity(directory / name) != manifest["files"][name]:
                raise ValueError(f"Frozen legacy split identity changed: {directory / name}")
        details = {key: spec[key] for key in ("repo", "revision", "config", "licenses", "license_urls", "attribution")
                   if key in spec}
        if source == "fineweb2_hq":
            details.update(licenses=["ODC-By-1.0", "Common Crawl Terms of Use"], config="cmn_Hani")
        result[source] = {"manifest": str((directory / "manifest.json").resolve()), **details}
    return result


def input_provenance(inputs):
    values = []
    for source in inputs:
        details = {"source": source.source, "path": str(source.path.resolve()), "protected": source.protected,
                   "limit": source.limit, "sample_every": source.sample_every, "file": file_identity(source.path)}
        if details["file"]["bytes"] <= 0:
            raise ValueError(f"Input must be nonempty: {source.path}")
        if source.acquisition is not None:
            acquisition = json.loads(source.acquisition.read_text(encoding="utf-8"))
            if (acquisition.get("status") != "success" or acquisition.get("source") != source.source
                    or acquisition.get("metadata_only") or acquisition.get("rows", 0) <= 0
                    or acquisition.get("output_file") != {"path": str(source.path.resolve()), **details["file"]}):
                raise ValueError(f"Acquisition report does not match {source.path}")
            details["acquisition"] = {k: v for k, v in acquisition.items() if k != "groups"}
        values.append(details)
    return values


def verify_candidate_pool(candidate_counts, budget):
    total = sum(stats["tokens"] for stats in candidate_counts.values())
    if not budget.candidate_min <= total <= budget.candidate_max:
        raise ValueError(f"Net candidate tokens {total:,} are outside the "
                         f"{budget.candidate_min:,}–{budget.candidate_max:,} target")
    if set(candidate_counts) != set(SOURCE_WEIGHTS):
        raise ValueError("Every Stage 2 corpus source must have net candidates")


def write_splits(corpus, output, budget=None):
    budget = budget or freeze_budget("freeze")
    counts = {
        split: {"documents": 0, "tokens": 0, "binary_bytes": 0,
                "source_tokens": Counter(), "special_token_counts": dict.fromkeys(SPECIAL_IDS, 0)}
        for split in SPLITS
    }
    prefix = Counter()
    limits = {source: budget.train * percent // 100 for source, percent in SOURCE_WEIGHTS.items()}
    ensure_disk_budget(output)
    with ExitStack() as opened:
        records = {split: opened.enter_context(
            (output / f"{split}.jsonl.partial").open("w", encoding="utf-8", newline="\n"),
        ) for split in SPLITS}
        binaries = {split: opened.enter_context((output / f"{split}.bin.partial").open("wb")) for split in SPLITS}
        cursor = corpus.db.execute("""
            SELECT record_id, source, metadata, text, token_ids, tokens FROM documents
            WHERE parent IS NULL AND blocked = 0 AND tokens > 0 ORDER BY shuffle
            """)
        while batch := cursor.fetchmany(BATCH_SIZE):
            ensure_disk_budget(output, required_bytes=sum(len(row[3]) * 4 + len(row[4]) + 1024 for row in batch))
            for record_id, source, metadata, text, token_ids, length in batch:
                split = split_for_cluster(record_id)
                stats = counts[split]
                if split == "train":
                    if stats["source_tokens"][source] >= limits[source]:
                        continue
                    start = stats["tokens"]
                    # LM targets are one position ahead of inputs.
                    prefix[source] += max(0, min(start + length, budget.prefix + 1) - max(start, 1))
                records[split].write(json_line({**json.loads(metadata), "text": text, "tokens": length}))
                binaries[split].write(token_ids)
                stats["source_tokens"][source] += length
                stats["documents"] += 1
                stats["tokens"] += length
                stats["special_token_counts"][EOS] += 1
    for split, stats in counts.items():
        stats["binary_bytes"] = 2 * stats["tokens"]
        if stats["documents"] == 0:
            raise ValueError(f"The {split} split is empty")
        print(f"[stage2-data] {split}: {stats['documents']:,} documents / {stats['tokens']:,} tokens", flush=True)
    return counts, prefix


def verify_mix(counts, prefix, budget):
    train = counts["train"]
    mix = train["source_tokens"]
    if train["tokens"] <= budget.prefix or sum(prefix.values()) != budget.prefix:
        raise ValueError(f"Training stream cannot provide {budget.prefix:,} next-token targets")
    if train["tokens"] < budget.train:
        raise ValueError(f"Training selection is short of the {budget.train:,}-token target: {train['tokens']:,}")
    for label, mixture in (("train", mix), ("train_prefix", prefix)):
        total = sum(mixture.values())
        for source, target in SOURCE_WEIGHTS.items():
            percent = 100 * mixture.get(source, 0) / total
            if abs(percent - target) > 1.5 or (source == "wiki" and percent > 15):
                raise ValueError(f"{label} source {source} contributes {percent:.2f}%; expected {target}%")


def publish(corpus, output, stage1_dir, old_manifest, tokenizer_path, provenance, report, budget=None):
    if (output / "manifest.json").exists():
        raise RuntimeError(f"A frozen manifest already exists at {output / 'manifest.json'}")
    budget = budget or freeze_budget("freeze")
    verify_candidate_pool(report["net_candidates"], budget)
    counts, prefix = write_splits(corpus, output, budget)
    verify_mix(counts, prefix, budget)
    old_validation = stage1_dir / "validation.bin"
    old_identity = file_identity(old_validation)
    if old_identity != old_manifest["files"]["validation.bin"]:
        raise ValueError("Fixed general validation corpus changed")
    ensure_disk_budget(output, required_bytes=old_identity["bytes"])
    shutil.copyfile(old_validation, output / "general_validation.bin.partial")
    filenames = [*(f"{split}.{suffix}" for split in SPLITS for suffix in ("bin", "jsonl")), "general_validation.bin"]
    for name in filenames:
        (output / f"{name}.partial").replace(output / name)
    report["splits"] = counts
    report["train_prefix_source_tokens"] = prefix
    report["train_prefix_targets"] = budget.prefix
    manifest = {
        "schema_version": 1, "corpus_id": CORPUS_ID, "frozen": True, "dtype": "<u2",
        "vocab_size": VOCAB_SIZE, "tokenizer_sha256": TOKENIZER_SHA256, "tokenizer_path": str(tokenizer_path),
        "special_token_ids": SPECIAL_IDS, "seed": SEED, "splits": counts, "sources": provenance,
        "files": {"tokenizer.json": file_identity(tokenizer_path),
                  **{name: file_identity(output / name) for name in filenames}},
        "stage2": {
            "profile": budget.profile,
            "general_validation": {"path": "general_validation.bin", **old_manifest["splits"]["validation"]},
            "general_validation_origin": {"path": str(old_validation.resolve()), **old_identity},
            "source_weight_percent": SOURCE_WEIGHTS,
            "train_prefix_source_tokens": prefix, "train_prefix_targets": budget.prefix,
            "net_candidates": report["net_candidates"],
            "local_source_provenance": report["local_source_provenance"],
            "evaluation_screen": corpus.screen.identities, "conversion": report["conversion"],
            "quality_rules": {**QUALITY_RULES, "stage2_repetition": REPETITION_RULES},
            "deduplication": report["deduplication"],
        },
    }
    write_json(output / "manifest.json", manifest)


def run(mode, output, report_path, stage1_dir, wiki_dir, eval_dir, remote_inputs=(),
        prepare_only=False, protected_contexts=()):
    started = time.monotonic()
    if mode == "probe-remote":
        inputs = [item for item in legacy_inputs(stage1_dir, wiki_dir, probe=True)
                  if item.protected or item.source == "wiki"]
        remote_inputs = [
            replace(item, limit=1000, sample_every=max(
                1, json.loads(item.acquisition.read_text(encoding="utf-8"))["rows"] // 1000,
            ))
            for item in remote_inputs
        ]
    else:
        inputs = legacy_inputs(stage1_dir, wiki_dir, probe=(mode == "probe-local"))
    if mode == "freeze-pilot":
        inputs = [*(item for item in inputs if item.protected), *remote_inputs,
                  *(item for item in inputs if not item.protected)]
    else:
        inputs.extend(remote_inputs)
    if mode == "freeze":
        if not prepare_only and not protected_contexts:
            raise ValueError("Formal freezing requires the held-out SFT context file")
        inputs = [*(Input(path, "sft_cmrc", protected=True) for path in protected_contexts), *inputs]
    elif protected_contexts:
        raise ValueError("Held-out SFT context files apply to the formal corpus only")
    provenance = input_provenance(inputs)
    tokenizer, tokenizer_path, old_manifest = read_tokenizer(stage1_dir)
    local_sources = local_provenance(stage1_dir, wiki_dir, old_manifest)
    converter = OpenCC("t2s")
    conversion = {"library": "opencc-python-reimplemented", "version": version("opencc-python-reimplemented"),
                  "config": "t2s"}
    paths = sorted(path for path in eval_dir.rglob("prompts*.jsonl") if path.name in {
        "prompts.jsonl", "prompts_for_overlap.jsonl",
    })
    if not paths:
        raise ValueError(f"No known evaluation files in {eval_dir}")
    screen = EvaluationScreen(paths, converter)
    budget = freeze_budget(mode)
    request = {"mode": mode, "seed": SEED, "conversion": conversion, "quality_rules": QUALITY_RULES,
               "processing_version": 3, "stage2_repetition": REPETITION_RULES,
               "tokenizer_sha256": TOKENIZER_SHA256, "evaluation": screen.identities}
    ensure_disk_budget(output)
    ensure_disk_budget(report_path)
    output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with build_lock(output), closing(sqlite3.connect(output / "candidates.sqlite3")) as db:
        corpus = Corpus(db, tokenizer, converter, screen, request, samples=mode == "probe-remote" or prepare_only)
        previous = {path for (path,) in db.execute("SELECT path FROM inputs")}
        if not previous <= {item["path"] for item in provenance}:
            raise ValueError("Keep previously ingested source files in the build inputs when continuing")
        report = {
            "mode": mode, "status": "failed", "sources": provenance, "conversion": conversion,
            "local_source_provenance": local_sources,
            "processing": [],
            "stage2_repetition_rules": REPETITION_RULES,
            "deduplication": "Exact normalized digest and character 5-gram Jaccard >= 0.8 over "
                             "candidates with at least two shared bottom-16 hashes; connected components",
        }
        try:
            for item in inputs:
                token_limit = (budget.candidate_target * SOURCE_WEIGHTS[item.source] // 100
                               if mode.startswith("freeze") and not item.protected else None)
                report["processing"].append(corpus.ingest(item, output, token_limit))
            report["net_candidates"] = corpus.candidate_counts()
            if mode.startswith("freeze") and not prepare_only:
                publish(corpus, output, stage1_dir, old_manifest, tokenizer_path, provenance, report, budget)
            if mode == "probe-remote" or prepare_only:
                samples_path = report_path.with_name(report_path.stem + "_samples.json")
                write_json(samples_path, corpus.quality_samples())
                report["quality_samples"] = str(samples_path.resolve())
            report["status"] = "prepared" if prepare_only else "success"
        except (OSError, RuntimeError, ValueError) as exc:
            report["error"] = str(exc)
            raise
        finally:
            memory = psutil.Process().memory_info()
            report.update(counters=corpus.counters(), elapsed_seconds=round(time.monotonic() - started, 3),
                          peak_rss_bytes=getattr(memory, "peak_wset", memory.rss))
            write_json(report_path, report)
            print(f"[stage2-data] wrote {report_path}", flush=True)
    return report


def remote_input(value):
    source, sep, names = value.partition("=")
    data, sep_report, report = names.partition("|")
    if not sep or not sep_report or not data or not report or source not in set(SOURCE_WEIGHTS) - {"wiki"}:
        raise argparse.ArgumentTypeError("Expected SOURCE=DATA.jsonl|ACQUISITION.json for a remote source")
    return Input(Path(data), source, acquisition=Path(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe-local", "probe-remote", "freeze-pilot", "freeze"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--stage1-dir", type=Path, default=STAGE1_DIR)
    parser.add_argument("--wiki-dir", type=Path, default=WIKI_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--remote", type=remote_input, action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true",
                        help="Ingest available formal inputs while additional sources are acquired.")
    parser.add_argument("--protected-contexts", action="append", type=Path, default=[],
                        help="Add a frozen SFT validation/test context JSONL to formal contamination screening.")
    args = parser.parse_args()
    if (args.mode == "probe-local" and args.remote) or (args.mode != "probe-local" and not args.remote):
        parser.error("Remote input is required in probe-remote, freeze-pilot and freeze modes only")
    if args.prepare_only and args.mode != "freeze":
        parser.error("--prepare-only requires formal freeze mode")
    output = args.output or (OUTPUT_DIR if args.mode == "freeze" else OUTPUT_DIR / args.mode)
    report = args.report or REPORT_DIR / f"data_{args.mode}.json"
    run(args.mode, output, report, args.stage1_dir, args.wiki_dir, args.evaluation_dir, args.remote,
        args.prepare_only, args.protected_contexts)


if __name__ == "__main__":
    main()
