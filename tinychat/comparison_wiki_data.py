"""Prepare a Wikipedia-only comparison corpus with the frozen Stage 1 tokenizer."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
import json
from pathlib import Path
import re
import sqlite3

from .chat_format import SPECIAL_IDS, contains_reserved
from .comparison_pretrain_data import (
    CORPUS_ID, MAX_POSTINGS, MIN_TRAIN_TOKENS, QUALITY_SAMPLE_SEED, SEED, WIKI_COLUMNS, WIKI_LICENSES, WIKI_REVISION,
    CorpusBuild, Document, document_id, sample_record, seeded_key, verify_legacy_inputs,
    wiki_metadata, write_splits,
)
from .stage1_data import HAN, SPLITS, build_lock, clean_text, eval_prompt_keys, file_identity, normalized_content, write_json


OUTPUT = Path("data/pretrain_clean_v2_wiki")
OLD_DATA = Path("data/stage1_v1")
SOURCE = Path("planning/train_150mb/comparison_source_probe.json")
REPORT = Path("reports/comparison_v2/predata_wiki_build.json")
SAMPLES = Path("reports/comparison_v2/predata_wiki_quality_samples.json")
REVIEW_MARKER = Path("reports/comparison_v2/predata_wiki_review_ready.txt")
EVAL_PROMPTS = (Path("evaluation/stage1_v1/prompts.jsonl"), Path("evaluation/comparison_v2/prompts.jsonl"))
SOURCE_STUB = re.compile(r"(?:列表|一覽|一览|年表|時間表|时间表|消歧義|消歧义)$")
REFERENCE = re.compile(r"^(?:參考(?:文獻|文献|資料|资料|來源|来源|書目|书目)?|参考(?:文獻|文献|資料|资料|來源|来源|書目|书目)?|外部[連链鏈]接|外部連結|延伸[閱阅][讀读]|相關[條条]目|相关[條条]目|参见|參見|註[釋释]|注[釋释]|引文|腳註|脚注|Bibliography|References|External links)[：:\s]*$", re.IGNORECASE)
BOILERPLATE = re.compile(r"(?:[编辑編輯]本[頁页]|本[條条]目.*(?:[擴扩]充|[維维]基百科)|此[條条]目.*(?:小作品|[擴扩]充)|取自[「『\"]?https?://|\[\[(?:Category|分类|分類):)", re.IGNORECASE)
ADVERT = re.compile(r"(?:加微信|添加微信|掃碼下單|扫码下单|立即[購购]買|立即购买|客服微信|招商加盟|点击购买|點擊購買)")
UNREADABLE = re.compile(r"[\ufffd\ud800-\udfff\ue000-\uf8ff]|锟斤拷|烫烫烫")
URL = re.compile(r"(?:https?://|www\.)[^\s、，。！？；）】》]+", re.IGNORECASE)
EMPTY_MARKUP = re.compile(r"[（(]\s*[）)]|[【\[]\s*[】\]]")
BULLET = re.compile(r"^(?:[*•●◆▪#│|]|\d{1,3}[.、)）])")
END = re.compile(r"[。！？；!?](?:[」』”’）)]*)")
QUALITY_RULES = {
    "minimum_han_characters": 210, "minimum_han_fraction": 0.50, "minimum_sentence_marks": 3,
    "metadata_policy": "Keep article URL, title, page ID, license, parquet file and row provenance outside tokenized text.",
    "body_policy": "Retain running prose and associated section headings; strip reference tails, lists, URL-only and navigation lines.",
    "limitations": "Heuristics can remove legitimate passages and retain prose with defects. Both traditional and simplified Chinese remain.",
}


def clean_article_body(raw):
    """Reduce plain-text dumps to prose and accompanying section headings."""
    cleaned = clean_text(raw)
    lines = []
    heading = None
    for line in cleaned.splitlines():
        if REFERENCE.fullmatch(line):
            break
        if BOILERPLATE.search(line) or BULLET.match(line):
            continue
        line = EMPTY_MARKUP.sub("", URL.sub("", line)).strip()
        if len(line) >= 25 and END.search(line) and len(HAN.findall(line)) >= 15:
            if heading and lines:
                lines.append(heading)
            lines.append(line)
            heading = None
        elif 2 <= len(line) <= 24 and re.fullmatch(r"[\w（）()、· -]+", line) and HAN.search(line):
            heading = line
    return "\n".join(lines)


def article_body(raw, title):
    if not isinstance(raw, str) or not isinstance(title, str) or len(raw) > 750_000:
        return "invalid_or_oversize", None, 0
    if SOURCE_STUB.search(title.strip()) or re.search(r"[（(]消歧[義义][）)]$", title.strip()):
        return "list_or_disambiguation", None, 0
    if UNREADABLE.search(raw) or contains_reserved(raw):
        return "garbled_or_reserved", None, 0
    if ADVERT.search(raw):
        return "commercial_call_to_action", None, 0
    body = clean_article_body(raw)
    han = len(HAN.findall(body))
    if han < QUALITY_RULES["minimum_han_characters"] or len(END.findall(body)) < QUALITY_RULES["minimum_sentence_marks"]:
        return "stub_or_nonprose", None, 0
    if han < QUALITY_RULES["minimum_han_fraction"] * sum(char.isalnum() for char in body):
        return "insufficient_chinese", None, 0
    return None, body, max(0, len(raw) - len(body))


def build_prompt_screen(paths):
    keys = set()
    identities = []
    for path in paths:
        path_keys, identity = eval_prompt_keys(path)
        identities.append({**identity, "substring_min_chars": 6})
        keys.update(path_keys)
    long_keys = tuple(sorted(key for key in keys if len(key) >= 6))

    def find_conflict(text):
        normalized = normalized_content(text)
        if normalized in keys:
            return "evaluation_exact_overlap"
        if any(key in normalized for key in long_keys):
            return "evaluation_substring_overlap"
        return None

    return find_conflict, identities


def add_old_holdouts(build, old_dir, progress):
    for split in ("validation", "test"):
        path = old_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record["id"] = document_id("stage1_v1", record["source"], record["row_group"], record["row_in_group"])
                original = len(build.index.documents)
                build.add(record, old_holdout=True)
                article_view = clean_article_body(record["text"])
                if article_view and article_view != record["text"]:
                    alternate = build.index.add(Document({**record, "text": article_view,
                                                           "old_holdout_view": "wikipedia_body"}, old_holdout=True))
                    build.index.connect(alternate, original)
                    build.counts[record["source"]]["old_holdout_wikipedia_body_views"] += 1
        progress(f"indexed old pretraining {split} originals and cleaning views")


def ingest_wiki(build, records, find_conflict):
    raw_bytes = retained_bytes = 0
    for record in records:
        raw = record["text"]
        if isinstance(raw, str):
            raw_bytes += len(raw.encode("utf-8", errors="replace"))
        counts = build.counts[record["source"]]
        counts["input_documents"] += 1
        reason, body, removed = article_body(raw, record["title"])
        if reason:
            counts[reason] += 1
            build.sample(reason, record)
            continue
        candidate = {**record, "text": body}
        reason = find_conflict(raw)
        if reason is None and body != raw:
            reason = find_conflict(body)
        if reason:
            build.sample(reason, candidate)
        retained_bytes += len(body.encode("utf-8"))
        counts["passes_quality"] += 1
        if removed:
            counts["documents_trimmed"] += 1
            counts["boilerplate_characters_removed"] += removed
        quality = 1 - removed / len(raw)
        build.index.add(Document(candidate, quality, evaluation_conflict=reason))
    return {"raw_text_utf8_bytes": raw_bytes, "quality_retained_utf8_bytes": retained_bytes}


def stratified_samples(selected, count=32):
    by_source = defaultdict(list)
    for doc in selected["train"]:
        by_source[doc.source].append(doc)
    strata = {}
    for source, docs in sorted(by_source.items()):
        by_length = sorted(docs, key=lambda doc: (len(doc.text), doc.id))
        for band in range(3):
            items = by_length[band * len(docs) // 3:(band + 1) * len(docs) // 3]
            strata[source, band] = sorted(items, key=lambda doc: seeded_key(f"wiki_quality_{QUALITY_SAMPLE_SEED}", doc.id))
    strata = {key: iter(items) for key, items in strata.items()}
    samples = []
    while strata and len(samples) < count:
        for (source, band), candidates in list(strata.items()):
            doc = next(candidates, None)
            if doc is None:
                del strata[source, band]
                continue
            samples.append({"case_id": f"wiki_{len(samples) + 1:02d}", "length_band": ("short", "middle", "long")[band],
                            **sample_record(doc.record)})
            if len(samples) == count:
                break
    return samples


def read_wiki_group(parquet, entry):
    path, group = entry["path"], entry["row_group"]
    rows = parquet.read_row_group(group, columns=list(WIKI_COLUMNS)).to_pylist()
    return [{"id": document_id(f"wiki:{WIKI_REVISION}", path, group, index), "source": f"wiki/{path}",
             "row_group": group, "row_in_group": index, "article_id": row["id"], "url": row["url"],
             "title": row["title"], "licenses": WIKI_LICENSES, "text": row["text"]}
            for index, row in enumerate(rows)]


def pending_group_batches(files, committed, progress):
    """Load one group per shard concurrently and deliver the original stable round order."""
    ordinal = 0
    with ThreadPoolExecutor(max_workers=len(files)) as workers:
        while files:
            remaining, jobs = [], []
            for path, parquet, order in files:
                group = next(order, None)
                if group is None:
                    continue
                remaining.append((path, parquet, order))
                ordinal += 1
                entry = {"path": path, "row_group": group,
                         "documents": parquet.metadata.row_group(group).num_rows}
                if ordinal <= len(committed):
                    if any(committed[ordinal - 1][key] != value for key, value in entry.items()):
                        raise ValueError("Source group ordering differs from the acquisition journal")
                    continue
                jobs.append((ordinal, entry, workers.submit(read_wiki_group, parquet, entry)))
            if jobs:
                progress("reading Wikipedia row groups " + ", ".join(f"{entry['path'].split('/')[-1]}:{entry['row_group']}"
                                                                      for _, entry, _ in jobs))
            for position, entry, job in jobs:
                yield position, entry, job.result()
            files = remaining
        if ordinal < len(committed):
            raise ValueError("Source exhausted before replaying committed row groups")


def read_pending_groups(metadata, progress):
    from huggingface_hub import HfFileSystem
    from pyarrow import parquet

    file_system = HfFileSystem(token=False)
    with ExitStack() as resources:
        files = []
        for entry in sorted(metadata["files"], key=lambda item: item["path"]):
            path = entry["path"]
            remote_path = f'datasets/{metadata["repo"]}@{metadata["revision"]}/{path}'
            progress(f"opening Wikipedia parquet metadata: {path}")
            handle = resources.enter_context(file_system.open(remote_path, "rb", block_size=4 << 20, cache_type="readahead"))
            if handle.size != entry["bytes"]:
                raise ValueError(f"Wrong size for the pinned Wikipedia parquet file: {path}")
            data = parquet.ParquetFile(handle)
            order = sorted(range(data.num_row_groups), key=lambda group: seeded_key("wiki_row_group", f"{path}:{group}"))
            files.append((path, data, iter(order)))
        yield from pending_group_batches(files, tuple(metadata["row_groups"]), progress)


def prepare(old_dir, output, source_spec, eval_paths, report_path, sample_path):
    legacy, tokenizer = verify_legacy_inputs(old_dir)
    metadata = wiki_metadata(source_spec)
    find_conflict, identities = build_prompt_screen(eval_paths)
    build = CorpusBuild()
    build.evaluation_paths = identities
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with build_lock(output):
        prepared_path = output / "prepared.json"
        if prepared_path.exists():
            raise FileExistsError(f"Already prepared a corpus at {output}; apply the review before freezing")

        def progress(message):
            print(message, flush=True)

        add_old_holdouts(build, old_dir, progress)
        selected = None
        tokens = 0
        request = json.dumps({"source": metadata, "screens": identities, "tokenizer": legacy["tokenizer_sha256"]},
                             ensure_ascii=False, sort_keys=True)
        with closing(sqlite3.connect(output / "acquisition.sqlite3")) as cache:
            cache.execute("CREATE TABLE IF NOT EXISTS identity (request TEXT NOT NULL)")
            cache.execute("CREATE TABLE IF NOT EXISTS batches (ordinal INTEGER PRIMARY KEY, entry TEXT NOT NULL, records TEXT NOT NULL)")
            recorded = cache.execute("SELECT request FROM identity").fetchone()
            if recorded is not None and recorded[0] != request:
                raise ValueError("Existing journal uses a different corpus, tokenizer or evaluation screen")
            if recorded is None:
                with cache:
                    cache.execute("INSERT INTO identity VALUES (?)", (request,))

            def consume(entry, records):
                metadata["row_groups"].append({**entry, **ingest_wiki(build, records, find_conflict)})
                if len(metadata["row_groups"]) % len(metadata["files"]) == 0:
                    return build.select_and_encode(tokenizer, progress)
                return None

            for entry_json, records_json in cache.execute("SELECT entry, records FROM batches ORDER BY ordinal"):
                state = consume(json.loads(entry_json), json.loads(records_json))
                if state is not None:
                    selected, discarded, groups, tokens = state
            if metadata["row_groups"]:
                progress(f"replayed {len(metadata['row_groups']):,} acquired row groups; current training tokens {tokens:,}")

            if tokens < MIN_TRAIN_TOKENS:
                with closing(read_pending_groups(metadata, progress)) as batches:
                    for ordinal, entry, records in batches:
                        with cache:
                            cache.execute("INSERT INTO batches VALUES (?, ?, ?)",
                                          (ordinal, json.dumps(entry), json.dumps(records, ensure_ascii=True)))
                        state = consume(entry, records)
                        if state is not None:
                            selected, discarded, groups, tokens = state
                            write_json(report_path, {"status": "acquiring", "wiki": metadata, "train_tokens": tokens,
                                                     "target_train_tokens": MIN_TRAIN_TOKENS})
                            if tokens >= MIN_TRAIN_TOKENS:
                                break

        selected, discarded, groups, tokens = build.select_and_encode(tokenizer, progress)
        if tokens < MIN_TRAIN_TOKENS or any(not selected[split] for split in SPLITS):
            raise RuntimeError(f"Insufficient distinct Wikipedia training stream: {tokens:,} tokens")
        stats, names = write_splits(output, selected, progress)
        if stats["train"]["tokens"] != tokens:
            raise RuntimeError("Packed training token count differs from the selected document count")
        frozen_stats = {split: {key: value for key, value in stats[split].items() if key != "quality_samples"}
                        for split in SPLITS}
        checks = build.data_report(discarded, groups)
        checks["quality_rules"] = QUALITY_RULES
        deduplication = {"exact": "SHA-256 on normalized alphanumeric content", "near": "Bottom 16 character 5-gram hashes, at least 2 shared, shingle length ratio >=0.8, verified exact Jaccard >=0.8",
                         "maximum_hash_postings": MAX_POSTINGS, "limitation": "Approximate candidate generation can miss pairs; frequent keys are suppressed.",
                         "holdout_scope": "Old validation/test original and navigation/Wikipedia-body views share their old group."}
        shuffle = {"seed": SEED, "key": "SHA-256 UTF-8 of '42:shuffle:<document_id>'", "order": "ascending digest then id",
                   "packing": "Documents written in global per-split order to JSONL and little-endian uint16 with one EOS per document."}
        quality = stratified_samples(selected)
        if len(quality) != 32:
            raise RuntimeError("Quality review requires 32 actual retained training examples")
        report = {**checks, "status": "awaiting_quality_review", "wiki": metadata, "train_tokens": tokens, "target_train_tokens": MIN_TRAIN_TOKENS,
                  "read_documents": sum(group["documents"] for group in metadata["row_groups"]),
                  "raw_text_utf8_bytes": sum(group["raw_text_utf8_bytes"] for group in metadata["row_groups"]),
                  "quality_retained_utf8_bytes": sum(group["quality_retained_utf8_bytes"] for group in metadata["row_groups"]),
                  "network_bytes": None, "new_stream_composition": {"wikipedia": 1.0, "original_web": 0.0},
                  "deduplication": deduplication, "shuffle": shuffle, "splits": frozen_stats,
                  "quality_review_samples": str(sample_path.resolve())}
        manifest = {"schema_version": 1, "corpus_id": CORPUS_ID, "frozen": True, "dtype": "<u2",
                    "tokenizer_path": legacy["tokenizer_path"], "tokenizer_sha256": legacy["tokenizer_sha256"],
                    "vocab_size": legacy["vocab_size"], "special_token_ids": SPECIAL_IDS, "splits": frozen_stats,
                    "files": {"tokenizer.json": legacy["files"]["tokenizer.json"],
                              **{name: file_identity(output / f"{name}.partial") for name in names}},
                    "sources": {"wiki": metadata}, "seed": SEED, "shuffle": shuffle, "deduplication": deduplication,
                    "split_percentages": [98, 1, 1], "filter_settings": QUALITY_RULES, "evaluation_screen": identities,
                    "quality_report_path": str(report_path.resolve())}
        write_json(sample_path, {"seed": QUALITY_SAMPLE_SEED, "method": "Each source's length thirds, then a seeded sample round-robin.", "cases": quality})
        write_json(report_path, report)
        write_json(prepared_path, {"proposed_manifest": manifest, "parts": names, "state": "awaiting_review"})
        progress(f"Prepared {tokens:,} training tokens; awaiting independent quality review")


def publish(output, review_marker):
    with build_lock(output):
        if not review_marker.is_file():
            raise FileNotFoundError(f"Missing independent quality review marker: {review_marker}")
        state = json.loads((output / "prepared.json").read_text(encoding="utf-8"))
        manifest = state["proposed_manifest"]
        for name in state["parts"]:
            part = output / f"{name}.partial"
            actual = part if part.exists() else output / name
            if file_identity(actual) != manifest["files"][name]:
                raise ValueError(f"Prepared file identity changed: {actual}")
        if file_identity(Path(manifest["tokenizer_path"])) != manifest["files"]["tokenizer.json"]:
            raise ValueError("Frozen tokenizer file identity changed before publishing")
        for name in state["parts"]:
            part = output / f"{name}.partial"
            if part.exists():
                part.replace(output / name)
        manifest["quality_review_marker"] = str(review_marker.resolve())
        write_json(output / "manifest.json", manifest)
        report_path = Path(manifest["quality_report_path"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        write_json(report_path, {**report, "status": "frozen", "quality_review_marker": str(review_marker.resolve())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--old-data", type=Path, default=OLD_DATA)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--samples", type=Path, default=SAMPLES)
    parser.add_argument("--eval-prompts", type=Path, action="append")
    parser.add_argument("--review-ready", type=Path, default=REVIEW_MARKER)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.old_data, args.output, args.source, args.eval_prompts or EVAL_PROMPTS, args.report, args.samples)
    else:
        publish(args.output, args.review_ready)


if __name__ == "__main__":
    main()
