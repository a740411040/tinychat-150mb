"""Build a source-traceable structural pool for later semantic adjudication."""

import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from itertools import combinations, islice

from pyarrow.parquet import ParquetFile
from tokenizers import Tokenizer

from .chat_format import IGNORE_LABEL, SPECIAL_IDS, check_special_ids, contains_reserved, encode_conversation
from .natural_sft_source_candidates import MOSS_PAIR, SOURCES
from .natural_sft_v4_candidates import (
    BELLE, BELLE_REVISIONS, BELLE_STAGE1, BELLE_USAGE, CMRC, MAX_TOKENS, MOSS,
    ROOT, TOKENIZER, json_rows, msg, relative,
)
from .stage1_data import normalized_content


OUTPUT = ROOT / "data/natural_sft_v4_semantic_pool"
REPORT = ROOT / "reports/natural_sft_v4"
SMOL = ROOT / "data/public_sources/natural_sft_v4/smoltalk-chinese/everyday.parquet"
SMOL_IDENTITY = ("opencsg/smoltalk-chinese", "5edf5d0d7b06794755cf78b274a150a4efe1c9d4", "apache-2.0")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LENGTH_RANGES = ((1, 64), (65, 128), (129, 256), (257, 384), (385, MAX_TOKENS))
CMRC_PREFIX = "请根据材料回答问题。\n\n材料："
CMRC_SEPARATOR = "\n\n问题："


@dataclass
class SourceInput:
    name: str
    provenance: dict
    messages: object
    transform: dict
    origin: tuple | None = None
    document: str | None = None
    reject: str | None = None


def parse_moss_exact(text, source_turns):
    if not isinstance(text, str):
        return None, [], "malformed_moss_markup"
    messages, spans = [], []
    cursor = 0
    while match := MOSS_PAIR.match(text, cursor):
        for index, role in enumerate(("user", "assistant"), 1):
            start, end = match.span(index)
            # The fixed source framing inserts one space after the role colon.
            if text[start:start + 1] == " ":
                start += 1
            content = text[start:end]
            if "<eoh>" in content or "<eoa>" in content:
                return None, [], "malformed_moss_markup"
            messages.append(msg(role, content))
            spans.append([start, end])
        cursor = match.end()
    if not messages or text[cursor:].strip() or type(source_turns) is not int or source_turns != len(messages) // 2:
        return None, [], "malformed_moss_markup"
    return messages, spans, None


def moss_rows():
    path = MOSS / "zh_helpfulness.prefix64m.jsonl"
    metadata_path = path.with_name("zh_helpfulness.prefix64m.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    identity = tuple(metadata[key] for key in ("repo", "revision", "license_from_card", "source_file"))
    byte_range = [metadata["range_start"], metadata["range_end_inclusive"]]
    if identity != SOURCES["moss_helpfulness"] or byte_range != [0, 67_108_863]:
        raise ValueError(f"Unexpected MOSS source metadata in {metadata_path}")
    repo, revision, license_name, source_file = identity
    for line, row in enumerate(json_rows(path), 1):
        row_id = row.get("id")
        if type(row_id) not in (str, int):
            row_id = None
        messages, spans, reject = parse_moss_exact(row.get("plain_text"), row.get("num_turns"))
        provenance = {
            "repo": repo, "revision": revision, "license": license_name, "source_file": source_file,
            "license_record": relative(MOSS / "README.md"), "local_file": relative(path),
            "acquisition_record": relative(metadata_path), "original_byte_range_inclusive": byte_range,
            "source_selection": metadata["selection_bias"], "source_row_id": row_id, "jsonl_line_1_based": line,
        }
        transform = {
            "type": "moss_plain_text_role_framing", "role_map": {"Human": "user", "MOSS": "assistant"},
            "removed_framing": "role marker, colon, one immediate ASCII separator space, turn delimiter, inter-turn whitespace",
            "content_chars_zero_based_half_open": spans, "source_num_turns": row.get("num_turns"),
            "source_prefix_metadata": row.get("prefix"),
        }
        origin = ("moss_id", row_id) if row_id is not None else ("moss_line", line)
        yield SourceInput("moss_helpfulness", provenance, messages, transform, origin, reject=reject)


def belle_rows():
    for filename, position, bucket in (
        ("candidates.sqlite3", "ordinal_1_based", "belle_cache"),
        ("candidates_spans.sqlite3", "source_byte_offset", "belle_spans"),
    ):
        path = BELLE / filename
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
            for repo, revision, index, family, payload in database.execute(
                "SELECT repo, revision, ordinal, family, payload FROM candidates ORDER BY repo, ordinal"
            ):
                expected_revision, source_file, kind = BELLE_REVISIONS[repo]
                if revision != expected_revision:
                    raise ValueError(f"Unexpected BELLE revision for {repo}: {revision}")
                provenance = {
                    "repo": repo, "revision": revision, "source_file": source_file, "usage": BELLE_USAGE,
                    "license_record": relative(BELLE / "manifest.prepared.json"), "local_file": relative(path),
                    "cache_table": "candidates", "cache_ordinal": index, position: index, "old_cache_family": family,
                    "source_selection": "Previously screened cache of normalized complete source conversations",
                }
                yield SourceInput(f"{bucket}_{kind}", provenance, json.loads(payload), {
                    "type": "copy_cached_normalized_messages",
                    "historical_normalization": "tinychat.sft_data.parse_conversation and clean_turn",
                })


def belle_stage1_rows():
    path = BELLE_STAGE1 / "train.jsonl"
    for line, row in enumerate(json_rows(path), 1):
        repo = row["source"]
        revision, source_file, kind = BELLE_REVISIONS[repo]
        provenance = {
            "repo": repo, "revision": revision, "source_file": source_file, "usage": BELLE_USAGE,
            "license_record": relative(BELLE_STAGE1 / "manifest.json"), "local_file": relative(path),
            "old_train_id": row["id"], "old_group_id": row["group_id"], "jsonl_line_1_based": line,
            "source_selection": "Legacy Stage 1 train; only original single-turn examples can contribute",
        }
        reject = None
        if kind != "single":
            reject = "legacy_multi_window_history_unavailable"
        elif not isinstance(row["messages"], list) or len(row["messages"]) != 2:
            reject = "legacy_single_not_one_pair"
        yield SourceInput("belle_stage1", provenance, row["messages"] if reject is None else None, {
            "type": "copy_legacy_normalized_single_pair",
            "historical_normalization": "tinychat.sft_data.parse_conversation and clean_turn",
        }, reject=reject)


def parquet_rows(path, *, columns=None):
    for batch in ParquetFile(path).iter_batches(batch_size=256, columns=columns):
        yield from batch.to_pylist()


def cmrc_render(context, question, annotations):
    if not isinstance(context, str) or not context.strip() or not isinstance(question, str) or not question.strip():
        return None, {}, "invalid_cmrc_text"
    texts = annotations.get("text") if isinstance(annotations, dict) else None
    starts = annotations.get("answer_start") if isinstance(annotations, dict) else None
    if not isinstance(texts, list) or not isinstance(starts, list) or not texts or len(texts) != len(starts):
        return None, {}, "invalid_cmrc_annotation_shape"
    verified = [
        index for index, (gold, start) in enumerate(zip(texts, starts))
        if isinstance(gold, str) and gold.strip() and type(start) is int and start >= 0
        and context[start:start + len(gold)] == gold
    ]
    if not verified:
        return None, {}, "cmrc_no_exact_answer_offset"
    annotation = verified[0]
    gold, start = texts[annotation], starts[annotation]
    transform = {
        "type": "cmrc_verbatim_full_context_and_question_wrapper",
        "passage_chars_zero_based_half_open": [0, len(context)],
        "answer_chars_zero_based_half_open": [start, start + len(gold)],
        "prompt_prefix": CMRC_PREFIX, "prompt_context_question_separator": CMRC_SEPARATOR,
        "source_annotations": annotations, "offset_exact_annotation_indices_zero_based": verified,
        "selected_annotation_index_zero_based": annotation,
    }
    return [msg("user", CMRC_PREFIX + context + CMRC_SEPARATOR + question), msg("assistant", gold)], transform, None


def cmrc_rows(split):
    name = f"cmrc_{split}"
    repo, revision, license_name, filename = SOURCES[name]
    path = CMRC / filename
    for index, row in enumerate(parquet_rows(path, columns=["id", "context", "question", "answers"]), 1):
        document_id, separator, _ = row["id"].rpartition("_QUERY_")
        if not separator or not document_id:
            raise ValueError(f"Unexpected CMRC record ID at {path}:{index}")
        provenance = {
            "repo": repo, "revision": revision, "license": license_name, "source_file": filename,
            "license_record": relative(CMRC / "README.md"), "local_file": relative(path),
            "source_row_1_based": index, "source_row_id": row["id"],
            "source_split": split, "source_document_id": document_id,
        }
        messages, transform, reject = cmrc_render(row["context"], row["question"], row["answers"])
        yield SourceInput(name, provenance, messages, transform, ("cmrc_document", document_id), row["context"], reject)


def smol_rows():
    metadata_path = SMOL.with_suffix(".parquet.source.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    identity = tuple(metadata[key] for key in ("repo", "revision", "license_metadata"))
    if identity != SMOL_IDENTITY or metadata["file"] != SMOL.name:
        raise ValueError(f"Unexpected SmolTalk source metadata in {metadata_path}")
    for index, row in enumerate(parquet_rows(SMOL), 1):
        repo, revision, license_name = identity
        provenance = {
            "repo": repo, "revision": revision, "license": license_name, "source_file": SMOL.name,
            "license_record": relative(metadata_path), "source_card_url": metadata["card_url"],
            "additional_source_terms": metadata["additional_source_terms"], "local_file": relative(SMOL),
            "source_row_1_based": index, "source_type": metadata["source_type"],
        }
        transform = {
            "type": "copy_conversations_role_and_content", "source_num_turns": row["n_turn"],
            "source_system_prompt_key": row["system_prompt_key"], "source_magpie_model": row["magpie_model"],
        }
        yield SourceInput("smoltalk_everyday", provenance, row["conversations"], transform, ("smol_row", index))


def structural_reject(messages):
    if not isinstance(messages, list) or not messages or any(not isinstance(item, dict) for item in messages):
        return "invalid_message_array"
    roles = [item.get("role") for item in messages]
    start = int(roles[0] == "system")
    pairs = (len(roles) - start) // 2
    if not pairs or roles[start:] != ["user", "assistant"] * pairs:
        return "invalid_role_sequence"
    for item in messages:
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            return "empty_or_non_text_content"
        if contains_reserved(content):
            return "reserved_chat_token_in_content"
        if CONTROL.search(content):
            return "invalid_control_character"
    return None


def prefix_options(tokenizer, messages):
    """Count complete prefixes using the frozen formatter and its single BOS."""
    start = int(messages[0]["role"] == "system")
    options = []
    tokens, targets = 0, 0
    for end in range(start + 2, len(messages) + 1, 2):
        first_pair = end == start + 2
        items = messages[:end] if first_pair else messages[end - 2:end]
        encoded = encode_conversation(tokenizer, items, include_system=first_pair, final_eos=False)
        tokens += len(encoded["input_ids"]) - int(not first_pair)
        targets += sum(label != IGNORE_LABEL for label in encoded["labels"])
        if tokens > MAX_TOKENS:
            break
        options.append({
            "turns": (end - start) // 2, "message_count": end, "tokens": tokens, "supervised_tokens": targets,
        })
    return options


def origin_key(source):
    if source.origin is not None:
        return source.origin
    first_pair = tuple(normalized_content(item["content"]) for item in source.messages[:2])
    # BELLE cache group identity follows the historical normalized first pair.
    # Fall back to exact text when normalization would discard an entire turn.
    if any(not item for item in first_pair):
        first_pair = tuple(item["content"] for item in source.messages[:2])
    return ("belle_first_pair", *first_pair)


def first_question_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()) or text


class Groups:
    """Unify source groups and matching first user turns before token filtering."""

    def __init__(self):
        self.parents = []
        self.owners = {}

    def root(self, index):
        while index != self.parents[index]:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def add(self, *keys):
        index = len(self.parents)
        self.parents.append(index)
        for key in keys:
            owner = self.owners.setdefault(key, index)
            previous, current = self.root(owner), self.root(index)
            self.parents[max(previous, current)] = min(previous, current)
        return index


def message_key(messages):
    return tuple((item["role"], item["content"]) for item in messages)


def length_band(tokens):
    return next(f"{low}-{high}" for low, high in LENGTH_RANGES if low <= tokens <= high)


def summarize(records):
    retained_groups = defaultdict(set)
    structural_groups = defaultdict(set)
    all_options = defaultdict(set)
    for row in records:
        group = row["group_id"]
        retained_groups["any_complete"].add(group)
        retained_groups[f'{row["prefix_options"][-1]["turns"]}_turns_max'].add(group)
        for option in row["prefix_options"]:
            turns = option["turns"]
            structural_groups["one_turn" if turns == 1 else "two_or_more_turns"].add(group)
            all_options[length_band(option["tokens"])].add(group)
    return {
        "unique_prefixes": len(records),
        "max_prefix_one_turn": sum(len(row["prefix_options"]) == 1 for row in records),
        "max_prefix_two_or_more_turns": sum(len(row["prefix_options"]) > 1 for row in records),
        "longest_prefix_length_distribution": dict(Counter(length_band(row["token_count"]) for row in records)),
        "unique_pool_groups": len(retained_groups.get("any_complete", ())),
        "groups_by_max_prefix_turns_nonexclusive": {
            key: len(groups) for key, groups in retained_groups.items() if key != "any_complete"
        },
        "groups_with_prefix_option": {key: len(structural_groups.get(key, ())) for key in ("one_turn", "two_or_more_turns")},
        "groups_with_prefix_in_length_band_nonexclusive": {key: len(groups) for key, groups in all_options.items()},
    }


def overlap(groups_by_source):
    return [
        {"sources": [left, right], "groups": len(groups_by_source[left] & groups_by_source[right])}
        for left, right in combinations(sorted(groups_by_source), 2)
        if not groups_by_source[left].isdisjoint(groups_by_source[right])
    ]


def build_pool(tokenizer, *, loaders=None, limit_per_input=None, progress=False):
    loaders = loaders or (
        moss_rows, belle_rows, belle_stage1_rows, lambda: cmrc_rows("train"),
        lambda: cmrc_rows("validation"), smol_rows,
    )
    counts, origins_by_source = defaultdict(Counter), defaultdict(set)
    rejects, cmrc_offset_notes = defaultdict(Counter), defaultdict(Counter)
    groups, origin_ids = Groups(), {}
    pool, membership = {}, defaultdict(list)
    for load in loaders:
        rows = load()
        if limit_per_input is not None:
            rows = islice(rows, limit_per_input)
        for source in rows:
            name = source.name
            counts[name]["raw_rows_processed"] += 1
            reject = source.reject or structural_reject(source.messages)
            origin = source.origin if reject else origin_key(source)
            source_group = None
            if origin is not None:
                source_group = origin_ids.setdefault(origin, f"original-{len(origin_ids) + 1:06}")
                origins_by_source[name].add(source_group)
            if reject:
                rejects[name][reject] += 1
                continue
            first_user = next(item["content"] for item in source.messages if item["role"] == "user")
            keys = [("origin", source_group), ("first_user", first_question_key(first_user))]
            if source.document is not None:
                keys.append(("cmrc_exact_context", source.document))
                annotations = source.transform["source_annotations"]["text"]
                indices = source.transform["offset_exact_annotation_indices_zero_based"]
                cmrc_offset_notes[name]["rows_with_unmatched_annotations"] += len(indices) < len(annotations)
                cmrc_offset_notes[name]["rows_with_distinct_exact_answer_texts"] += len({annotations[i] for i in indices}) > 1
            index = groups.add(*keys)
            membership[name].append(index)
            options = prefix_options(tokenizer, source.messages)
            if not options:
                rejects[name]["first_complete_pair_over_context"] += 1
                continue
            counts[name]["rows_with_prefix_before_exact_dedup"] += 1
            max_option = options[-1]
            messages = source.messages[:max_option["message_count"]]
            source_info = {
                "input": name, "source_group_id": source_group, **source.provenance,
                "original_message_count": len(source.messages),
                "message_range_zero_based_half_open": [0, max_option["message_count"]], "transform": source.transform,
            }
            key = message_key(messages)
            entry = pool.get(key)
            if entry is not None:
                entry["sources"].append(source_info)
                entry["group_nodes"].append(index)
                counts[name]["duplicate_prefix_aliases"] += 1
                continue
            pool[key] = {
                "id": f"prefix-{len(pool) + 1:06}", "group_nodes": [index],
                "messages": [msg(item["role"], item["content"]) for item in messages],
                "prefix_options": options, "token_count": max_option["tokens"], "sources": [source_info],
            }
            counts[name]["new_exact_prefixes"] += 1
        if progress:
            print(f"Processed {', '.join(counts)} ({sum(value['raw_rows_processed'] for value in counts.values()):,} rows)", flush=True)

    by_source, retained_origins = defaultdict(list), defaultdict(set)
    final_membership = {name: {groups.root(index) for index in indices} for name, indices in membership.items()}
    retained_groups = defaultdict(set)
    duplicate_overlap = Counter()
    for row in pool.values():
        nodes = row.pop("group_nodes")
        root = groups.root(nodes[0])
        if any(groups.root(node) != root for node in nodes):
            raise ValueError(f"Exact duplicate prefix spans separate groups: {row['id']}")
        row["group_id"] = f"pool-{root + 1:06}"
        source_names = sorted({source["input"] for source in row["sources"]})
        duplicate_overlap.update(combinations(source_names, 2))
        for name in source_names:
            by_source[name].append(row)
            retained_groups[name].add(root)
        for source_info in row["sources"]:
            retained_origins[source_info["input"]].add(source_info["source_group_id"])

    records = list(pool.values())
    result = {
        "status": "structural_pool_only",
        "chat_format": {"tokenizer": relative(TOKENIZER), "max_tokens": MAX_TOKENS, "final_eos": False},
        "grouping": [
            "Native MOSS row ID, CMRC document ID and SmolTalk row locator; BELLE historical normalized first pair.",
            "Transitive union across native groups, equal verbatim CMRC contexts, and normalized identical first user texts.",
            "Exact duplicate emitted role/content prefixes collapse to one entry with all source locators.",
            "Counts are structural upper bounds; each final pool group contributes at most one conservative slot.",
        ],
        "pending_gates": ["semantic eligibility", "global near-duplicate review", "frozen prompt-only and historical holdout isolation", "split assignment"],
        "totals": {
            "raw_rows_processed": sum(counter["raw_rows_processed"] for counter in counts.values()),
            "unique_original_groups_identified": len(origin_ids),
            "unique_original_groups_retained": len(set().union(*retained_origins.values())),
            "source_rows_with_prefix_before_exact_dedup": sum(counter["rows_with_prefix_before_exact_dedup"] for counter in counts.values()),
            "exact_duplicate_prefix_aliases": sum(counter["duplicate_prefix_aliases"] for counter in counts.values()),
            "rejections": dict(sum(rejects.values(), Counter())),
            **summarize(records),
        },
        "inputs": {
            name: {
                **{key: counts[name][key] for key in (
                    "raw_rows_processed", "rows_with_prefix_before_exact_dedup", "duplicate_prefix_aliases", "new_exact_prefixes",
                )},
                "original_groups_identified": len(origins_by_source[name]),
                "original_groups_retained": len(retained_origins[name]),
                "structurally_parsed_pool_groups_before_length_gate": len(final_membership.get(name, ())),
                "rejections": dict(rejects[name]),
                **summarize(by_source[name]),
                **({"cmrc_offset_notes_before_length_gate": dict(cmrc_offset_notes[name])} if name.startswith("cmrc_") else {}),
            }
            for name in counts
        },
        "source_overlap": {
            "original_groups_before_length_gate": overlap(origins_by_source),
            "pool_groups_before_length_gate": overlap(final_membership),
            "retained_pool_groups": overlap(retained_groups),
            "exact_prefixes_shared_between_inputs": [
                {"sources": list(names), "prefixes": count} for names, count in sorted(duplicate_overlap.items())
            ],
        },
    }
    return records, by_source, result


def example_entries(by_source):
    for name, rows in by_source.items():
        used_groups = set()
        for fraction in (1, 5, 9):
            start = fraction * len(rows) // 10
            row = next((rows[(start + i) % len(rows)] for i in range(len(rows))
                        if rows[(start + i) % len(rows)]["group_id"] not in used_groups), None)
            if row is None:
                break
            used_groups.add(row["group_id"])
            source = next(source for source in row["sources"] if source["input"] == name)
            yield row, source


def verify_example(row, source, original, tokenizer):
    size = len(row["messages"])
    transform = source["transform"]
    transform_type = transform["type"]
    if transform_type == "moss_plain_text_role_framing":
        text = original["plain_text"]
        spans = transform["content_chars_zero_based_half_open"]
        expected = [msg(("user", "assistant")[index % 2], text[start:end])
                    for index, (start, end) in enumerate(spans[:size])]
        if original["id"] != source["source_row_id"] or original["num_turns"] * 2 != len(spans):
            raise ValueError(f"MOSS source row mismatch for {row['id']}")
    elif transform_type == "cmrc_verbatim_full_context_and_question_wrapper":
        start, end = transform["answer_chars_zero_based_half_open"]
        context, question = original["context"], original["question"]
        expected = [
            msg("user", transform["prompt_prefix"] + context + transform["prompt_context_question_separator"] + question),
            msg("assistant", context[start:end]),
        ]
        if original["id"] != source["source_row_id"] or original["answers"] != transform["source_annotations"]:
            raise ValueError(f"CMRC source row mismatch for {row['id']}")
    elif transform_type == "copy_conversations_role_and_content":
        expected = original["conversations"][:size]
    elif transform_type == "copy_legacy_normalized_single_pair":
        expected = original["messages"][:size]
        if original["id"] != source["old_train_id"] or original["group_id"] != source["old_group_id"]:
            raise ValueError(f"Stage 1 source row mismatch for {row['id']}")
    elif transform_type == "copy_cached_normalized_messages":
        expected = original[:size]
    else:
        raise ValueError(f"Unknown source transform: {transform_type}")

    if message_key(row["messages"]) != message_key(expected):
        raise ValueError(f"Retained messages differ from the located source for {row['id']}")
    for option in row["prefix_options"]:
        encoded = encode_conversation(tokenizer, expected[:option["message_count"]], final_eos=False)
        ids = encoded["input_ids"]
        if (
            len(ids) != option["tokens"] or sum(label != IGNORE_LABEL for label in encoded["labels"]) != option["supervised_tokens"]
            or SPECIAL_IDS["<|eos|>"] in ids or ids[-1] != SPECIAL_IDS["<|end_turn|>"]
        ):
            raise ValueError(f"Prefix encoding mismatch for {row['id']}")
    return {
        "id": row["id"], "group_id": row["group_id"], "source": source,
        "messages": row["messages"], "prefix_options": row["prefix_options"],
        "verification": "Relocated local source and matched role/content plus full formatter counts for every retained prefix.",
    }


def verify_examples(by_source, tokenizer):
    sample_files = defaultdict(list)
    for row, source in example_entries(by_source):
        sample_files[ROOT / source["local_file"]].append((row, source))
    results = defaultdict(list)
    for path, examples in sample_files.items():
        if path.suffix == ".sqlite3":
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as database:
                for row, source in examples:
                    record = database.execute(
                        "SELECT payload FROM candidates WHERE repo=? AND revision=? AND ordinal=?",
                        (source["repo"], source["revision"], source["cache_ordinal"]),
                    ).fetchone()
                    if record is None:
                        raise ValueError(f"Could not relocate source for {row['id']}")
                    results[source["input"]].append(verify_example(row, source, json.loads(record[0]), tokenizer))
            continue
        is_parquet = path.suffix == ".parquet"
        locator = "source_row_1_based" if is_parquet else "jsonl_line_1_based"
        positions = defaultdict(list)
        for row, source in examples:
            positions[source[locator]].append((row, source))
        originals = parquet_rows(path) if is_parquet else json_rows(path)
        for number, original in enumerate(islice(originals, max(positions)), 1):
            for row, source in positions.pop(number, ()):
                results[source["input"]].append(verify_example(row, source, original, tokenizer))
        if positions:
            raise ValueError(f"Could not relocate selected rows in {path}: {sorted(positions)}")
    if any(len(results[name]) != 3 for name in by_source):
        raise ValueError("Expected three source examples with distinct groups for every populated input")
    return dict(results)


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")


def main():
    pool_path, counts_path = OUTPUT / "pool.jsonl", OUTPUT / "counts.json"
    examples_path = REPORT / "semantic_pool_examples.json"
    for path in (pool_path, counts_path, examples_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    check_special_ids(tokenizer)
    if tokenizer.get_vocab_size() != 16_384:
        raise ValueError(f"Unexpected frozen tokenizer vocabulary size in {TOKENIZER}")
    records, by_source, counts = build_pool(tokenizer, progress=True)
    samples = verify_examples(by_source, tokenizer)
    counts["verified_source_examples"] = {name: len(records) for name, records in samples.items()}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    with pool_path.open("x", encoding="utf-8", newline="\n") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(examples_path, {"scope": "Source transform and prefix validation only", "inputs": samples})
    write_json(counts_path, counts)
    print(f"Wrote {len(records):,} unique prefixes to {pool_path}", flush=True)


if __name__ == "__main__":
    main()
