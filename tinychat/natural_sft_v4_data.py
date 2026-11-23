"""Build local candidates and an independent-review pilot for natural Chinese SFT v4."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

from .chat_format import check_special_ids
from .comparison_sft_data import ConflictIndex
from .natural_sft_v4_candidates import (
    CATEGORIES,
    MAX_TOKENS,
    ROOT,
    TOKENIZER,
    belle_rows,
    belle_stage1_rows,
    cmrc_candidate,
    cmrc_rows,
    dialogue_candidate,
    json_rows,
    moss_rows,
)
from .sft_data import conversation_key
from .stage1_data import normalized_content


SEED = 104729
OUTPUT = ROOT / "data/natural_sft_v4_candidates_round2"
REPORT = ROOT / "reports/natural_sft_v4"
REPORT_PREFIX = "data_round2"
ROUND_ONE_COUNTS = REPORT / "data_candidate_counts.json"
QUOTAS = {
    "train": dict(zip(CATEGORIES, (7000, 5000, 5000, 3000))),
    "validation": dict(zip(CATEGORIES, (175, 125, 125, 75))),
    "test": dict(zip(CATEGORIES, (175, 125, 125, 75))),
}
SOURCE_LOADERS = {
    "moss_helpfulness": (moss_rows, dialogue_candidate),
    "belle_cache": (belle_rows, dialogue_candidate),
    "belle_stage1": (belle_stage1_rows, dialogue_candidate),
    "cmrc_train": (lambda: cmrc_rows("train"), cmrc_candidate),
    "cmrc_validation": (lambda: cmrc_rows("validation"), cmrc_candidate),
}
FROZEN_INPUTS = ROOT / "evaluation/natural_sft_v3/prompts_for_overlap.jsonl"
HISTORIC_HOLDOUTS = tuple(
    ROOT / f"data/{corpus}/{split}.jsonl"
    for corpus in ("sft_stage1_v1", "sft_verified_v2") for split in ("validation", "test")
)


def length_band(tokens):
    return "1-96" if tokens <= 96 else "97-192" if tokens <= 192 else "193-320" if tokens <= 320 else "321-512"


def pilot_samples(candidates, *, per_category=20, seed=SEED):
    """Round-robin sources first, then length bands within each source."""
    rng = random.Random(seed)
    strata = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in candidates:
        strata[row["category"]][row["source"]][length_band(row["token_count"])].append(row)
    result = []
    counts = Counter()
    coverage = defaultdict(Counter)
    for category in CATEGORIES:
        sources = strata[category]
        band_order = {}
        for source, bands in sorted(sources.items()):
            band_order[source] = sorted(bands)
            rng.shuffle(band_order[source])
            for rows in bands.values():
                rows.sort(key=lambda row: row["id"])
                rng.shuffle(rows)
        source_order = sorted(sources)
        rng.shuffle(source_order)
        groups = set()
        pointers = Counter()
        while counts[category] < per_category:
            moved = False
            for source in source_order:
                bands = band_order[source]
                for _ in bands:
                    index = pointers[source] % len(bands)
                    pointers[source] += 1
                    band = bands[index]
                    rows = sources[source][band]
                    while rows and rows[-1]["group_id"] in groups:
                        rows.pop()
                    if not rows:
                        continue
                    row = rows.pop()
                    groups.add(row["group_id"])
                    result.append(row)
                    counts[category] += 1
                    coverage[category][f"{source}/{band}"] += 1
                    moved = True
                    break
                if counts[category] == per_category:
                    break
            if not moved:
                break
    return result, {category: counts[category] for category in CATEGORIES}, dict(coverage)


def quota_upper_bounds(candidates):
    required = {name: sum(split[name] for split in QUOTAS.values()) for name in CATEGORIES}
    groups = defaultdict(set)
    rows = Counter()
    non_cmrc_grounded = set()
    non_cmrc_rows = 0
    for row in candidates:
        category = row["category"]
        groups[category].add(row["group_id"])
        rows[category] += 1
        if category == "grounded" and not row["source"].startswith("cmrc_"):
            non_cmrc_grounded.add(row["group_id"])
            non_cmrc_rows += 1
    upper = {category: len(groups[category]) for category in CATEGORIES}
    upper["grounded"] = min(upper["grounded"], len(non_cmrc_grounded) + 3_000 + 125 + 125)
    return {
        "required_all_splits": required,
        "raw_candidates": {category: rows[category] for category in CATEGORIES},
        "distinct_source_groups": {category: len(groups[category]) for category in CATEGORIES},
        "one_per_source_group_bound_with_cmrc_cap": upper,
        "raw_count_deficit_all_splits": {category: max(0, required[category] - rows[category]) for category in CATEGORIES},
        "one_per_source_group_deficit_all_splits_with_cmrc_cap": {
            category: max(0, required[category] - upper[category]) for category in CATEGORIES
        },
        "non_cmrc_grounded_rows": non_cmrc_rows,
        "non_cmrc_grounded_distinct_source_groups": len(non_cmrc_grounded),
        "grounded_train_deficit_given_raw_candidates_and_hard_cap": max(
            0, QUOTAS["train"]["grounded"] - non_cmrc_rows - min(3_000, rows["grounded"] - non_cmrc_rows),
        ),
        "grounded_train_minimum_non_cmrc_required": 2_000,
        "grounded_train_deficit_at_cmrc_hard_cap_even_if_enough_cmrc": max(0, 2_000 - len(non_cmrc_grounded)),
        "grounded_train_deficit_at_cmrc_target_even_if_enough_cmrc": max(0, 3_000 - len(non_cmrc_grounded)),
        "cmrc_train_hard_cap": 3_000,
        "cmrc_train_target": 2_000,
    }


def write_report(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def summarize(rows, outcomes):
    accepted = Counter(row["source"] for row in rows)
    result = {}
    for source, reasons in sorted(outcomes.items()):
        selected = [row for row in rows if row["source"] == source]
        result[source] = {
            "scanned": sum(reasons.values()),
            "accepted_pending_semantic_review": accepted[source],
            "rejections": {reason: count for reason, count in reasons.most_common() if reason != "accepted"},
            "categories": Counter(row["category"] for row in selected),
            "distinct_source_groups": len({row["source_group_id"] for row in selected}),
            "families": Counter(row["task_family"] for row in selected),
            "length_bands": Counter(length_band(row["token_count"]) for row in selected),
            "selected_turns": Counter(len(row["messages"]) // 2 for row in selected),
            "max_tokens": max((row["token_count"] for row in selected), default=None),
        }
    return result


def user_keys(messages):
    keys = [normalized_content(message["content"]) for message in messages if message["role"] == "user"]
    return [*keys, "|".join(keys)] if len(keys) > 1 else keys


def input_index(paths):
    index = ConflictIndex(containment=True)
    inspected = {}
    for path in paths:
        inspected[str(path)] = 0
        for row in json_rows(path):
            for key in user_keys(row["messages"]):
                index.add(key)
            inspected[str(path)] += 1
        if inspected[str(path)] == 0:
            raise ValueError(f"Empty protected input file: {path}")
    return index, inspected


def build_pilot(output, report, *, report_prefix=REPORT_PREFIX, limit_per_input=None):
    candidate_path = output / "candidates.jsonl"
    counts_path = report / f"{report_prefix}_candidate_counts.json"
    pilot_path = report / f"{report_prefix}_pilot.json"
    for path in (candidate_path, counts_path, pilot_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite v4 pilot artifact: {path}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    check_special_ids(tokenizer)
    if tokenizer.get_vocab_size() != 16_384:
        raise ValueError("Expected the existing 16,384-token Stage 1 tokenizer")
    frozen, frozen_records = input_index((FROZEN_INPUTS,))
    heldout, heldout_records = input_index(HISTORIC_HOLDOUTS)

    outcomes = defaultdict(Counter)
    accepted = []
    cache_groups = set()
    seen_messages = set()
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    with candidate_path.open("x", encoding="utf-8", newline="\n") as handle:
        for source_name, (load, transform) in SOURCE_LOADERS.items():
            tot = 0
            source_rows = load()
            try:
                for source in source_rows:
                    if source_name == "belle_cache":
                        cache_groups.add(source.group_id)
                    if source_name == "belle_stage1" and source.group_id in cache_groups:
                        row, reason = None, "stage1_group_already_in_belle_cache"
                    elif source.name == "belle_stage1_multi":
                        row, reason = None, "stage1_window_may_omit_original_history"
                    else:
                        row, reason = transform(tokenizer, source)
                    if row is not None:
                        keys = user_keys(row["messages"])
                        if any(frozen.match(key) for key in keys):
                            row, reason = None, "frozen_natural_input_conflict"
                        elif any(heldout.match(key) for key in keys):
                            row, reason = None, "historic_holdout_input_conflict"
                    if row is not None:
                        key = conversation_key(row["messages"])
                        if key in seen_messages:
                            row, reason = None, "duplicate_selected_conversation"
                        else:
                            seen_messages.add(key)
                    outcomes[source.name][reason or "accepted"] += 1
                    if row is not None:
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        accepted.append(row)
                    tot += 1
                    if tot % 10_000 == 0:
                        print(f"[natural v4 candidate pilot] {source_name}: {tot:,} scanned", flush=True)
                    if limit_per_input is not None and tot >= limit_per_input:
                        break
            finally:
                source_rows.close()
            print(f"[natural v4 candidate pilot] {source_name}: {tot:,} scanned", flush=True)

    samples, pilot_counts, coverage = pilot_samples(accepted)
    status = "candidate_pilot_pending_independent_semantic_review"
    bounds = quota_upper_bounds(accepted)
    prior = json.loads(ROUND_ONE_COUNTS.read_text(encoding="utf-8"))["quota_upper_bounds"] if ROUND_ONE_COUNTS.is_file() else None
    comparison = None if prior is None else {
        "prior_report": str(ROUND_ONE_COUNTS),
        "round_one_candidates": prior["raw_candidates"],
        "round_two_candidates": bounds["raw_candidates"],
        "round_one_distinct_source_groups": prior["distinct_source_groups"],
        "round_two_distinct_source_groups": bounds["distinct_source_groups"],
        "candidate_count_difference": {
            category: bounds["raw_candidates"][category] - prior["raw_candidates"][category] for category in CATEGORIES
        },
    }
    write_report(counts_path, {
        "status": status, "rows": len(accepted), "sources": summarize(accepted, outcomes),
        "candidate_file": str(candidate_path), "limit_per_input": limit_per_input,
        "chat_format": {"final_eos": False, "max_tokens_including_roles_and_system": MAX_TOKENS},
        "quota_upper_bounds": bounds, "round_comparison": comparison,
        "protected_inputs": {
            "natural_frozen_records": frozen_records, "historic_holdout_records": heldout_records,
            "scope": "Normalized user messages and user-only trajectories. No rubric or answer matching.",
        },
        "scope": "Complete source conversations or leading whole-message prefixes, with normalized exact conversation deduplication. No accepted pool has a semantic PASS.",
        "remaining_semantic_risks": [
            "Natural relevance and social appropriateness need review even when prompt, action, and topic cues align.",
            "Generative multi-turn specifics need review for the original objective and prior constraints.",
            "Named-entity and local relation checks do not prove every pronoun resolution or gold answer.",
            "Original source groups may still have near-duplicates or shared documents across sources.",
        ],
        "deferred": ["cross-source near-duplicate consolidation", "group split", "final sample/approval"],
    })
    write_report(pilot_path, {
        "status": status, "per_category_requested": 20, "seed": SEED,
        "stratification": "Round-robin source and length-band coverage, at most one draw per source group per category.",
        "counts": pilot_counts, "source_length_coverage": coverage,
        "shortages": {category: 20 - count for category, count in pilot_counts.items() if count < 20},
        "samples": samples,
    })
    print(json.dumps({"candidates": len(accepted), "pilot": pilot_counts, "report": str(counts_path)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--report-prefix", default=REPORT_PREFIX)
    parser.add_argument("--limit-per-input", type=int, help="Bound a diagnostic run per loader")
    args = parser.parse_args()
    if args.limit_per_input is not None and args.limit_per_input < 1:
        parser.error("--limit-per-input must be positive")
    if not args.report_prefix or Path(args.report_prefix).name != args.report_prefix:
        parser.error("--report-prefix must be a plain filename prefix")
    build_pilot(args.output, args.report, report_prefix=args.report_prefix, limit_per_input=args.limit_per_input)


if __name__ == "__main__":
    main()
