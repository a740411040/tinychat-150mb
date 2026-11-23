"""Deterministic stage sampling, with decisions saved separately from data."""
import argparse
from collections import Counter, defaultdict, deque
from itertools import product
import json
from pathlib import Path
import random

from scripts.natural_v4_staged_queue import DATA, REPORT, phase_path, result_rows, status as phase_status
from scripts.natural_v4_scale_queue import serialized_mutation
from scripts.natural_v4_rewrite_pilot import write_new


def balanced(rows, maximum):
    strata = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda item: item["id"]):
        source = row["sources"][0]["repo"]
        length = row.get("input_tokens", 512 - row["assistant_content_token_budget"])
        strata[source]["short" if length < 128 else "medium" if length < 320 else "long"].append(row)
    rng = random.Random(2026112303)
    sources = sorted(strata)
    rng.shuffle(sources)
    queues = []
    for source in sources:
        bands = sorted(strata[source])
        rng.shuffle(bands)
        band_queues = []
        for band in bands:
            rng.shuffle(strata[source][band])
            band_queues.append(deque(strata[source][band]))
        source_queue = deque()
        while any(band_queues):
            for queue in band_queues:
                if queue:
                    source_queue.append(queue.popleft())
        queues.append(source_queue)
    selected = []
    while len(selected) < maximum and any(queues):
        for queue in queues:
            if queue and len(selected) < maximum:
                selected.append(queue.popleft())
    return selected


def save_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def balanced_required(rows, maximum, requirements=()):
    """Cover predetermined defect families, never choose based on answer quality."""
    selected = []
    for name, predicate in requirements:
        if any(predicate(row) for row in selected):
            continue
        options = [row for row in rows if predicate(row) and row["id"] not in {x["id"] for x in selected}]
        if not options:
            raise ValueError(f"Missing prespecified validation stratum: {name}")
        selected.extend(balanced(options, 1))
    if len(selected) > maximum:
        raise ValueError("Required validation strata exceed category capacity")
    selected.extend(balanced([row for row in rows if row["id"] not in {x["id"] for x in selected}], maximum - len(selected)))
    return selected


def handoff_requirements(action, category):
    if category == "grounded":
        return [("clean_cmrc", lambda row: row["sources"][0]["repo"] == "hfl/cmrc2018")]
    if category != "instruction":
        return []
    if action in {"handoff_writer", "round2_writer"}:
        return [(kind, lambda row, kind=kind: kind in row.get("validation_challenges", []))
                for kind in ("full_rewrite", "simplify")]
    if action in {"semantic_review", "round2_review"}:
        return [("full_rewrite", lambda row: requested_rewrite_stratum(row, simplify=False)),
                ("simplify", lambda row: requested_rewrite_stratum(row, simplify=True))]
    return [(kind, lambda row, kind=kind: any(audit["task_kind"] == kind
                for audit in row["automated_checker_response"]["turn_audits"])) for kind in ("rewrite", "simplify")]


def balanced_round2_instruction(rows, maximum):
    """Reserve two separately eligible challenge rows before filling the quota."""
    ranked = []
    for name, predicate in handoff_requirements("round2_writer", "instruction"):
        candidates = [row for row in rows if predicate(row)]
        if not candidates:
            raise ValueError(f"Missing prespecified validation stratum: {name}")
        ranked.append(balanced(candidates, len(candidates)))
    pair = next(
        ((rewrite, simplify) for rewrite, simplify in product(*ranked) if rewrite["id"] != simplify["id"]),
        None,
    )
    if pair is None:
        raise ValueError("Round2 writer requires distinct full-rewrite and simplify challenges")
    if len(pair) > maximum:
        raise ValueError("Required validation strata exceed category capacity")
    reserved = {row["id"] for row in pair}
    remaining = [row for row in rows if row["id"] not in reserved]
    return [*pair, *balanced(remaining, maximum - len(pair))]


def requested_rewrite_stratum(row, simplify):
    """Sampling coverage uses the actual request, not exclusive subtype names.

    These are necessary sampling signals only. Independent review must still
    judge the actual instruction, answer preservation, and relative simplicity.
    """
    from tinychat.natural_sft_v4_admission import HANDOFF_REWRITE, HANDOFF_SIMPLIFY
    direction = row["messages"][0]["content"].partition("\n")[0].split("：", 1)[0].split(":", 1)[0][:160]
    asks_simpler = bool(HANDOFF_SIMPLIFY.search(direction))
    if simplify != asks_simpler or (not simplify and not HANDOFF_REWRITE.search(direction)):
        return False
    return any(audit["task_kind"] in ({"simplify", "rewrite"} if simplify else {"rewrite"})
               and audit["style_verdict"] in ({"satisfied"} if simplify else {"satisfied", "not_applicable"}) and audit["coverage"]
               for audit in row["automated_checker_response"]["turn_audits"])


def multi_subtype(row, checked=False):
    declared = row.get("multi_subtype")
    subtype = row.get("automated_checker_response", {}).get("multi_subtype") if checked else declared
    if subtype not in {"daily", "document"} or (checked and subtype != declared):
        raise ValueError("Missing or inconsistent independently checked multi subtype")
    return subtype


def balanced_multi(rows, maximum, checked=False):
    groups = {subtype: [row for row in rows if multi_subtype(row, checked) == subtype]
              for subtype in ("daily", "document")}
    shortage = {subtype: 2 - len(group) for subtype, group in groups.items() if len(group) < 2}
    if shortage:
        raise ValueError(f"At least two daily and two document conversations are required: {shortage}")
    # Do not replace missing daily examples with extra document operations.
    return [row for subtype in ("daily", "document")
            for row in balanced(groups[subtype], maximum // 2)]


@serialized_mutation
def select(action):
    profiles = {
        "micro_writer": ("micro_admit_v1", "micro_admitted_selection_v1.jsonl"),
        "micro_checker": ("micro_write_v1", "micro_written_selection_v1.jsonl"),
        "micro_review": ("micro_check_v1", "micro_review_selected_v1.jsonl"),
        "handoff_writer": ("handoff_admit_v1", "micro_admitted_handoff_v1.jsonl"),
        "handoff_checker": ("handoff_write_v1", "micro_written_handoff_v1.jsonl"),
        "handoff_review": ("handoff_check_v1", "micro_review_selected_handoff_v1.jsonl"),
        "indices_review": ("handoff_check_v1", "micro_review_selected_handoff_indices_v1.jsonl"),
        "complete_review": ("handoff_check_v1", "micro_review_selected_handoff_complete_v1.jsonl"),
        "semantic_review": ("handoff_check_v1", "micro_review_selected_handoff_semantic_v1.jsonl"),
        "round2_writer": ("round2_admit_v1", "micro_admitted_handoff_round2_v1.jsonl"),
        "round2_checker": ("round2_write_v1", "micro_written_handoff_round2_v1.jsonl"),
        "round2_review": ("round2_check_v1", "micro_review_selected_handoff_round2_v1.jsonl"),
    }
    phase, output = profiles[action]
    outcomes = list(result_rows(phase_path(phase)))
    if action in {"indices_review", "complete_review", "semantic_review"}:
        outcomes.extend(result_rows(phase_path("handoff_check_indices_v1")))
    if action in {"complete_review", "semantic_review"}:
        outcomes.extend(result_rows(phase_path("handoff_check_simplify_v1")))
    rows = [row["next_seed"] for row in outcomes if row["status"] == "accepted"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("A previously accepted row cannot be selected twice")
    counts = Counter(row["category"] for row in rows)
    report = {"action": action, "parent_phase": phase, "parent_total": len(outcomes),
              "parent_accepted": len(rows), "available_categories": dict(counts),
              "training_authorized": False, "semantic_passed": False}
    if action == "round2_writer":
        parent = phase_status(phase)
        if parent["completed"] != len(outcomes) or parent["submitted"] != len(outcomes) + parent["failed_no_result"]:
            raise ValueError("Round2 admission result and terminal failure counts disagree")
        report.update(parent_submitted=parent["submitted"], parent_completed=parent["completed"],
                      parent_failed_no_result=parent["failed_no_result"])
    if action.startswith("round2_"):
        report.update(new_chain_round=2, previous_new_chain_revise_count=1,
                      available_multi_subtypes=dict(Counter(row.get("multi_subtype") for row in rows if row["category"] == "multi")),
                      automatic_supplements_allowed=False)
    if action in {"indices_review", "complete_review", "semantic_review"}:
        report.update(parent_phases=[phase, "handoff_check_indices_v1"],
                      distinct_parent_candidates=len({row["id"] for row in outcomes}),
                      prior_insufficient_report="handoff_review_selection_v1.json")
        if action in {"complete_review", "semantic_review"}:
            report["parent_phases"].append("handoff_check_simplify_v1")
            report["prior_insufficient_report"] = "indices_review_selection_v1.json"
        if action == "semantic_review":
            report.update(prior_insufficient_report="complete_review_selection_v1.json",
                          stratum_basis="actual user directive plus explicit checker coverage/style; independent semantic review pending",
                          machine_subtype_labels_preserved=True)
    if action.endswith("_checker"):
        selected = rows
        if not rows:
            report.update(status="INSUFFICIENT", shortage={"written_conversations": 1})
            write_new(REPORT / f"{action}_selection_v1.json", report)
            raise ValueError("No complete written conversations to check")
    else:
        categories = ("natural", "grounded", "multi", "instruction")
        shortage = {category: 4 - counts[category] for category in categories if counts[category] < 4}
        if shortage:
            report.update(status="INSUFFICIENT", shortage=shortage)
            write_new(REPORT / f"{action}_selection_v1.json", report)
            raise ValueError(f"Not enough filtered examples; do not pad or relabel: {shortage}")
        maximum = 8 if action.endswith("_writer") else 4
        selected = []
        try:
            for category in categories:
                candidates = [row for row in rows if row["category"] == category]
                if action.startswith("round2_") and category == "multi":
                    selected.extend(balanced_multi(candidates, maximum, checked=action == "round2_review"))
                    continue
                if action == "round2_writer" and category == "instruction":
                    selected.extend(balanced_round2_instruction(candidates, maximum))
                    continue
                requirements = handoff_requirements(action, category) if action.startswith(("handoff_", "round2_")) or action in {"indices_review", "complete_review", "semantic_review"} else []
                selected.extend(balanced_required(candidates, maximum, requirements))
        except ValueError as error:
            report.update(status="INSUFFICIENT", stratum_shortage=str(error))
            write_new(REPORT / f"{action}_selection_v1.json", report)
            raise
    if len({row["id"] for row in selected}) != len(selected) or len({row["source_group"] for row in selected}) != len(selected):
        raise ValueError("Selected examples must have distinct IDs and groups")
    random.Random(2026112317).shuffle(selected)
    path = DATA / output
    save_jsonl(path, selected)
    report.update(status="SELECTED_NOT_INDEPENDENTLY_REVIEWED", selected=len(selected),
                  selected_categories=dict(Counter(row["category"] for row in selected)),
                  selected_repositories=dict(Counter(row["sources"][0]["repo"] for row in selected)),
                  selected_inputs=dict(Counter(row["sources"][0]["input"] for row in selected)),
                  selected_ids=[row["id"] for row in selected], output=str(path))
    if action.startswith("round2_"):
        report["selected_multi_subtypes"] = dict(Counter(row.get("multi_subtype") for row in selected if row["category"] == "multi"))
    write_new(REPORT / f"{action}_selection_v1.json", report)
    if action.endswith("_review"):
        # The reviewer first reads the blind file, then joins labels/provenance.
        suffix = "handoff_round2_v1" if action == "round2_review" else "handoff_semantic_v1" if action == "semantic_review" else "handoff_complete_v1" if action == "complete_review" else "handoff_indices_v1" if action == "indices_review" else "handoff_v1" if action.startswith("handoff_") else "v1"
        write_new(REPORT / f"micro_review_blind_{suffix}.json", {"records": [
            {"id": row["id"], "messages": row["messages"]} for row in selected]})
        write_new(REPORT / f"micro_review_labels_{suffix}.json", {"records": [
            {"id": row["id"], "category": row["category"], "source_group": row["source_group"],
             "sources": row["sources"], "lineage": row["stage_lineage"],
             **({"multi_subtype": row.get("multi_subtype")} if action == "round2_review" else {})} for row in selected]})
    return {key: report[key] for key in ("action", "selected", "selected_categories", "output", "semantic_passed")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("micro_writer", "micro_checker", "micro_review", "handoff_writer", "handoff_checker", "handoff_review", "indices_review", "complete_review", "semantic_review", "round2_writer", "round2_checker", "round2_review"))
    print(json.dumps(select(parser.parse_args().action), ensure_ascii=False))
