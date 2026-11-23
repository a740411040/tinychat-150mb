"""Bounded admission/writing/checking of local public text. Never trains a model."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

from tokenizers import Tokenizer
from tinychat.chat_format import contains_reserved, encode_conversation
from scripts.natural_v4_rewrite_pilot import completed_result, write_new
from scripts.natural_v4_scale_queue import (
    ROOT, ENTRY, PWSH, TOKENIZER, PUBLIC_SOURCES, PUBLIC_REVISIONS,
    read, jsonl, replace_json, find_receipt, serialized_mutation, now,
)

DATA = ROOT / "data/natural_sft_v4_revision_v3"
REPORT = ROOT / "reports/natural_sft_v4/scale20k"
POLICY = ROOT / "planning/train_150mb/natural_sft_v4_staged_dispatch_policy.json"
PROMPTS = {kind: ROOT / f"planning/train_150mb/natural_sft_v4_{kind}_v3_prompt.txt"
           for kind in ("admission", "writer", "checker")}
CATEGORIES = {"natural", "grounded", "instruction", "multi"}
MAX_BYTES = 65536
ROUND2_ADMISSION = "round2_admit_v1"
TERMINAL_TIMEOUT = "terminal_failed_no_retry"
TIMEOUT_PROVIDER_ERROR = {"code": -32070, "message": "Timed out after 180000ms"}


def phase_path(phase):
    if not re.fullmatch(r"[a-z0-9_]+", phase):
        raise ValueError("Invalid phase name")
    return DATA / "queue" / phase


def policy_for(phase):
    policy = read(POLICY)
    spec = policy["phases"].get(phase)
    if spec is None or spec["mode"] not in PROMPTS or not 1 <= spec["batch_size"] <= 16:
        raise ValueError("Phase is not enabled in the frozen dispatch policy")
    version = spec.get("prompt_version", 3)
    if version not in (3, 4, 5, 6):
        raise ValueError("Unknown frozen prompt version")
    if spec["mode"] == "checker" and version >= 4 and spec.get("audit_schema") != (3 if version == 6 else 2):
        raise ValueError("Checker version and evidence-audit schema disagree")
    if version == 5 and spec["mode"] != "checker":
        raise ValueError("Version 5 is only the explicitly indexed checker")
    if spec["mode"] == "checker" and version >= 5 and spec.get("index_messages") is not True:
        raise ValueError("Checker version 5/6 requires computed message indices")
    if spec.get("index_messages") and (spec["mode"] != "checker" or version not in (5, 6)):
        raise ValueError("Message indices require the reviewed checker protocol")
    return policy, spec


def source_identity(seed):
    if not seed.get("sources") or not seed.get("source_group"):
        raise ValueError("Missing public-source lineage")
    for source in seed["sources"]:
        repo = PUBLIC_SOURCES.get(source.get("input"))
        if not repo or source.get("repo") != repo or source.get("revision") != PUBLIC_REVISIONS[repo]:
            raise ValueError("Unapproved public-source identity/revision")


def clean_messages(messages):
    if not isinstance(messages, list) or len(messages) % 2 or not messages:
        raise ValueError("Expected complete exchanges")
    result = []
    for i, message in enumerate(messages):
        role, content = message.get("role"), message.get("content")
        if role != ("user" if i % 2 == 0 else "assistant") or not isinstance(content, str) or not content.strip() or contains_reserved(content):
            raise ValueError("Invalid role/content or reserved-token injection")
        result.append({"role": role, "content": content})
    return result


def payload(seed, mode, indexed_messages=False):
    source_identity(seed)
    if mode == "admission":
        from tinychat.natural_sft_v4_admission import admission_payload
        projected = admission_payload(seed)
        projected["original_messages"] = clean_messages(projected["original_messages"])
        return projected
    if mode == "checker":
        # Deliberately hide the admission label/rationale and writer conclusions.
        messages = clean_messages(seed["messages"])
        if indexed_messages:
            messages = [{"message_index": i, **message} for i, message in enumerate(messages)]
        return {"id": seed["id"], "messages": messages}
    return {"id": seed["id"], "category": seed["category"],
            "original_messages": clean_messages(seed["original_messages"]),
            "rewrite_assistant_turn_indices": list(seed["rewrite_assistant_turn_indices"]),
            "assistant_content_token_budget": seed["assistant_content_token_budget"],
            "requirements": [{"assistant_message_index": x["assistant_message_index"], "text": x["text"]}
                             for x in seed["admission_requirements"]],
            "evidence": [{"message_index": x["message_index"], "quote": x["quote"]}
                         for x in seed["admission_evidence"]],
            "dependencies": [{"user_message_index": x["user_message_index"], "history_message_index": x["history_message_index"],
                              "history_quote": x["history_quote"], "missing_without_history": x["missing_without_history"]}
                             for x in seed["admission_dependencies"]]}


def explicit_transform_check(messages):
    """A narrow exact check; absence of a recognized rule never means PASS."""
    checks = []
    for i in range(0, len(messages), 2):
        user, answer = messages[i]["content"], messages[i + 1]["content"]
        marker = re.search(r"(?:文本|输入|段落)[：:]\s*\n", user)
        if not marker:
            continue
        instructions, source = user[:marker.start()], user[marker.end():]
        operation = re.fullmatch(
            r"\s*(?:请)?(?:把|将)?(?:所有|全部)(?:的)?(?P<scope>中文|英文)?逗号"
            r"(?:替换|改|换)(?:成|为)(?:中文)?句号[。.!！]?\s*", instructions)
        if operation is None:
            continue
        # A full, single-operation instruction is essential: partial matching
        # must not silently ignore other requested transformations.
        expected = source
        if operation["scope"] != "英文":
            expected = expected.replace("，", "。")
        if operation["scope"] != "中文":
            expected = expected.replace(",", "。")
        checks.append({"message_index": i + 1, "kind": "commas_to_full_stops", "scope": operation["scope"] or "all",
                       "passed": answer == expected})
    return checks


def validate_turn_audits(messages, audits):
    """Validate traceable checks, without equating quoted evidence with truth."""
    indices = list(range(1, len(messages), 2))
    if not isinstance(audits, list) or len(audits) != len(indices):
        raise ValueError("Checker must audit every assistant turn")
    actual = [item.get("assistant_message_index") if isinstance(item, dict) else None for item in audits]
    if any(type(index) is not int for index in actual) or sorted(actual) != indices:
        raise ValueError("Missing, duplicated or invalid assistant audit indices")
    errors = []
    for audit in audits:
        index = audit["assistant_message_index"]
        kind = audit.get("task_kind")
        integrity, style = audit.get("source_integrity"), audit.get("style_verdict")
        if kind not in {"rewrite", "simplify", "summary", "mechanical", "other"}:
            raise ValueError("Invalid audited task kind")
        if integrity not in {"clean", "defective", "uncertain"} or style not in {"satisfied", "failed", "uncertain", "not_applicable"}:
            raise ValueError("Invalid audited source/style verdict")
        for field in ("source_note", "style_note"):
            if not isinstance(audit.get(field), str) or not audit[field].strip():
                raise ValueError("Missing audited source/style explanation")
        if integrity != "clean":
            errors.append(f"checker_source_not_clean:{index}")
        if style in {"failed", "uncertain"} or (kind == "simplify" and style != "satisfied"):
            errors.append(f"checker_style_not_satisfied:{index}")
        coverage = audit.get("coverage")
        if not isinstance(coverage, list) or (kind in {"rewrite", "simplify"} and not coverage):
            raise ValueError("Rewriting/simplification needs explicit coverage evidence")
        for unit in coverage:
            if not isinstance(unit, dict):
                raise ValueError("Malformed coverage unit")
            source_index, source_quote = unit.get("source_message_index"), unit.get("source_quote")
            answer_quote, verdict = unit.get("answer_quote"), unit.get("verdict")
            if (type(source_index) is not int or not 0 <= source_index < index
                    or not isinstance(source_quote, str) or not source_quote.strip()
                    or source_quote not in messages[source_index]["content"]):
                raise ValueError("Coverage source must quote an earlier visible message exactly")
            if (not isinstance(answer_quote, str) or (answer_quote and answer_quote not in messages[index]["content"])
                    or verdict not in {"covered", "missing", "changed", "uncertain"}
                    or not isinstance(unit.get("detail"), str) or not unit["detail"].strip()):
                raise ValueError("Invalid coverage verdict or answer quotation")
            if verdict == "covered" and not answer_quote.strip():
                raise ValueError("Covered obligation needs an actual answer span")
            if verdict != "covered":
                errors.append(f"checker_coverage_not_satisfied:{index}")
    return errors


def interpret(mode, seed, response, tokenizer, require_turn_audits=False, require_context_audits=False):
    if mode == "admission":
        from tinychat.natural_sft_v4_admission import validate_admission
        outcome = validate_admission(seed, response, tokenizer)
        return {"id": seed["id"], "status": "accepted" if outcome["decision"] == "accept" and not outcome["errors"] else "rejected",
                "errors": outcome["errors"], "next_seed": outcome["admitted_seed"], "response": response}
    errors, checks = [], []
    result = copy.deepcopy(seed)
    if mode == "writer":
        if response.get("status") == "reject":
            if response.get("assistant_replies") != [] or not isinstance(response.get("note"), str) or not response["note"].strip():
                errors.append("malformed_writer_rejection")
            else:
                errors.append("writer_rejected")
        elif response.get("status") != "rewritten":
            errors.append("invalid_writer_status")
        else:
            replies = response.get("assistant_replies")
            indices = seed["rewrite_assistant_turn_indices"]
            if not isinstance(replies, list) or len(replies) != len(indices) or any(
                not isinstance(reply, str) or not reply.strip() or contains_reserved(reply) for reply in replies
            ):
                errors.append("invalid_or_incomplete_assistant_replies")
            else:
                messages = clean_messages(seed["original_messages"])
                for index, reply in zip(indices, replies):
                    messages[index]["content"] = reply
                if messages[::2] != clean_messages(seed["original_messages"])[::2]:
                    raise AssertionError("User text changed")
                encoded = encode_conversation(tokenizer, messages, final_eos=False)
                if len(encoded["input_ids"]) > 512:
                    errors.append("over_512_no_truncation")
                if 2 in encoded["input_ids"] or encoded["labels"][-1] != 6:
                    raise AssertionError("Unexpected termination format")
                checks = explicit_transform_check(messages)
                if any(not item["passed"] for item in checks):
                    errors.append("explicit_mechanical_transform_failed")
                result.update(messages=messages, input_tokens=len(encoded["input_ids"]),
                              assistant_tokens=sum(x != -100 for x in encoded["labels"]),
                              encoded=encoded, mechanical_checks=checks)
    else:
        decision, category, issues = response.get("decision"), response.get("category"), response.get("issues")
        if decision not in ("keep", "reject") or category not in CATEGORIES | {"unsupported"} or not isinstance(issues, list):
            raise ValueError("Malformed checker response")
        if not isinstance(response.get("reason"), str) or not response["reason"].strip():
            raise ValueError("Checker must give a reason")
        for issue in issues:
            if (not isinstance(issue, dict) or issue.get("kind") not in {"source", "content", "instruction", "grounding", "dependency", "category"}
                    or type(issue.get("message_index")) is not int or not 0 <= issue["message_index"] < len(seed["messages"])
                    or not isinstance(issue.get("detail"), str) or not issue["detail"].strip()):
                raise ValueError("Malformed checker issue")
        if decision != "keep" or issues:
            errors.append("checker_flagged_defect")
        if category != seed["category"]:
            errors.append("checker_category_disagreement")
        if require_turn_audits:
            errors.extend(validate_turn_audits(seed["messages"], response.get("turn_audits")))
        if require_context_audits:
            from tinychat.natural_sft_v4_admission import validate_checker_context
            errors.extend(validate_checker_context(seed["messages"], response, seed.get("multi_subtype")))
        result["automated_checker_response"] = response
    result.update(semantic_passed=False, training_released=False)
    return {"id": seed["id"], "status": "accepted" if not errors else "rejected", "errors": errors,
            "next_seed": result if not errors else None, "response": response, "mechanical_checks": checks}


def timeout_receipt(directory, job):
    if (job.get("status") != "quarantined" or job.get("phase") != ROUND2_ADMISSION
            or job.get("mode") != "admission" or job.get("id") != directory.name
            or job.get("attempts") != 1 or not job.get("completed_at")
            or job.get("error") != "RuntimeError: Launcher exit=1; receipt_present=True"
            or (directory / "result.json").exists()):
        raise ValueError("Expected a quarantined, once-submitted admission job without an imported result")
    pid = job.get("launcher_pid")
    if type(pid) is not int or not 0 < pid < 2**32:
        raise ValueError("Quarantine does not identify its launcher PID")
    receipt = read(directory / "receipt.json")
    if not isinstance(receipt, dict):
        raise ValueError("Malformed safe-entry receipt")
    fields = {key: receipt.get(key) for key in ("taskId", "resultPath", "exitCode", "executionPass", "taskCompleted")}
    task_id, result_path = fields["taskId"], fields["resultPath"]
    if (not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", task_id)
            or not isinstance(result_path, str) or not Path(result_path).is_absolute()
            or fields["exitCode"] != 3 or fields["executionPass"] is not False
            or fields["taskCompleted"] is not False):
        raise ValueError("Expected a failed safe-entry receipt with an exact task ID and result path")
    expected = (ENTRY.parent / "state" / f"{task_id}.result.json").resolve()
    if Path(result_path).resolve() != expected:
        raise ValueError("Receipt result path differs from the launcher task identity")
    return fields


def timeout_result_facts(receipt, result):
    if (not isinstance(result, dict) or result.get("taskId") != receipt["taskId"]
            or result.get("exitCode") != 3 or result.get("completed") is not False
            or result.get("taskCompleted") is not False or result.get("executionPass") is not False):
        raise ValueError("Provider result has a different task ID or lacks terminal failure fields")
    errors = result.get("providerErrors")
    if not isinstance(errors, list) or not any(
        isinstance(error, dict) and all(error.get(key) == value for key, value in TIMEOUT_PROVIDER_ERROR.items())
        for error in errors
    ):
        raise ValueError("Provider result lacks the recorded 180000ms timeout")
    return {key: result[key] for key in ("taskId", "exitCode", "completed", "taskCompleted", "executionPass")} | {
        "providerErrors": [TIMEOUT_PROVIDER_ERROR.copy()],
    }


def launcher_has_exited(pid):
    # Process lookup is strictly read-only. A live or uninspectable PID blocks confirmation.
    command = (
        f"try {{ $launcher = [System.Diagnostics.Process]::GetProcessById({pid}); "
        "if ($launcher.HasExited) { exit 0 }; exit 1 } "
        "catch [System.ArgumentException] { exit 0 } catch { exit 2 }"
    )
    process = subprocess.run([str(PWSH), "-NoLogo", "-NonInteractive", "-Command", command],
                             capture_output=True, check=False, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if process.returncode not in (0, 1):
        raise ValueError("Could not confirm the launcher process has exited")
    return process.returncode == 0


@serialized_mutation
def resolve_admission_timeout(phase, name):
    if phase != ROUND2_ADMISSION or not re.fullmatch(r"batch_[0-9]{5}", name):
        raise ValueError("Timeout resolution requires one round2_admit_v1 batch")
    target = phase_path(phase)
    manifest = read(target / "manifest.json")
    if manifest.get("phase") != phase or name not in manifest["jobs"]:
        raise ValueError("Job is outside the frozen admission manifest")
    directory = target / name
    evidence_path = directory / "failure_resolution.json"
    if evidence_path.exists():
        raise FileExistsError("Terminal failure resolution already exists")
    job = read(directory / "job.json")
    receipt = timeout_receipt(directory, job)
    if not launcher_has_exited(job["launcher_pid"]):
        raise ValueError("Admission launcher is still running")
    # The exact receipt points to the structured result. Extract only completion metadata.
    result = timeout_result_facts(receipt, read(Path(receipt["resultPath"])))
    evidence = {
        "phase": phase, "job": name, "status": TERMINAL_TIMEOUT,
        "recorded_at": now(), "quarantine_job": job, "receipt": receipt, "provider_result": result,
        "launcher_exited": True,
    }
    # The queue lock makes the create-only guard safe. Publish the sidecar atomically.
    replace_json(evidence_path, evidence)
    return {"phase": phase, "job": name, "status": TERMINAL_TIMEOUT, "task_id": receipt["taskId"],
            "failed_no_result": len(job["seed_ids"])}


def effective_job_status(target, manifest, name, job):
    if manifest.get("phase") != ROUND2_ADMISSION or target.name != ROUND2_ADMISSION:
        return job["status"]
    if job["status"] == TERMINAL_TIMEOUT:
        raise ValueError("Terminal admission timeouts must retain the original quarantine job")
    directory = target / name
    evidence_path = directory / "failure_resolution.json"
    if job["status"] != "quarantined" or not evidence_path.exists():
        return job["status"]
    evidence = read(evidence_path)
    receipt = timeout_receipt(directory, job)
    result = evidence.get("provider_result")
    if (not isinstance(evidence, dict) or evidence.get("phase") != ROUND2_ADMISSION or evidence.get("job") != name
            or evidence.get("status") != TERMINAL_TIMEOUT or not evidence.get("recorded_at")
            or evidence.get("quarantine_job") != job or evidence.get("receipt") != receipt
            or evidence.get("launcher_exited") is not True or timeout_result_facts(receipt, result) != result):
        raise ValueError("Invalid terminal admission timeout evidence")
    return TERMINAL_TIMEOUT


def result_rows(target):
    manifest = read(target / "manifest.json")
    for name in manifest["jobs"]:
        directory = target / name
        job = read(directory / "job.json")
        state = effective_job_status(target, manifest, name, job)
        if state == TERMINAL_TIMEOUT and manifest.get("phase") == target.name == ROUND2_ADMISSION:
            continue
        if state != "done":
            raise ValueError("Parent phase must finish before a child phase is prepared")
        yield from read(directory / "result.json")["rows"]


def validate_index_recheck(seed, previous):
    """Allow only the two-turn [1,4] audit-index defect, never rejected content."""
    expected_error = "invalid_stage_response: Missing, duplicated or invalid assistant audit indices"
    response = previous.get("response", {})
    if (previous.get("status") != "rejected" or previous.get("errors") != [expected_error]
            or response.get("decision") != "keep" or response.get("issues") != []
            or response.get("category") != seed["category"] or len(seed["messages"]) != 4):
        raise ValueError("Only an isolated index-format rejection can enter this diagnostic recheck")
    audits = copy.deepcopy(response.get("turn_audits"))
    if not isinstance(audits, list) or [item.get("assistant_message_index") for item in audits] != [1, 4]:
        raise ValueError("Not the diagnosed [1,4] index defect")
    audits[1]["assistant_message_index"] = 3
    if validate_turn_audits(seed["messages"], audits):
        raise ValueError("The original response has other source/coverage/style defects")
    # The in-memory correction diagnoses eligibility only. It is never saved or
    # used as a successful result; a new explicitly numbered check must finish.


def validate_round2_writer_selection(seeds):
    """Enforce the frozen sampling contract before any writer queue is created."""
    counts = Counter(seed.get("category") for seed in seeds)
    if set(counts) != CATEGORIES or any(not 4 <= counts[category] <= 8 for category in CATEGORIES):
        raise ValueError("Round2 writer requires 4 to 8 rows in each category")
    subtypes = Counter(seed.get("multi_subtype") for seed in seeds if seed["category"] == "multi")
    if set(subtypes) != {"daily", "document"} or any(not 2 <= count <= 4 for count in subtypes.values()):
        raise ValueError("Round2 writer requires 2 to 4 daily and 2 to 4 document conversations")
    challenges = {kind: {seed["id"] for seed in seeds if seed["category"] == "instruction"
                         and kind in seed.get("validation_challenges", [])}
                  for kind in ("full_rewrite", "simplify")}
    if not all(challenges.values()) or len(set.union(*challenges.values())) < 2:
        raise ValueError("Round2 writer requires distinct full-rewrite and simplify challenges")
    if not any(seed["category"] == "grounded" and seed.get("sources")
               and seed["sources"][0].get("repo") == "hfl/cmrc2018" for seed in seeds):
        raise ValueError("Round2 writer requires a CMRC grounded challenge")


@serialized_mutation
def prepare(phase, input_file):
    _, spec = policy_for(phase)
    expected = (ROOT / spec["input"]).resolve()
    if Path(input_file).resolve() != expected:
        raise ValueError("Input differs from the explicit phase policy")
    seeds = jsonl(input_file)
    if not spec["min_rows"] <= len(seeds) <= spec["max_rows"] or len({x["id"] for x in seeds}) != len(seeds):
        raise ValueError("Phase count or unique-ID contract violated")
    if "allowed_seed_ids" in spec and {x["id"] for x in seeds} != set(spec["allowed_seed_ids"]):
        raise ValueError("Input IDs differ from the explicitly reviewed phase scope")
    if len({x["source_group"] for x in seeds}) != len(seeds):
        raise ValueError("Probe requires distinct source groups")
    if phase == "round2_write_v1":
        validate_round2_writer_selection(seeds)
    if spec["parent"]:
        parents = {row["id"]: row["next_seed"] for row in result_rows(phase_path(spec["parent"])) if row["status"] == "accepted"}
        for seed in seeds:
            if seed["id"] not in parents or seed != parents[seed["id"]]:
                raise ValueError("Child seed must exactly match a successful parent-stage artifact")
    else:
        previous_ids = set()
        for manifest in (DATA / "queue").glob("*/manifest.json"):
            if read(manifest)["mode"] == "admission":
                previous_ids.update(read(manifest)["seed_ids"])
        if previous_ids.intersection(x["id"] for x in seeds):
            raise ValueError("Fresh admission cannot implicitly rerun a previously submitted ID")
    if spec.get("recheck_of"):
        if spec["mode"] != "checker" or set(spec["recheck_ids"]) != {seed["id"] for seed in seeds}:
            raise ValueError("Diagnostic recheck must match its exact reviewed ID list")
        previous = {row["id"]: row for row in result_rows(phase_path(spec["recheck_of"]))}
        if len(previous) + len(seeds) > spec["recheck_total_max_rows"]:
            raise ValueError("Diagnostic check would exceed the existing micro checker cap")
        for seed in seeds:
            validate_index_recheck(seed, previous[seed["id"]])
    if spec.get("cumulative_with"):
        total, used_ids = len(seeds), set()
        for previous_phase in spec["cumulative_with"]:
            earlier = read(phase_path(previous_phase) / "manifest.json")
            if earlier["mode"] != spec["mode"]:
                raise ValueError("Cumulative limits must compare the same operation")
            total += earlier["rows"]
            used_ids.update(earlier["seed_ids"])
        if total > spec["cumulative_max_rows"] or used_ids.intersection(seed["id"] for seed in seeds):
            raise ValueError("Supplement exceeds its cumulative cap or repeats an already submitted record")
    target = phase_path(phase)
    if target.exists():
        raise FileExistsError("Existing phase is immutable; use status/reconcile")
    packets, batch = [], []
    for seed in seeds:
        public = payload(seed, spec["mode"], indexed_messages=spec.get("index_messages", False))
        tentative = batch + [(seed, public)]
        size = len((json.dumps({"records": [x[1] for x in tentative]}, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        if batch and (len(tentative) > spec["batch_size"] or size > MAX_BYTES):
            packets.append(batch)
            batch = [(seed, public)]
        else:
            batch = tentative
        if len((json.dumps({"records": [x[1] for x in batch]}, ensure_ascii=False, indent=2) + "\n").encode("utf-8")) > MAX_BYTES:
            raise ValueError("Even one public record exceeds the small-text limit; do not truncate it")
    if batch:
        packets.append(batch)
    target.mkdir(parents=True)
    prompt = ROOT / f"planning/train_150mb/natural_sft_v4_{spec['mode']}_v{spec.get('prompt_version', 3)}_prompt.txt"
    (target / "prompt.txt").write_text(prompt.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    names = []
    for i, packet in enumerate(packets, 1):
        name = f"batch_{i:05d}"
        directory = target / name
        directory.mkdir()
        with (directory / "seeds.jsonl").open("x", encoding="utf-8", newline="\n") as out:
            for seed, _ in packet:
                out.write(json.dumps(seed, ensure_ascii=False) + "\n")
        write_new(directory / "input.txt", {"records": [x[1] for x in packet]})
        write_new(directory / "job.json", {"id": name, "phase": phase, "mode": spec["mode"], "status": "prepared",
                  "seed_ids": [x[0]["id"] for x in packet], "attempts": 0, "created_at": now()})
        names.append(name)
    write_new(target / "manifest.json", {"phase": phase, "mode": spec["mode"], "spec": spec,
              "seed_ids": [x["id"] for x in seeds], "groups": [x["source_group"] for x in seeds],
              "rows": len(seeds), "jobs": names, "created_at": now(), "training_authorized": False})
    return {"phase": phase, "rows": len(seeds), "jobs": len(names)}


def import_job(directory, receipt, elapsed=None):
    job = read(directory / "job.json")
    spec = read(directory.parent / "manifest.json")["spec"]
    if not receipt.get("executionPass") or not receipt.get("taskCompleted"):
        raise ValueError("Safe entry did not complete its read-only contract")
    raw = completed_result(Path(receipt["resultPath"]))
    if raw["taskId"] != receipt["taskId"]:
        raise ValueError("Mismatched receipt task identity")
    seeds = {x["id"]: x for x in jsonl(directory / "seeds.jsonl")}
    answer = json.loads(raw["finalText"])
    responses = answer.get("rows") if isinstance(answer, dict) else None
    if not isinstance(responses, list) or len(responses) != len(seeds):
        raise ValueError("Incomplete response batch")
    ids = [x.get("id") for x in responses if isinstance(x, dict)]
    if len(ids) != len(responses) or len(set(ids)) != len(ids) or set(ids) != set(seeds):
        raise ValueError("Missing, repeated or unassigned response IDs")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    rows = []
    for response in responses:
        try:
            outcome = interpret(job["mode"], seeds[response["id"]], response, tokenizer,
                                require_turn_audits=spec.get("audit_schema", 1) >= 2,
                                require_context_audits=spec.get("audit_schema", 1) == 3)
        except (ValueError, KeyError, TypeError) as error:
            outcome = {"id": response["id"], "status": "rejected", "errors": [f"invalid_stage_response: {error}"],
                       "next_seed": None, "response": response}
        if outcome["next_seed"] is not None:
            lineage = list(outcome["next_seed"].get("stage_lineage", []))
            lineage.append({"phase": job["phase"], "mode": job["mode"], "job": job["id"], "task_id": raw["taskId"]})
            if spec.get("recheck_of"):
                lineage[-1]["recheck_of"] = spec["recheck_of"]
            outcome["next_seed"]["stage_lineage"] = lineage
        rows.append(outcome)
    result = {"task_id": raw["taskId"], "result_file": receipt["resultPath"], "actual_input_reads": 1,
              "seed_ids": job["seed_ids"], "rows": rows, "elapsed_seconds": elapsed, "training_released": False}
    if not (directory / "result.json").exists():
        write_new(directory / "result.json", result)
    else:
        existing = read(directory / "result.json")
        if existing["task_id"] != result["task_id"] or existing["rows"] != result["rows"]:
            raise ValueError("Existing result differs; refusing overwrite")
    job.update(status="done", task_id=raw["taskId"], completed_at=now(), elapsed_seconds=elapsed,
               accepted=sum(x["status"] == "accepted" for x in rows), result_file=receipt["resultPath"])
    replace_json(directory / "job.json", job)
    return {"job": job["id"], "status": "done", "accepted": job["accepted"], "submitted": len(seeds), "seconds": elapsed}


def execute(directory):
    job = read(directory / "job.json")
    if job["status"] != "prepared" or job["attempts"]:
        raise ValueError("No automatic external retries")
    job.update(status="running", attempts=1, started_at=now())
    replace_json(directory / "job.json", job)
    began = time.monotonic()
    try:
        command = [str(PWSH), "-NoLogo", "-NonInteractive", "-File", str(ENTRY), "-Agent", "kimi",
                   "-InputFile", str(directory / "input.txt"), "-PromptFile", str(directory.parent / "prompt.txt")]
        with (directory / "launcher_stdout.txt").open("x", encoding="utf-8") as out, (directory / "launcher_stderr.txt").open("x", encoding="utf-8") as err:
            process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            job["launcher_pid"] = process.pid
            replace_json(directory / "job.json", job)
            try:
                code = process.wait(timeout=330)
            except subprocess.TimeoutExpired:
                subprocess.run([str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/taskkill.exe"),
                                "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                raise RuntimeError("Owned launcher exceeded its watchdog")
        receipt = find_receipt(directory)
        if receipt:
            write_new(directory / "receipt.json", receipt)
        if code != 0 or receipt is None:
            raise RuntimeError(f"Launcher exit={code}; receipt_present={receipt is not None}")
        return import_job(directory, receipt, round(time.monotonic() - began, 3))
    except Exception as error:
        job = read(directory / "job.json")
        job.update(status="quarantined", error=f"{type(error).__name__}: {error}", completed_at=now())
        replace_json(directory / "job.json", job)
        return {"job": job["id"], "status": "quarantined", "error": job["error"]}


def status(phase):
    target = phase_path(phase)
    manifest = read(target / "manifest.json")
    jobs = []
    for name in manifest["jobs"]:
        job = read(target / name / "job.json")
        jobs.append((job, effective_job_status(target, manifest, name, job)))
    summary = {"phase": phase, "mode": manifest["mode"], "jobs": dict(Counter(state for _, state in jobs)),
               "submitted": sum(len(job["seed_ids"]) for job, _ in jobs if job["attempts"]),
               "accepted": sum(job.get("accepted", 0) for job, state in jobs
                               if phase != ROUND2_ADMISSION or state == "done"), "training_authorized": False}
    if phase == ROUND2_ADMISSION:
        summary.update(completed=sum(len(job["seed_ids"]) for job, state in jobs if state == "done"),
                       failed_no_result=sum(len(job["seed_ids"]) for job, state in jobs if state == TERMINAL_TIMEOUT))
    return summary


def reconcile_unlocked(target):
    for name in read(target / "manifest.json")["jobs"]:
        directory = target / name
        if read(directory / "job.json")["status"] == "running":
            receipt = find_receipt(directory)
            if receipt:
                import_job(directory, receipt)


@serialized_mutation
def reconcile(phase):
    reconcile_unlocked(phase_path(phase))
    return status(phase)


@serialized_mutation
def run(phase, max_batches, concurrency):
    if not ENTRY.is_file() or not PWSH.is_file() or not 1 <= concurrency <= 3 or max_batches < 1:
        raise ValueError("Unavailable verified runtime or invalid run bounds")
    policy, spec = policy_for(phase)
    review = read(ROOT / policy["code_review"])
    if review.get("status") != "PASS" or phase not in review.get("allowed_phases", []):
        raise ValueError("Independent code review does not allow this bounded phase")
    target = phase_path(phase)
    manifest = read(target / "manifest.json")
    if manifest["spec"] != spec:
        raise ValueError("Dispatch policy changed after this queue was frozen")
    reconcile_unlocked(target)
    jobs = [(target / name, read(target / name / "job.json")) for name in manifest["jobs"]]
    if any(effective_job_status(target, manifest, path.name, job) in ("running", "quarantined") for path, job in jobs):
        raise ValueError("Unresolved job; diagnose rather than retry")
    pending = iter([path for path, job in jobs if job["status"] == "prepared"][:max_batches])
    active, stopped = {}, False
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while True:
            while not stopped and len(active) < concurrency:
                if (DATA / "PAUSE").exists() or (target / "PAUSE").exists():
                    stopped = True
                    break
                directory = next(pending, None)
                if directory is None:
                    break
                active[executor.submit(execute, directory)] = directory
            if not active:
                break
            done, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
            for future in done:
                outcome = future.result()
                print(json.dumps(outcome, ensure_ascii=False), flush=True)
                stopped = stopped or outcome["status"] != "done"
                active.pop(future)
            replace_json(target / "progress.json", {**status(phase), "updated_at": now(), "stopped_dispatch": stopped})
    return status(phase)


@serialized_mutation
def collect(phase, output):
    rows = list(result_rows(phase_path(phase)))
    seeds = [x["next_seed"] for x in rows if x["status"] == "accepted"]
    target = Path(output).resolve()
    if not target.is_relative_to(DATA.resolve()):
        raise ValueError("Collected data must stay in the new revision directory")
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for seed in seeds:
            handle.write(json.dumps(seed, ensure_ascii=False) + "\n")
    collected = {"phase": phase, "submitted": len(rows), "accepted": len(seeds),
                 "categories": dict(Counter(x.get("category", "unknown") for x in seeds)), "path": str(target)}
    if phase == ROUND2_ADMISSION:
        summary = status(phase)
        collected.update({key: summary[key] for key in ("submitted", "completed", "failed_no_result")})
    return collected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status", "reconcile", "collect", "resolve-admission-timeout"))
    parser.add_argument("--phase", required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job", help="A single quarantined admission batch for explicit timeout resolution")
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.input is None:
            parser.error("prepare requires --input")
        value = prepare(args.phase, args.input)
    elif args.action == "run":
        value = run(args.phase, args.max_batches, args.concurrency)
    elif args.action == "status":
        value = status(args.phase)
    elif args.action == "reconcile":
        value = reconcile(args.phase)
    elif args.action == "resolve-admission-timeout":
        if args.job is None:
            parser.error("resolve-admission-timeout requires --job")
        value = resolve_admission_timeout(args.phase, args.job)
    else:
        if args.output is None:
            parser.error("collect requires --output")
        value = collect(args.phase, args.output)
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
