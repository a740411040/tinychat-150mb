"""Bounded, resumable public-text authoring queue; never imports a model trainer."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from functools import wraps
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tokenizers import Tokenizer
from tinychat.chat_format import contains_reserved, encode_conversation
from scripts.natural_v4_rewrite_pilot import completed_result, write_new

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/natural_sft_v4_scale20k"
REPORT = ROOT / "reports/natural_sft_v4/scale20k"
WRITER_PROMPT = ROOT / "planning/train_150mb/natural_sft_v4_scale20k_writer_prompt.txt"
ENTRY = Path.home() / ".codex" / "external-agents" / "acpx-safe" / "Invoke-SafeAgent.ps1"
# Verified locally with Get-Command pwsh / Get-Process: the desktop runtime
# provides pwsh 7.6.9 here; this machine has no standalone Program Files install.
PWSH = Path(r"C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt\dependencies\native\powershell\pwsh.exe")
TOKENIZER = ROOT / "data/stage1_v1/tokenizer.json"
MAX_PACKET_BYTES = 65536
MAX_CONCURRENCY = 3
CATEGORIES = {"natural", "grounded", "multi", "instruction"}
POLICY = ROOT / "planning/train_150mb/natural_sft_v4_scale20k_dispatch_policy.json"
PUBLIC_SOURCES = {
    "moss_helpfulness": "OpenMOSS-Team/moss-002-sft-data",
    "moss_honesty": "OpenMOSS-Team/moss-002-sft-data",
    "belle_cache_single": "BelleGroup/train_0.5M_CN",
    "belle_cache_multi": "BelleGroup/multiturn_chat_0.8M",
    "belle_spans_multi": "BelleGroup/multiturn_chat_0.8M",
    "belle_stage1": "BelleGroup/train_0.5M_CN",
    "cmrc_train": "hfl/cmrc2018", "cmrc_validation": "hfl/cmrc2018",
    "smoltalk_everyday": "opencsg/smoltalk-chinese",
}
PUBLIC_REVISIONS = {
    "OpenMOSS-Team/moss-002-sft-data": "d530c921add4c7551b4e2309e0e7fe9468278311",
    "BelleGroup/train_0.5M_CN": "29c35e9b56c7fc91f04a613ed33896f7a19bad54",
    "BelleGroup/multiturn_chat_0.8M": "dc32eb8955e10ae1cd4189a50a566ef8f8254a65",
    "hfl/cmrc2018": "137f2c45a24275fb68f6961c4d357f46288886aa",
    "opencsg/smoltalk-chinese": "5edf5d0d7b06794755cf78b274a150a4efe1c9d4",
}


@contextmanager
def shared_queue_lock():
    """One OS-held lock covers reservations, dispatch and receipt reconciliation."""
    import msvcrt
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "queue.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def serialized_mutation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with shared_queue_lock():
            return function(*args, **kwargs)
    return wrapped


def check_dispatch(phase, batch_size, count):
    policy = read(POLICY)
    entry = policy.get("phases", {}).get(phase)
    if not entry or batch_size != entry["batch_size"] or not entry.get("min_seeds", 1) <= count <= entry["max_seeds"]:
        raise ValueError("Phase/batch/count not enabled by the frozen dispatch policy")
    if entry["kind"] == "probe":
        if count > 96 or batch_size > 16:
            raise ValueError("Probe exceeds the independently planned bounded scope")
        if read(REPORT / "minor_repairs_release_v2.json").get("status") != "PASS":
            raise ValueError("The six pilot repairs must pass before expansion")
    elif entry["kind"] == "production":
        gate_path = (ROOT / entry["gate"]).resolve()
        if not gate_path.is_relative_to(REPORT.resolve()):
            raise ValueError("Quality gate must be a local scale20k review artifact")
        gate = read(gate_path)
        if (gate.get("status") != "PASS" or gate.get("allow_bulk_text_authoring") is not True
                or gate.get("review_type") != "scale20k_bulk_authoring_gate"):
            raise ValueError("Production requires an explicit independent quality gate")
        if not gate.get("independent_review_files"):
            raise ValueError("Quality gate lacks independent evidence")
        for name in gate["independent_review_files"]:
            review_path = (ROOT / name).resolve()
            if not review_path.is_relative_to(REPORT.resolve()) or read(review_path).get("status") != "PASS":
                raise ValueError("Independent review is absent or has not passed")
        probe_phase = entry.get("required_probe_phase")
        probe_review_path = (ROOT / gate.get("probe_semantic_review", "missing")).resolve()
        if (not probe_phase or not probe_review_path.is_relative_to(REPORT.resolve())
                or gate.get("probe_semantic_review") not in gate["independent_review_files"]):
            raise ValueError("Production gate must bind the independently reviewed expansion probe")
        probe_review = read(probe_review_path)
        if (probe_review.get("review_type") != "expansion_probe_semantic"
                or probe_review.get("phase") != probe_phase
                or probe_review.get("submitted_conversations") != 96
                or probe_review.get("reviewed_conversations") != 96
                or probe_review.get("major_count") != 0
                or probe_review.get("unresolved_minor_count") != 0):
            raise ValueError("An unrelated PASS cannot substitute for the 96-row semantic gate")
        probe_manifest = read(DATA / "queue" / probe_phase / "manifest.json")
        if probe_manifest["conversations"] != 96:
            raise ValueError("Probe review does not correspond to the frozen 96-row queue")
    else:
        raise ValueError("Unknown dispatch policy kind")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def replace_json(path, value):
    """Atomic state replacement; immutable inputs/results use write_new instead."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".next")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_seed(seed):
    if seed["category"] not in CATEGORIES:
        raise ValueError("Unexpected route/category")
    messages = seed["original_messages"]
    if not messages or len(messages) % 2 or any(
        message.get("role") != ("user" if i % 2 == 0 else "assistant")
        or not isinstance(message.get("content"), str)
        or not message["content"].strip() or contains_reserved(message["content"])
        for i, message in enumerate(messages)
    ):
        raise ValueError("Expected a complete, nonempty source prefix starting with user")
    if seed["rewrite_assistant_turn_indices"] != list(range(1, len(messages), 2)):
        raise ValueError("Must rewrite every retained assistant turn")
    if seed["category"] == "multi" and len(messages) < 4:
        raise ValueError("Multi route requires at least two exchanges")
    if not seed.get("sources") or not seed.get("source_group"):
        raise ValueError("Missing source provenance")
    if any(source.get("input") not in PUBLIC_SOURCES
           or source.get("repo") != PUBLIC_SOURCES[source["input"]]
           or source.get("revision") != PUBLIC_REVISIONS[PUBLIC_SOURCES[source["input"]]]
           for source in seed["sources"]):
        raise ValueError("Seed is not tied to one of the explicitly permitted local public sources")
    if not isinstance(seed["assistant_content_token_budget"], int) or seed["assistant_content_token_budget"] < 24:
        raise ValueError("No useful answer budget")


def packet_row(seed):
    validate_seed(seed)
    # Public dialogue text and authoring constraints only. No local paths, model
    # evaluations, expected benchmark answers or complete project provenance.
    result = {"id": seed["id"], "category": seed["category"],
        "original_messages": [{"role": message["role"], "content": message["content"]}
                              for message in seed["original_messages"]],
        "rewrite_assistant_turn_indices": list(seed["rewrite_assistant_turn_indices"]),
        "assistant_content_token_budget": seed["assistant_content_token_budget"],
        "constraints": [], "dependency_anchors": []}
    for constraint in seed.get("constraints", []):
        index, focus = constraint["assistant_message_index"], constraint["review_focus"]
        if index not in seed["rewrite_assistant_turn_indices"] or not isinstance(focus, str):
            raise ValueError("Invalid assistant constraint")
        result["constraints"].append({"assistant_message_index": index, "review_focus": focus})
    for anchor in seed.get("dependency_anchors", []):
        user_index = anchor["user_message_index"]
        through = anchor["history_through_assistant_message_index"]
        if type(user_index) is not int or user_index <= 0 or user_index >= len(seed["original_messages"]) or user_index % 2:
            raise ValueError("Invalid dependent user index")
        if through not in seed["rewrite_assistant_turn_indices"] or through >= user_index:
            raise ValueError("Invalid history boundary")
        projected = {"user_message_index": user_index,
                     "history_through_assistant_message_index": through, "source_spans": []}
        for span in anchor.get("source_spans", []):
            index = span["message_index"]
            start, end = span["chars_zero_based_half_open"]
            if any(type(value) is not int for value in (index, start, end)) or not 0 <= index < len(seed["original_messages"]):
                raise ValueError("Invalid source span coordinates")
            content = seed["original_messages"][index]["content"]
            if not 0 <= start < end <= len(content) or content[start:end] != span["text"]:
                raise ValueError("Dependency span must be an exact public source substring")
            projected["source_spans"].append({"message_index": index, "text": content[start:end]})
        result["dependency_anchors"].append(projected)
    return result


@serialized_mutation
def prepare(seed_path, phase, batch_size):
    if not phase.replace("_", "").isalnum() or not 1 <= batch_size <= 32:
        raise ValueError("Invalid queue phase or bounded batch size")
    seeds = jsonl(seed_path)
    if not seeds or len({row["id"] for row in seeds}) != len(seeds):
        raise ValueError("Empty or repeated seed IDs")
    check_dispatch(phase, batch_size, len(seeds))
    prior_ids = set()
    for manifest_path in (DATA / "queue").glob("*/manifest.json"):
        prior_ids.update(row["id"] for row in jsonl(manifest_path.parent / "seeds.jsonl"))
    if prior_ids.intersection(row["id"] for row in seeds):
        raise ValueError("Seed IDs already assigned in another phase; no implicit retry")
    if 64 + len(prior_ids) + len(seeds) > 25200:
        raise ValueError("Cumulative submission budget including the 64-item pilot exceeded")
    target = DATA / "queue" / phase
    if target.exists():
        raise FileExistsError("Queue already exists; use status/reconcile, never overwrite")
    batches = [seeds[i:i + batch_size] for i in range(0, len(seeds), batch_size)]
    packets = []
    for batch in batches:
        packet = {"purpose": "Rewrite only assistant responses for existing public practice questions. No training or corpus downloads.",
                  "records": [packet_row(seed) for seed in batch]}
        content = json.dumps(packet, ensure_ascii=False, indent=2)
        if len((content + "\n").encode("utf-8")) > MAX_PACKET_BYTES:
            raise ValueError("Packet exceeds 64 KiB; reduce the bounded batch size")
        packets.append((packet, batch))
    target.mkdir(parents=True)
    # Immutable snapshot makes later changes to the upstream seed file harmless.
    with (target / "seeds.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for seed in seeds:
            handle.write(json.dumps(seed, ensure_ascii=False) + "\n")
    prompt = WRITER_PROMPT.read_text(encoding="utf-8")
    (target / "writer_prompt.txt").write_text(prompt, encoding="utf-8", newline="\n")
    jobs = []
    for i, (packet, batch) in enumerate(packets, 1):
        job = f"batch_{i:05d}"
        directory = target / job
        directory.mkdir()
        write_new(directory / "input.txt", packet)
        write_new(directory / "job.json", {"id": job, "phase": phase, "status": "prepared",
            "seed_ids": [row["id"] for row in batch], "created_at": now(), "attempts": 0,
            "training_authorized": False})
        jobs.append(job)
    write_new(target / "manifest.json", {"phase": phase, "source_seeds": str(Path(seed_path).resolve()),
        "conversations": len(seeds), "batch_size": batch_size, "jobs": jobs,
        "category": dict(Counter(row["category"] for row in seeds)), "training_authorized": False,
        "corpus_downloads_authorized": False, "created_at": now()})
    return {"phase": phase, "jobs": len(jobs), "conversations": len(seeds), "path": str(target)}


def interpret_result(raw, assigned, tokenizer):
    payload = json.loads(raw["finalText"])
    responses = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(responses, list) or len(responses) != len(assigned):
        raise ValueError("Writer returned an incomplete batch")
    ids = [row.get("id") for row in responses if isinstance(row, dict)]
    if len(ids) != len(responses) or len(set(ids)) != len(ids) or set(ids) != set(assigned):
        raise ValueError("Writer IDs must match the unique assigned set exactly")
    rows = []
    for response in responses:
        seed = assigned[response["id"]]
        errors = []
        replies = response.get("assistant_replies")
        if response.get("status") == "reject":
            if replies != [] or not isinstance(response.get("note"), str) or not response["note"].strip():
                errors.append("malformed_writer_rejection")
            else:
                errors.append("writer_rejected_seed")
        elif response.get("status") != "rewritten":
            errors.append("unknown_writer_status")
        elif not isinstance(replies, list) or len(replies) != len(seed["rewrite_assistant_turn_indices"]):
            errors.append("incomplete_assistant_turns")
        elif any(not isinstance(reply, str) or not reply.strip() or contains_reserved(reply) for reply in replies):
            errors.append("invalid_or_reserved_reply")
        encoded = None
        messages = copy.deepcopy(seed["original_messages"])
        if not errors:
            for index, reply in zip(seed["rewrite_assistant_turn_indices"], replies):
                messages[index]["content"] = reply
            if messages[::2] != seed["original_messages"][::2]:
                raise AssertionError("User text must remain exact")
            encoded = encode_conversation(tokenizer, messages, final_eos=False)
            if len(encoded["input_ids"]) > 512:
                errors.append("over_512_no_truncation")
            if 2 in encoded["input_ids"] or encoded["labels"][-1] != 6:
                raise AssertionError("Unexpected EOS or missing supervised END_TURN")
        row = {**copy.deepcopy(seed), "writer_task_id": raw["taskId"], "writer_response": response,
            "structural_errors": errors, "semantic_passed": False, "training_released": False,
            "messages": messages if not errors else None,
            "input_tokens": len(encoded["input_ids"]) if encoded else None,
            "assistant_tokens": sum(label != -100 for label in encoded["labels"]) if encoded else None}
        rows.append(row)
    return rows


def import_receipt(directory, receipt, elapsed=None):
    job = read(directory / "job.json")
    if (directory / "result.json").exists():
        result = read(directory / "result.json")
        if result["seed_ids"] != job["seed_ids"]:
            raise ValueError("Existing result does not match job")
    else:
        if not receipt.get("executionPass") or not receipt.get("taskCompleted"):
            raise ValueError("Safe entry did not complete successfully")
        raw = completed_result(Path(receipt["resultPath"]))
        if raw["taskId"] != receipt["taskId"]:
            raise ValueError("Receipt/task identity mismatch")
        seed_map = {row["id"]: row for row in jsonl(directory.parent / "seeds.jsonl")}
        assigned = {key: seed_map[key] for key in job["seed_ids"]}
        tokenizer = Tokenizer.from_file(str(TOKENIZER))
        rows = interpret_result(raw, assigned, tokenizer)
        result = {"job": job["id"], "task_id": raw["taskId"], "seed_ids": job["seed_ids"],
            "result_file": receipt["resultPath"], "input_reads": 1, "elapsed_seconds": elapsed,
            "rows": rows, "training_released": False, "semantic_passed": False}
        write_new(directory / "result.json", result)
    job.update(status="done", completed_at=now(), task_id=receipt.get("taskId"),
               result_file=receipt.get("resultPath"), elapsed_seconds=elapsed,
               structurally_accepted=sum(not row["structural_errors"] for row in result["rows"]))
    replace_json(directory / "job.json", job)
    return {"job": job["id"], "status": job["status"], "accepted": job["structurally_accepted"],
            "submitted": len(job["seed_ids"]), "elapsed_seconds": elapsed}


def find_receipt(directory):
    path = directory / "launcher_stdout.txt"
    if not path.exists():
        return None
    raw = path.read_bytes()
    try:
        decoded, encoding = raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        # The verified pwsh host emits GBK when its console is redirected.
        # Decode strictly; never replace invalid characters in artifact paths.
        decoded, encoding = raw.decode("gb18030"), "gb18030"
    for line in reversed(decoded.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "taskId" in value and "resultPath" in value:
            value["launcher_stdout_encoding"] = encoding
            return value
    return None


def execute_job(directory):
    job = read(directory / "job.json")
    if job["status"] != "prepared" or job["attempts"]:
        raise ValueError("No automatic reruns or retries")
    if not ENTRY.is_file() or not PWSH.is_file():
        raise FileNotFoundError("The authorized safe entry / pwsh 7 is unavailable")
    job.update(status="running", attempts=1, started_at=now(), controller_pid=os.getpid())
    replace_json(directory / "job.json", job)
    start = time.monotonic()
    try:
        command = [str(PWSH), "-NoLogo", "-NonInteractive", "-File", str(ENTRY), "-Agent", "kimi",
                   "-InputFile", str(directory / "input.txt"),
                   "-PromptFile", str(directory.parent / "writer_prompt.txt")]
        with (directory / "launcher_stdout.txt").open("x", encoding="utf-8") as stdout, \
             (directory / "launcher_stderr.txt").open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            job["launcher_pid"] = process.pid
            replace_json(directory / "job.json", job)
            try:
                exit_code = process.wait(timeout=330)
            except subprocess.TimeoutExpired:
                # This PID was created by this controller. The wrapper itself has
                # a 300s watchdog; never terminate unrelated applications.
                subprocess.run([str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/taskkill.exe"),
                                "/PID", str(process.pid), "/T", "/F"], check=False,
                               capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                raise RuntimeError("Owned safe-entry launcher exceeded its outer watchdog")
        receipt = find_receipt(directory)
        if receipt is not None:
            write_new(directory / "receipt.json", receipt)
        if exit_code != 0 or receipt is None:
            raise RuntimeError(f"Safe-entry launcher exit={exit_code}; receipt_present={receipt is not None}")
        return import_receipt(directory, receipt, round(time.monotonic() - start, 3))
    except Exception as error:
        job = read(directory / "job.json")
        job.update(status="quarantined", error=f"{type(error).__name__}: {error}",
                   completed_at=now(), elapsed_seconds=round(time.monotonic() - start, 3))
        replace_json(directory / "job.json", job)
        return {"job": job["id"], "status": "quarantined", "error": job["error"]}


def _reconcile_unlocked(target):
    """Collect completed receipts after interruption; never resubmit an uncertain job."""
    outcomes = []
    for name in read(target / "manifest.json")["jobs"]:
        directory = target / name
        job = read(directory / "job.json")
        if job["status"] != "running":
            continue
        receipt = find_receipt(directory)
        if receipt is None:
            outcomes.append({"job": name, "status": "running_or_unknown_no_retry"})
            continue
        try:
            outcomes.append(import_receipt(directory, receipt))
        except Exception as error:
            job.update(status="quarantined", error=f"{type(error).__name__}: {error}")
            replace_json(directory / "job.json", job)
            outcomes.append({"job": name, "status": "quarantined", "error": job["error"]})
    return outcomes


@serialized_mutation
def reconcile(target):
    return _reconcile_unlocked(target)


@serialized_mutation
def recover_encoding(target):
    """Explicit local import repair only: never re-launch an external writer."""
    outcomes = []
    for name in read(target / "manifest.json")["jobs"]:
        directory = target / name
        job = read(directory / "job.json")
        if job["status"] != "quarantined" or not job.get("error", "").startswith("FileNotFoundError:"):
            continue
        receipt = find_receipt(directory)
        if receipt is None or receipt.get("launcher_stdout_encoding") != "gb18030":
            raise ValueError("This recovery command only handles the diagnosed GBK receipt case")
        write_new(directory / "receipt_encoding_recovered.json", receipt)
        # Keep original raw stdout and old receipt as failure evidence. New
        # state uses valid Unicode and references the recovered receipt.
        outcome = import_receipt(directory, receipt, job.get("elapsed_seconds"))
        repaired = read(directory / "job.json")
        repaired.pop("error", None)
        repaired["recovery"] = {"reason": "GBK launcher stdout originally decoded as UTF-8",
                               "receipt": "receipt_encoding_recovered.json", "external_retry": False}
        replace_json(directory / "job.json", repaired)
        outcomes.append(outcome)
    return outcomes


def status(target):
    manifest = read(target / "manifest.json")
    jobs = [read(target / name / "job.json") for name in manifest["jobs"]]
    return {"phase": manifest["phase"], "jobs": dict(Counter(job["status"] for job in jobs)),
        "submitted_conversations": sum(len(job["seed_ids"]) for job in jobs if job["attempts"]),
        "structurally_accepted": sum(job.get("structurally_accepted", 0) for job in jobs),
        "training_authorized": False,
        "failures": [{"job": job["id"], "error": job.get("error")} for job in jobs if job["status"] == "quarantined"][:10]}


@serialized_mutation
def run(target, concurrency, maximum):
    if not 1 <= concurrency <= MAX_CONCURRENCY or maximum < 1:
        raise ValueError("Concurrency must be 1..3 and max-batches positive")
    if not ENTRY.is_file() or not PWSH.is_file():
        raise FileNotFoundError("The verified safe entry / pwsh 7 path is unavailable; no jobs dispatched")
    # The global lock also covers direct Python API callers, every phase and
    # standalone reconciliation. Worker threads run under this one held lock.
    _reconcile_unlocked(target)
    manifest = read(target / "manifest.json")
    check_dispatch(manifest["phase"], manifest["batch_size"], manifest["conversations"])
    jobs = [(target / name, read(target / name / "job.json")) for name in manifest["jobs"]]
    if any(job["status"] in ("running", "quarantined") for _, job in jobs):
        raise RuntimeError("Unresolved running/quarantined job: diagnose before dispatch")
    pending = iter([directory for directory, job in jobs if job["status"] == "prepared"][:maximum])
    active, stopped = {}, False
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while True:
            stopped = stopped or (target / "PAUSE").exists() or (DATA / "PAUSE").exists()
            while not stopped and len(active) < concurrency:
                if (target / "PAUSE").exists() or (DATA / "PAUSE").exists():
                    stopped = True
                    break
                directory = next(pending, None)
                if directory is None:
                    break
                active[executor.submit(execute_job, directory)] = directory
            if not active:
                break
            finished, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
            for future in finished:
                result = future.result()
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if result["status"] != "done":
                    stopped = True
                active.pop(future)
            current = status(target)
            current.update(updated_at=now(), controller_pid=os.getpid(), dispatch_paused=stopped)
            replace_json(target / "progress.json", current)
    final = status(target)
    final.update(finished_at=now(), dispatch_paused=stopped)
    replace_json(target / "progress.json", final)
    return final


@serialized_mutation
def assemble(target):
    manifest = read(target / "manifest.json")
    rows = []
    for name in manifest["jobs"]:
        if read(target / name / "job.json")["status"] != "done":
            raise ValueError("All assigned jobs must finish or be explicitly reconciled first")
        rows.extend(read(target / name / "result.json")["rows"])
    path = target / "candidates.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"phase": manifest["phase"], "submitted": len(rows),
        "structurally_accepted": sum(not row["structural_errors"] for row in rows),
        "reject_reasons": dict(Counter(error for row in rows for error in row["structural_errors"])),
        "accepted_by_category": dict(Counter(row["category"] for row in rows if not row["structural_errors"])),
        "quality_passed": False, "training_authorized": False, "path": str(path)}
    write_new(target / "assembly.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status", "reconcile", "recover-encoding", "assemble"))
    parser.add_argument("--phase", required=True)
    parser.add_argument("--seeds", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=6)
    args = parser.parse_args()
    if not args.phase.replace("_", "").isalnum():
        parser.error("Invalid phase")
    target = DATA / "queue" / args.phase
    if args.action == "status":
        output = status(target)
    elif args.action == "prepare":
        if args.seeds is None:
            parser.error("prepare requires --seeds")
        output = prepare(args.seeds, args.phase, args.batch_size)
    elif args.action == "run":
        output = run(target, args.concurrency, args.max_batches)
    elif args.action == "reconcile":
        output = reconcile(target)
    elif args.action == "recover-encoding":
        output = recover_encoding(target)
    else:
        output = assemble(target)
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
