"""Record the observed first semantic fuse and current bounded handoff."""
import json
from pathlib import Path
from scripts.natural_v4_scale_queue import now


root = Path(__file__).resolve().parents[1]
path = root / "reports/natural_sft_v4/status.json"
backup = path.with_name("status.before_handoff_v1.json")
original = path.read_bytes()
with backup.open("xb") as handle:
    handle.write(original)
bom = original.startswith(b"\xef\xbb\xbf")
newline = "\r\n" if b"\r\n" in original else "\n"
status = json.loads(original.decode("utf-8-sig"))
status["historical_download_before_current_authorization"] = {
    "started": status.get("new_corpus_downloads_started"),
    "bytes": status.get("new_corpus_bytes"),
    "scope": "Earlier preparation only; no downloads in the active revision/handoff",
}
status.update(status="scale20k_handoff_v1_process_repair_no_training", stop_reason=None,
              new_corpus_downloads_started=False, new_corpus_bytes=0,
              training_started=False, dataset_released=False, data_quality_passed=False,
              new_user_direction="修正筛选与重写流程；新题独立门禁通过再扩量并去重/划分/抽检；不训练不下载",
              updated_local=now(), awaiting_user_direction=False)
status["active_scale20k"].update(semantic_revise_rounds=2, first_fuse_triggered=True,
                               bulk_authoring_enabled=False, final_dataset_released=False)
status["active_revision_v3"].update(
    phase="micro_independent_REVISE2_handoff",
    screened=64, admitted=48, rewritten=31, checker_kept=27,
    independently_reviewed=16, independent_pass=12, independent_minor=3, independent_major=1,
    independent_report="reports/natural_sft_v4/scale20k/revision_micro_semantic_review_v1.json",
    production_enabled=False, updated_at=now())
status["active_handoff_v1"] = {
    "contract": "planning/train_150mb/natural_sft_v4_handoff_contract_v1.txt",
    "phase": "process_repair_pending_independent_code_review",
    "process_executor": "natural_v4_semantic_pool (Sol)",
    "limited_generation_tool": "Kimi safe entry",
    "independent_reviewer": "natural_v4_quality_review (Sol)",
    "prior_chain_consecutive_revise": 2, "fuses_triggered": 1,
    "new_chain_semantic_round": 1, "new_chain_consecutive_revise": 0,
    "grok_attempt": {"task_id": "2026-11-22T18-26-02-197Z-grok-22c5091c",
                     "completed": False, "error": "Authentication required", "retried": False},
    "max_raw_candidates": 64, "max_written": 32, "next_independent_gate": 16,
    "production_enabled": False, "training_authorized": False, "new_downloads_authorized": False,
    "updated_at": now(),
}
encoded = (json.dumps(status, ensure_ascii=False, indent=2) + "\n").replace("\n", newline).encode("utf-8")
temporary = path.with_name("status.handoff_v1.tmp")
with temporary.open("xb") as handle:
    handle.write((b"\xef\xbb\xbf" if bom else b"") + encoded)
temporary.replace(path)
print(json.dumps({"status": status["status"], "first_fuse": True, "training": False, "downloads": False}))
