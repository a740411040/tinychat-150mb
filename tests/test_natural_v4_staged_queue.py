import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer
from scripts import natural_v4_staged_queue as queue
from scripts.natural_v4_staged_queue import payload, interpret, explicit_transform_check, validate_index_recheck


class StagedQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(Path(__file__).resolve().parents[1] / "data/stage1_v1/tokenizer.json"))

    def seed(self):
        return {"id": "example", "category": "instruction", "source_group": "fixture",
            "sources": [{"input": "belle_cache_single", "repo": "BelleGroup/train_0.5M_CN", "revision": "29c35e9b56c7fc91f04a613ed33896f7a19bad54"}],
            "original_messages": [{"role": "user", "content": "把所有逗号替换为句号。\n文本：\n今天下雨，我留在家里，两周后出门。"},
                                  {"role": "assistant", "content": "旧答"}],
            "rewrite_assistant_turn_indices": [1], "assistant_content_token_budget": 400,
            "admission_requirements": [{"assistant_message_index": 1, "text": "替换所有逗号，保留其他文字"}],
            "admission_evidence": [], "admission_dependencies": [], "admission_reason": "PRIVATE_ADMISSION_CONCLUSION"}

    def test_exact_transform_catches_an_interior_missed_comma(self):
        messages = self.seed()["original_messages"]
        messages[1]["content"] = "今天下雨。我留在家里，两周后出门。"
        self.assertEqual(explicit_transform_check(messages)[0]["passed"], False)
        messages[1]["content"] = "今天下雨。我留在家里。两周后出门。"
        self.assertEqual(explicit_transform_check(messages)[0]["passed"], True)

    def test_exact_transform_respects_script_scope_and_ignores_compound_task(self):
        messages = [{"role": "user", "content": "把所有中文逗号替换为句号。\n文本：\n甲，乙,A,B"},
                    {"role": "assistant", "content": "甲。乙,A,B"}]
        self.assertTrue(explicit_transform_check(messages)[0]["passed"])
        messages[1]["content"] = "甲。乙。A。B"
        self.assertFalse(explicit_transform_check(messages)[0]["passed"])
        messages[0]["content"] = "把所有英文逗号替换为句号。\n文本：\n甲，乙,A,B"
        messages[1]["content"] = "甲，乙。A。B"
        self.assertTrue(explicit_transform_check(messages)[0]["passed"])
        messages[0]["content"] = "把所有逗号替换为句号，并在末尾加感叹号。\n文本：\n甲，乙"
        self.assertEqual(explicit_transform_check(messages), [])

    def test_writer_keeps_user_and_rejects_incomplete_transform(self):
        seed = self.seed()
        original = copy.deepcopy(seed)
        bad = {"id": "example", "status": "rewritten", "assistant_replies": ["今天下雨。我留在家里，两周后出门。"], "note": ""}
        outcome = interpret("writer", seed, bad, self.tokenizer)
        self.assertEqual(outcome["status"], "rejected")
        self.assertIn("explicit_mechanical_transform_failed", outcome["errors"])
        good = {**bad, "assistant_replies": ["今天下雨。我留在家里。两周后出门。"]}
        revised = interpret("writer", seed, good, self.tokenizer)
        self.assertEqual(revised["status"], "accepted")
        self.assertEqual(revised["next_seed"]["messages"][0], seed["original_messages"][0])
        self.assertNotIn(2, revised["next_seed"]["encoded"]["input_ids"])
        self.assertEqual(seed, original)

    def test_checker_packet_hides_prior_category_and_judgements(self):
        seed = self.seed()
        seed["messages"] = seed["original_messages"]
        seed["messages"][0]["local_only"] = "PRIVATE_EXTRA"
        packet = payload(seed, "checker")
        self.assertEqual(set(packet), {"id", "messages"})
        self.assertEqual(set(packet["messages"][0]), {"role", "content"})
        self.assertNotIn("PRIVATE", str(packet))

    def test_checker_cannot_keep_a_defect_or_silently_change_category(self):
        seed = self.seed()
        seed["messages"] = seed["original_messages"]
        response = {"id": "example", "decision": "keep", "category": "natural", "issues": [], "reason": "一句话"}
        result = interpret("checker", seed, response, self.tokenizer)
        self.assertIn("checker_category_disagreement", result["errors"])
        response.update(category="instruction", issues=[{"kind": "instruction", "message_index": 1, "detail": "漏了一个逗号"}])
        self.assertEqual(interpret("checker", seed, response, self.tokenizer)["status"], "rejected")
        response["issues"] = []
        accepted = interpret("checker", seed, response, self.tokenizer)
        self.assertEqual(accepted["status"], "accepted")
        self.assertFalse(accepted["next_seed"]["semantic_passed"])
        self.assertFalse(accepted["next_seed"]["training_released"])

    def audited_rewrite(self):
        seed = self.seed()
        seed["messages"] = [{"role": "user", "content": "用更容易理解的话重写：研究者致力于帮助人们认识情绪。"},
                            {"role": "assistant", "content": "研究者努力帮助人们了解情绪。"}]
        response = {"id": "example", "decision": "keep", "category": "instruction", "issues": [], "reason": "已逐项核对",
            "turn_audits": [{"assistant_message_index": 1, "task_kind": "simplify", "source_integrity": "clean",
                "source_note": "完整原句", "coverage": [{"source_message_index": 0, "source_quote": "帮助人们认识情绪",
                    "answer_quote": "帮助人们了解情绪", "verdict": "covered", "detail": "行为与对象保留"}],
                "style_verdict": "satisfied", "style_note": "致力于改为努力，认识改为了解"}]}
        return seed, response

    def test_audited_checker_requires_complete_turns_and_real_quotes(self):
        seed, response = self.audited_rewrite()
        for mutate in (
            lambda r: r.pop("turn_audits"),
            lambda r: r["turn_audits"].append(copy.deepcopy(r["turn_audits"][0])),
            lambda r: r["turn_audits"][0]["coverage"][0].update(source_message_index=1),
            lambda r: r["turn_audits"][0]["coverage"][0].update(source_quote="原文没有的主张"),
            lambda r: r["turn_audits"][0]["coverage"][0].update(answer_quote="并未给出的回答"),
            lambda r: r["turn_audits"][0].update(coverage=[]),
        ):
            changed = copy.deepcopy(response)
            mutate(changed)
            with self.assertRaises(ValueError):
                interpret("checker", seed, changed, self.tokenizer, require_turn_audits=True)

    def test_claimed_keep_cannot_override_bad_source_coverage_or_style(self):
        seed, response = self.audited_rewrite()
        for field, value in (("source_integrity", "defective"), ("source_integrity", "uncertain"),
                             ("style_verdict", "failed"), ("style_verdict", "uncertain"),
                             ("style_verdict", "not_applicable")):
            changed = copy.deepcopy(response)
            changed["turn_audits"][0][field] = value
            self.assertEqual(interpret("checker", seed, changed, self.tokenizer, True)["status"], "rejected")
        changed = copy.deepcopy(response)
        changed["turn_audits"][0]["coverage"][0].update(verdict="missing", answer_quote="")
        self.assertEqual(interpret("checker", seed, changed, self.tokenizer, True)["status"], "rejected")

    def test_traceable_audit_still_is_not_an_independent_semantic_pass(self):
        seed, response = self.audited_rewrite()
        result = interpret("checker", seed, response, self.tokenizer, True)
        self.assertEqual(result["status"], "accepted")
        self.assertFalse(result["next_seed"]["semantic_passed"])
        self.assertFalse(result["next_seed"]["training_released"])

    def test_indexed_packet_contains_only_computed_indices_and_public_messages(self):
        seed, _ = self.audited_rewrite()
        seed["messages"].extend([{"role": "user", "content": "再简短一点", "private": "PRIVATE"},
                                 {"role": "assistant", "content": "研究者帮助人们了解情绪。"}])
        packet = payload(seed, "checker", indexed_messages=True)
        self.assertEqual([x["message_index"] for x in packet["messages"]], [0, 1, 2, 3])
        self.assertTrue(all(set(x) == {"message_index", "role", "content"} for x in packet["messages"]))
        self.assertNotIn("PRIVATE", str(packet))
        self.assertNotIn("category", packet)
        self.assertEqual([x["content"] for x in packet["messages"]], [x["content"] for x in seed["messages"]])

    def test_diagnostic_recheck_does_not_admit_content_rejections_or_edit_old_audits(self):
        seed, response = self.audited_rewrite()
        seed["messages"].extend([{"role": "user", "content": "再简短一点"},
                                 {"role": "assistant", "content": "研究者帮助人们了解情绪。"}])
        response["turn_audits"].append({"assistant_message_index": 4, "task_kind": "other", "source_integrity": "clean",
            "source_note": "可读", "coverage": [], "style_verdict": "not_applicable", "style_note": "无额外风格"})
        previous = {"status": "rejected", "errors": ["invalid_stage_response: Missing, duplicated or invalid assistant audit indices"],
                    "response": response}
        original = copy.deepcopy(previous)
        validate_index_recheck(seed, previous)
        self.assertEqual(previous, original)
        for mutate in (
            lambda row: row["response"].update(decision="reject"),
            lambda row: row["response"].update(issues=[{"kind": "content"}]),
            lambda row: row.update(errors=["checker_flagged_defect"]),
            lambda row: row["response"]["turn_audits"][1].update(source_integrity="defective"),
        ):
            changed = copy.deepcopy(previous)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_index_recheck(seed, changed)


    def test_round2_writer_prepare_enforces_quotas_before_creating_phase(self):
        seeds = []
        for category in ("natural", "grounded", "instruction", "multi"):
            for index in range(4):
                seed = self.seed()
                seed.update(id=f"{category}-{index}", source_group=f"{category}-{index}", category=category)
                if category == "multi":
                    seed["multi_subtype"] = "daily" if index < 2 else "document"
                if category == "instruction":
                    seed["validation_challenges"] = [["full_rewrite"], ["simplify"], [], []][index]
                if category == "grounded" and index == 0:
                    seed["sources"][0]["repo"] = "hfl/cmrc2018"
                seeds.append(seed)
        invalid = []
        natural_only = copy.deepcopy(seeds)
        for seed in natural_only:
            seed["category"] = "natural"
        invalid.append(("natural-only", natural_only, "each category"))
        document_only = copy.deepcopy(seeds)
        for seed in document_only:
            if seed["category"] == "multi":
                seed["multi_subtype"] = "document"
        invalid.append(("document-only", document_only, "daily"))
        shared_challenge = copy.deepcopy(seeds)
        for seed in shared_challenge:
            if seed["category"] == "instruction":
                seed["validation_challenges"] = ["full_rewrite", "simplify"] if seed["id"] == "instruction-0" else []
        invalid.append(("shared-challenge", shared_challenge, "distinct"))
        no_cmrc = copy.deepcopy(seeds)
        for seed in no_cmrc:
            seed["sources"][0]["repo"] = "BelleGroup/train_0.5M_CN"
        invalid.append(("no-cmrc", no_cmrc, "CMRC"))
        _, spec = queue.policy_for("round2_write_v1")
        for name, rows, error in invalid + [("valid", seeds, None)]:
            with self.subTest(name=name), patch.object(queue, "jsonl", return_value=rows), \
                 patch.object(queue, "result_rows", return_value=[{"id": row["id"], "status": "accepted", "next_seed": row} for row in rows]), \
                 patch.object(queue, "payload", side_effect=lambda row, *args, **kwargs: {"id": row["id"]}), \
                 patch.object(Path, "exists", return_value=False), \
                 patch.object(Path, "mkdir", side_effect=RuntimeError("phase creation boundary")) as mkdir:
                if error:
                    with self.assertRaisesRegex(ValueError, error):
                        queue.prepare.__wrapped__("round2_write_v1", queue.ROOT / spec["input"])
                    mkdir.assert_not_called()
                else:
                    with self.assertRaisesRegex(RuntimeError, "phase creation boundary"):
                        queue.prepare.__wrapped__("round2_write_v1", queue.ROOT / spec["input"])
                    mkdir.assert_called_once()

    def test_v6_context_is_required_and_subtype_disagreement_is_rejected(self):
        seed = self.seed()
        seed.update(category="multi", multi_subtype="daily", messages=[
            {"role": "user", "content": "我只有20分钟做晚饭，想用面条。"},
            {"role": "assistant", "content": "可以做汤面。"},
            {"role": "user", "content": "还有其他做法吗？"},
            {"role": "assistant", "content": "也可以做拌面，控制在20分钟内完成。"}])
        response = {"id": "example", "decision": "keep", "category": "multi", "issues": [], "reason": "后问沿用前轮食材与时限",
            "multi_subtype": "daily", "context_audits": [{"user_message_index": 2, "history_message_index": 0,
                "history_quote": "只有20分钟做晚饭", "missing_without_history": "不知道做晚饭的20分钟限制和面条这一食材", "answer_quote": "20分钟内完成"}],
            "turn_audits": [{"assistant_message_index": index, "task_kind": "other", "source_integrity": "clean",
                "source_note": "完整可读", "coverage": [], "style_verdict": "not_applicable", "style_note": "没有额外语体指令"}
                for index in (1, 3)]}
        missing = copy.deepcopy(response)
        missing.pop("context_audits")
        with self.assertRaises(ValueError):
            interpret("checker", seed, missing, self.tokenizer, True, True)
        result = interpret("checker", seed, response, self.tokenizer, True, True)
        self.assertEqual(result["status"], "accepted")
        self.assertFalse(result["next_seed"]["semantic_passed"])
        response["multi_subtype"] = "document"
        rejected = interpret("checker", seed, response, self.tokenizer, True, True)
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("checker_multi_subtype_disagreement", rejected["errors"])


class AdmissionTimeoutTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(dir=queue.ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.phase = queue.ROUND2_ADMISSION
        entry, pwsh = self.root / "entry.ps1", self.root / "pwsh.exe"
        entry.touch()
        pwsh.touch()
        paths = patch.multiple(queue, DATA=self.root / "data", ENTRY=entry, PWSH=pwsh)
        paths.start()
        self.addCleanup(paths.stop)
        self.target = queue.phase_path(self.phase)
        self.names = []
        self.failed = self.job("batch_00001", "quarantined")
        self.provider = self.root / "state" / "test-task.result.json"
        self.receipt = {"taskId": "test-task", "resultPath": str(self.provider), "exitCode": 3,
                        "executionPass": False, "taskCompleted": False}
        queue.write_new(self.failed / "receipt.json", self.receipt)
        self.provider_data = {
            **self.receipt, "completed": False, "providerErrors": [queue.TIMEOUT_PROVIDER_ERROR],
            "events": [{"synthetic": "external event payload"}], "finalText": "synthetic partial answer",
        }
        queue.write_new(self.provider, self.provider_data)

    def job(self, name, state):
        directory = self.target / name
        record = {"id": name, "phase": self.phase, "mode": "admission", "status": state,
                  "seed_ids": [f"{name}-{i}" for i in range(8)], "attempts": int(state != "prepared")}
        if state == "quarantined":
            record.update(completed_at="fixture", launcher_pid=100,
                          error="RuntimeError: Launcher exit=1; receipt_present=True")
        if state == "done":
            record["accepted"] = 7
            queue.write_new(directory / "result.json", {"rows": [
                {"id": identifier, "status": "accepted" if index < 7 else "rejected"}
                for index, identifier in enumerate(record["seed_ids"])
            ]})
        queue.write_new(directory / "job.json", record)
        self.names.append(name)
        queue.replace_json(self.target / "manifest.json", {
            "phase": self.phase, "mode": "admission", "spec": {"mode": "admission"}, "jobs": self.names,
        })
        return directory

    def test_round2_admission_timeout_is_separate_from_results_and_never_retried(self):
        done = self.job("batch_00002", "done")
        with self.assertRaisesRegex(ValueError, "Parent phase must finish"):
            list(queue.result_rows(self.target))
        previous_bytes = {name: (self.failed / name).read_bytes() for name in ("job.json", "receipt.json")}
        with patch.object(queue, "launcher_has_exited", return_value=True):
            result = queue.resolve_admission_timeout.__wrapped__(self.phase, self.failed.name)
            with self.assertRaises(FileExistsError):
                queue.resolve_admission_timeout.__wrapped__(self.phase, self.failed.name)
        self.assertEqual(result["status"], queue.TERMINAL_TIMEOUT)
        for name, data in previous_bytes.items():
            self.assertEqual((self.failed / name).read_bytes(), data)
        self.assertFalse((self.failed / "result.json").exists())
        evidence = queue.read(self.failed / "failure_resolution.json")
        self.assertEqual(set(evidence["provider_result"]), {
            "taskId", "exitCode", "completed", "executionPass", "taskCompleted", "providerErrors",
        })
        self.assertEqual(list(queue.result_rows(self.target)), queue.read(done / "result.json")["rows"])
        summary = queue.status(self.phase)
        self.assertEqual(tuple(summary[key] for key in ("submitted", "completed", "failed_no_result", "accepted")),
                         (16, 8, 8, 7))
        self.assertEqual(summary["jobs"], {queue.TERMINAL_TIMEOUT: 1, "done": 1})

        prepared = self.job("batch_00003", "prepared")
        review = self.root / "review.json"
        queue.write_new(review, {"status": "PASS", "allowed_phases": [self.phase]})
        with (
            patch.object(queue, "policy_for", return_value=({"code_review": str(review)}, {"mode": "admission"})),
            patch.object(queue, "execute", return_value={"status": "done"}) as execute,
            patch("builtins.print"),
        ):
            queue.run.__wrapped__(self.phase, max_batches=3, concurrency=1)
        execute.assert_called_once_with(prepared)

    def test_round2_admission_timeout_requires_terminal_provider_and_launcher_proof(self):
        with patch.object(queue, "launcher_has_exited", return_value=False):
            with self.assertRaisesRegex(ValueError, "still running"):
                queue.resolve_admission_timeout.__wrapped__(self.phase, self.failed.name)
        with patch.object(queue, "launcher_has_exited", return_value=True):
            for bad_result in ({"taskId": "different-task"}, {"completed": True}, {"providerErrors": []}):
                with self.subTest(bad_result=bad_result):
                    queue.replace_json(self.provider, {**self.provider_data, **bad_result})
                    with self.assertRaisesRegex(ValueError, "Provider result"):
                        queue.resolve_admission_timeout.__wrapped__(self.phase, self.failed.name)
        evidence = self.failed / "failure_resolution.json"
        self.assertFalse(evidence.exists())
        queue.write_new(evidence, {"status": queue.TERMINAL_TIMEOUT})
        with self.assertRaisesRegex(ValueError, "Invalid terminal admission timeout evidence"):
            list(queue.result_rows(self.target))


if __name__ == "__main__":
    unittest.main()
