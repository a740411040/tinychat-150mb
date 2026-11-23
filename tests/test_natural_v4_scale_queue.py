"""Meaningful author-result contract checks; no network or training."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile

from tokenizers import Tokenizer
from scripts.natural_v4_scale_queue import interpret_result, validate_seed, packet_row, check_dispatch, shared_queue_lock, find_receipt


class ScaleQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(Path(__file__).resolve().parents[1] / "data/stage1_v1/tokenizer.json"))

    def seed(self):
        return {"id": "fixture-1", "category": "multi", "source_group": "fixture-source",
            "sources": [{"input": "belle_cache_single", "repo": "BelleGroup/train_0.5M_CN", "revision": "29c35e9b56c7fc91f04a613ed33896f7a19bad54"}], "original_messages": [
                {"role": "user", "content": "请把一句话缩短：今天我准备去图书馆读书。"},
                {"role": "assistant", "content": "我今天去图书馆读书。"},
                {"role": "user", "content": "再短一点。"},
                {"role": "assistant", "content": "今天去图书馆。"}],
            "rewrite_assistant_turn_indices": [1, 3], "assistant_content_token_budget": 400}

    def result(self, rows):
        return {"taskId": "fixture", "finalText": json.dumps({"rows": rows}, ensure_ascii=False)}

    def response(self, replies=None):
        return {"id": "fixture-1", "status": "rewritten", "assistant_replies": replies or ["今天去图书馆读书。", "去图书馆读书。"], "note": ""}

    def test_complete_rewrite_keeps_source_users_and_no_eos_format(self):
        seed = self.seed()
        original = copy.deepcopy(seed)
        validate_seed(seed)
        row = interpret_result(self.result([self.response()]), {seed["id"]: seed}, self.tokenizer)[0]
        self.assertFalse(row["structural_errors"])
        self.assertEqual(row["messages"][::2], original["original_messages"][::2])
        self.assertEqual(seed, original)
        self.assertFalse(row["training_released"])
        self.assertFalse(row["semantic_passed"])

    def test_missing_or_duplicated_ids_reject_entire_batch(self):
        assigned = {"fixture-1": self.seed()}
        for responses in ([], [self.response(), self.response()], [{**self.response(), "id": "unassigned"}]):
            with self.subTest(responses=responses), self.assertRaises(ValueError):
                interpret_result(self.result(responses), assigned, self.tokenizer)

    def test_incomplete_or_reserved_turn_is_quarantined_without_publishing_original(self):
        for replies in (["只有第一答"], ["正常回答", "<|eos|>"], ["", "正常回答"]):
            row = interpret_result(self.result([self.response(replies)]), {"fixture-1": self.seed()}, self.tokenizer)[0]
            self.assertTrue(row["structural_errors"])
            self.assertIsNone(row["messages"])

    def test_overlength_answer_is_not_truncated_or_accepted(self):
        replies = ["这是一段超出长度限制的完整回答。" * 300, "第二个答复。"]
        row = interpret_result(self.result([self.response(replies)]), {"fixture-1": self.seed()}, self.tokenizer)[0]
        self.assertIn("over_512_no_truncation", row["structural_errors"])
        self.assertEqual(row["writer_response"]["assistant_replies"], replies)
        self.assertIsNone(row["messages"])

    def test_writer_can_reject_a_seed_but_cannot_pass_its_own_quality(self):
        response = {"id": "fixture-1", "status": "reject", "assistant_replies": [], "note": "缺必要上下文"}
        row = interpret_result(self.result([response]), {"fixture-1": self.seed()}, self.tokenizer)[0]
        self.assertEqual(row["structural_errors"], ["writer_rejected_seed"])
        self.assertFalse(row["semantic_passed"])

    def test_packet_projects_nested_fields_and_rejects_unapproved_source(self):
        seed = self.seed()
        seed["original_messages"][0]["local_private_metadata"] = "DO_NOT_EXPORT"
        seed["sources"][0]["local_file"] = "DO_NOT_EXPORT"
        self.assertNotIn("DO_NOT_EXPORT", json.dumps(packet_row(seed)))
        original_revision = seed["sources"][0]["revision"]
        seed["sources"][0]["revision"] = "unreviewed-revision"
        with self.assertRaises(ValueError):
            packet_row(seed)
        seed["sources"][0]["revision"] = original_revision
        seed["sources"][0]["repo"] = "private/unapproved"
        with self.assertRaises(ValueError):
            packet_row(seed)

    def test_no_production_or_large_probe_before_independent_gate(self):
        check_dispatch("probe_v1", 16, 96)
        for phase, batch, count in (("production", 16, 100), ("probe_v1", 32, 96), ("probe_v1", 16, 97)):
            with self.subTest(phase=phase, batch=batch, count=count), self.assertRaises(ValueError):
                check_dispatch(phase, batch, count)

    def test_global_queue_lock_rejects_another_mutator(self):
        with tempfile.TemporaryDirectory() as directory, patch("scripts.natural_v4_scale_queue.DATA", Path(directory)):
            with shared_queue_lock():
                with self.assertRaises(OSError):
                    with shared_queue_lock():
                        self.fail("Another mutator acquired the shared queue lock")

    def test_native_gbk_receipt_keeps_unicode_result_path(self):
        receipt = {"taskId": "fixture", "resultPath": "C:\\Users\\测试用户\\result.json"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "launcher_stdout.txt").write_bytes(json.dumps(receipt, ensure_ascii=False).encode("gb18030"))
            parsed = find_receipt(path)
            self.assertEqual(parsed["resultPath"], receipt["resultPath"])
            self.assertEqual(parsed["launcher_stdout_encoding"], "gb18030")


if __name__ == "__main__":
    unittest.main()
