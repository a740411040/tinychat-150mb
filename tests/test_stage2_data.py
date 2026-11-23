import io
import json
from pathlib import Path
import sqlite3
from ssl import SSLEOFError
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
from opencc import OpenCC

from scripts.acquire_stage2_sources import (
    RangeFile, TransferBudgetExceeded, choose_groups, recover_row_groups, write_text_batch,
)
from tinychat.chat_format import SPECIAL_IDS
from tinychat.stage1_data import file_identity, normalized_content
from tinychat.stage2_data import (
    Candidate, Corpus, EvaluationScreen, Input, SOURCE_WEIGHTS, document_repetition, publish, write_splits,
)


class Stage2DataTests(unittest.TestCase):
    def add(self, corpus, source, record_id, text, tokens=10, protected=False):
        encoding = None if protected else SimpleNamespace(ids=[100] * (tokens - 1))
        return corpus.add(
            Candidate(source, record_id, {"id": record_id, "source": source}, text, text, protected), encoding,
        )

    def test_near_cluster_is_excluded_by_connected_legacy_holdout(self):
        text = "海洋环流把温暖的海水从低纬度带到高纬度并影响沿岸气候和生物分布南北半球的风向和地球自转共同改变表层水的运动方向"
        changed = text[:18] + "暖" + text[19:]
        with sqlite3.connect(":memory:") as db:
            corpus = Corpus(db, None, None, EvaluationScreen([], None), {})
            self.assertEqual(self.add(corpus, "fineweb2_hq", "first", text)[0], "kept_at_ingest")
            self.assertEqual(self.add(corpus, "fineweb_edu", "second", changed)[0], "near_duplicate")
            self.assertEqual(self.add(corpus, "wiki", "old-test", changed, protected=True)[0], "protected_duplicate")
            self.assertEqual(corpus.candidate_counts(), {})

    def test_formal_sft_context_protects_a_previously_ingested_wiki_cluster(self):
        text = "海洋环流把温暖的海水从低纬度带到高纬度并影响沿岸气候和生物分布南北半球的风向和地球自转共同改变表层水的运动方向"
        changed = text[:18] + "暖" + text[19:]
        protected = Input(Path("sft_cmrc_holdouts.jsonl"), "sft_cmrc", protected=True)
        row = {"source_group": "cmrc:9", "split": "validation", "context": changed, "answer": "独立答案"}
        with sqlite3.connect(":memory:") as db:
            corpus = Corpus(db, None, OpenCC("t2s"), EvaluationScreen([], None), {})
            self.assertEqual(self.add(corpus, "wiki", "existing", text)[0], "kept_at_ingest")
            reason, candidate = corpus.prepare(protected, row, 0)
            self.assertIsNone(reason)
            self.assertEqual(candidate.text, changed)
            self.assertEqual(corpus.add(candidate, None), ("protected_holdout", 0))
            self.assertEqual(corpus.candidate_counts(), {})
            with self.assertRaisesRegex(ValueError, "Invalid SFT held-out context"):
                corpus.prepare(protected, {**row, "split": "train"}, 0)

    def test_eval_prompt_screen_uses_simplified_text_and_detects_near_line(self):
        traditional = "請說明潮汐如何改變沿海城市的日常漁業活動和居民出行安排"
        simple = "请说明潮汐如何改变沿海城市的日常渔业活动和居民出行安排"
        lines = "课程问题汇总\n" + simple.replace("出行", "航行") + "\n还有其他课后练习"
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            path.write_text(json.dumps({"messages": [{"role": "user", "content": traditional}]}),
                            encoding="utf-8")
            screen = EvaluationScreen([path], OpenCC("t2s"))
            self.assertTrue(screen.conflict(simple, simple))
            self.assertTrue(screen.conflict(lines, lines.replace("\n", "")))
            self.assertFalse(screen.conflict("今天讨论山地小气候", "今天讨论山地小气候"))

    def test_document_repetition_rejects_recycled_web_definitions_but_keeps_a_list(self):
        sections = (
            "测试者容易忽略仪器的额定范围，进而把显示屏上的故障符号误读为实际结果，或忽略现场温度的影响。",
            "标识应与图纸的一致，电工需检查元件附近是否已经充分绝缘，以及外露导体周围有无潮湿或者腐蚀痕迹。",
            "出现波动之后记录发生的时间，方便排除设备内部接点老化或电池电量下降的可能，并与初次测量结果对照。",
            "开关应置于断开位置，拆掉外壳之后可以拍下安装记录以及被遮挡的出厂编号，给后续保养留下一份依据。",
            "不同工况有各自的管理规范，根据设备的供电说明，可以确认哪些步骤应该交给专业人员以及何时需要停机。",
        )
        bad_titles = ("短路和开路的区别", "什么是开路短路", "短路与开路的区别是什么",
                      "什么叫短路开路", "开路和短路分别是什么意思")
        good_titles = ("核对仪器读数", "检查绝缘标识", "记录异常发生时间", "确保检修环境安全", "查阅供电管理规范")
        repeated = "相邻两端没有电流通过且读数保持稳定。"
        bad = "\n".join(f"{i}、{title}\n{repeated}{text}" for i, (title, text) in enumerate(zip(bad_titles, sections), 1))
        good = "\n".join(f"{i}、{title}\n{repeated}{text}" for i, (title, text) in enumerate(zip(good_titles, sections), 1))
        self.assertEqual(document_repetition(bad, normalized_content(bad), "fineweb2_hq"),
                         "repeated_definition_sections")
        self.assertIsNone(document_repetition(good, normalized_content(good), "fineweb2_hq"))
        copied = "\n".join(sections * 3)
        self.assertEqual(document_repetition(copied, normalized_content(copied), "wiki"), "repeated_document_phrases")

    def test_split_output_has_real_prefix_counts_and_eos_only(self):
        with TemporaryDirectory() as temporary, sqlite3.connect(":memory:") as db:
            output = Path(temporary)
            corpus = Corpus(db, None, None, EvaluationScreen([], None), {})
            for number, source in enumerate(SOURCE_WEIGHTS, start=1):
                self.add(corpus, source, f"train-{number}", chr(0x4E00 + number) * 20, tokens=5)
            self.add(corpus, "fineweb2_hq", "validation-1", "山间小雨初晴翠竹苔衣渐深", tokens=6)
            self.add(corpus, "fineweb_edu", "test-1", "居民翻修旧家具回收椅垫", tokens=7)
            with patch("tinychat.stage2_data.PREFIX_TOKENS", 17), \
                    patch("tinychat.stage2_data.split_for_cluster", side_effect=lambda value: value.split("-")[0]), \
                    patch("tinychat.stage2_data.ensure_disk_budget"):
                splits, prefix = write_splits(corpus, output)
            self.assertEqual(splits["train"]["tokens"], 20)
            self.assertEqual(splits["train"]["special_token_counts"]["<|eos|>"], 4)
            self.assertEqual(sum(splits["train"]["special_token_counts"].values()), 4)
            self.assertEqual(sorted(prefix.values()), [3, 4, 5, 5])
            binary = np.fromfile(output / "train.bin.partial", dtype="<u2")
            self.assertEqual(binary[4::5].tolist(), [SPECIAL_IDS["<|eos|>"]] * 4)

    def test_publish_preserves_fixed_general_validation_identity(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary, sqlite3.connect(":memory:") as db:
            directory = Path(temporary)
            output = directory / "new"
            output.mkdir()
            old_validation = directory / "validation.bin"
            old_validation.write_bytes(b"\x08\x00\x09\x00")
            tokenizer = directory / "tokenizer.json"
            tokenizer.write_text("{}", encoding="utf-8")
            old_manifest = {
                "files": {"validation.bin": file_identity(old_validation)},
                "splits": {"validation": {"tokens": 2, "documents": 1, "binary_bytes": 4}},
            }
            corpus = Corpus(db, None, None, EvaluationScreen([], None), {})
            for number, (source, tokens) in enumerate(SOURCE_WEIGHTS.items(), start=1):
                self.add(corpus, source, f"train-{number}", chr(0x4E00 + number) * 20, tokens=tokens)
            self.add(corpus, "fineweb2_hq", "validation-1", "山间小雨初晴翠竹苔衣渐深", tokens=6)
            self.add(corpus, "fineweb_edu", "test-1", "居民翻修旧家具回收椅垫", tokens=7)
            report = {
                "net_candidates": corpus.candidate_counts(), "conversion": {},
                "deduplication": "test", "local_source_provenance": {},
            }
            with patch.multiple("tinychat.stage2_data", PREFIX_TOKENS=99, TARGET_TRAIN_TOKENS=100,
                                MIN_CANDIDATE_TOKENS=100, MAX_CANDIDATE_TOKENS=120), \
                    patch("tinychat.stage2_data.split_for_cluster", side_effect=lambda value: value.split("-")[0]), \
                    patch("tinychat.stage2_data.ensure_disk_budget"):
                publish(corpus, output, directory, old_manifest, tokenizer, [], report)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            general = manifest["stage2"]["general_validation"]
            self.assertEqual(general["tokens"], 2)
            self.assertEqual((output / general["path"]).read_bytes(), old_validation.read_bytes())
            self.assertEqual(manifest["files"][general["path"]], file_identity(output / general["path"]))
            self.assertEqual(manifest["splits"]["train"]["source_tokens"], SOURCE_WEIGHTS)
            self.assertEqual(sum(manifest["stage2"]["train_prefix_source_tokens"].values()), 99)
            original = file_identity(output / "train.bin")
            with self.assertRaisesRegex(RuntimeError, "frozen manifest already exists"):
                publish(corpus, output, directory, old_manifest, tokenizer, [], report)
            self.assertEqual(file_identity(output / "train.bin"), original)

    def test_spool_guards_actual_serialized_bytes_before_write(self):
        stream = io.BytesIO()
        with patch("scripts.acquire_stage2_sources.ensure_disk_budget") as guard:
            write_text_batch(stream, Path("sample.jsonl"), ['简体"\n和空格'], "shard.parquet", 2, 5)
        guard.assert_called_once_with(Path("sample.jsonl"), required_bytes=len(stream.getvalue()))
        self.assertEqual(json.loads(stream.getvalue())["text"], '简体"\n和空格')
        with patch("scripts.acquire_stage2_sources.ensure_disk_budget", side_effect=RuntimeError("disk reserve")):
            previous = stream.getvalue()
            with self.assertRaisesRegex(RuntimeError, "disk reserve"):
                write_text_batch(stream, Path("sample.jsonl"), ["第二篇"], "shard.parquet", 2, 6)
            self.assertEqual(stream.getvalue(), previous)

    def test_spool_recovery_preserves_completed_groups_and_leaves_the_original(self):
        with TemporaryDirectory() as directory:
            previous = Path(directory) / "interrupted.jsonl.partial"
            target = Path(directory) / "renewed.jsonl.partial"
            shard = "source.parquet"
            with patch("scripts.acquire_stage2_sources.ensure_disk_budget"):
                with previous.open("wb") as original:
                    write_text_batch(original, previous, ["甲", "乙"], shard, 0, 0)
                    boundary = original.tell()
                    write_text_batch(original, previous, ["丙"], shard, 1, 0)
                    original.write(b'{"text":')
                old_bytes = previous.read_bytes()
                target_stream = io.BytesIO()
                recovered = recover_row_groups(previous, target_stream, target,
                                               [{"id": 0, "rows": 2}, {"id": 1, "rows": 2}], shard, 4)
            self.assertEqual(recovered, {"rows": 2, "bytes": boundary, "groups": [0]})
            self.assertEqual(target_stream.getvalue(), old_bytes[:boundary])
            self.assertEqual(previous.read_bytes(), old_bytes)

    def test_range_reader_refuses_over_budget_and_full_shard(self):
        with RangeFile("https://example.com/data", 100, 20, 1) as bounded:
            with self.assertRaises(TransferBudgetExceeded):
                bounded.readinto(bytearray(21))
            self.assertEqual(bounded.requests, 0)
        with RangeFile("https://example.com/data", 100, 100, 1) as full:
            with self.assertRaises(TransferBudgetExceeded):
                full.readinto(bytearray(100))
            self.assertEqual(full.requests, 0)

    def test_range_reader_validates_exact_content_range(self):
        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range": "bytes 10-12/100"}

        with RangeFile("https://example.com/data", 100, 10, 1) as bounded, \
                patch("scripts.acquire_stage2_sources.urlopen", side_effect=[Response(b"abc"), Response(b"abc")]) as fetch:
            bounded.seek(10)
            self.assertEqual(bounded.read(3), b"abc")
            self.assertEqual(bounded.bytes_read, 3)
            self.assertEqual(fetch.call_args.args[0].headers["Range"], "bytes=10-12")
            bounded.seek(20)
            with self.assertRaisesRegex(OSError, "did not honor"):
                bounded.read(3)

    def test_group_selection_checks_decoded_memory_and_total_transfer(self):
        groups = [
            {"id": 0, "compressed_bytes": 40, "uncompressed_bytes": 101},
            {"id": 1, "compressed_bytes": 40, "uncompressed_bytes": 90},
            {"id": 2, "compressed_bytes": 40, "uncompressed_bytes": 90},
        ]
        self.assertEqual(choose_groups(groups, None, 75, 100), groups[1:2])
        with self.assertRaises(TransferBudgetExceeded):
            choose_groups(groups, [0], 75, 100)
        with self.assertRaises(TransferBudgetExceeded):
            choose_groups(groups, [1, 2], 75, 100)

    def test_range_reader_retries_only_a_transient_connection_before_body_read(self):
        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range": "bytes 0-2/100"}

        with RangeFile("https://example.com/data", 100, 3, 1) as bounded, \
                patch("scripts.acquire_stage2_sources.time.sleep"), \
                patch("scripts.acquire_stage2_sources.urlopen", side_effect=[
                    URLError(SSLEOFError("unexpected EOF")), Response(b"abc"),
                ]) as fetch:
            self.assertEqual(bounded.read(3), b"abc")
            self.assertEqual(bounded.bytes_read, 3)
            self.assertEqual(bounded.requests, 2)
            self.assertEqual(fetch.call_args_list[0].args[0].headers, fetch.call_args_list[1].args[0].headers)
        with RangeFile("https://example.com/data", 100, 3, 1) as bounded, \
                patch("scripts.acquire_stage2_sources.urlopen", side_effect=URLError(PermissionError())):
            with self.assertRaises(URLError):
                bounded.read(3)
            self.assertEqual(bounded.requests, 1)
            self.assertEqual(bounded.bytes_read, 0)


if __name__ == "__main__":
    unittest.main()
