from collections import Counter
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from tinychat.chat_format import IGNORE_LABEL, SPECIALS, SPECIAL_IDS, encode_conversation
from tinychat.sft_data import build_sft, conversation_split, parse_conversation, read_json_records, select_window
from tinychat.stage1_data import (
    CorpusSettings, SPLITS, build_pretrain, document_split, ingest_batch, json_line, normalized_content, sha256,
)


def multi_row(index, *, followup=False):
    prior = f"Human:请说明分类编号{index}的流程。\nAssistant:先记录编号，然后检查归类条件。\n"
    if followup:
        prior += "Human:怎么验证编号？\nAssistant:核对登记表中的编号。\n"
    return {"instruction": prior + "Human:接下来需要做什么？\nAssistant:",
            "input": "", "output": "整理审核记录，并提交复核。"}


class Stage1DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = TemporaryDirectory(dir=Path.cwd())
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.base = cls.root / "base"
        cls.eval_path = cls.root / "prompts.jsonl"
        cls.eval_question = "请按照顺序说明这道保密数学练习题从观察条件到写出结论的全部步骤"
        cls.eval_path.write_text(json_line({"messages": [{"role": "user", "content": cls.eval_question}]}),
                                 encoding="utf-8")

        examples = {split: [] for split in SPLITS}
        for index in range(10000):
            item = f"生产线上值班人员需要检查电源和工作台，随后把测试结果详细写入交接日志。记录编号{index}。"
            key = sha256(normalized_content(item).encode("utf-8"))
            split = document_split(key, CorpusSettings.seed)
            if len(examples[split]) < 4:
                examples[split].append(item)
            if all(len(group) == 4 for group in examples.values()):
                break
        unique = [item for split in SPLITS for item in examples[split]]
        # Duplicate and evaluation candidates sit together in a single batch before the quota.
        rows = [unique[0], unique[0], unique[0].replace("。", "！"),
                "关于一次讨论：" + cls.eval_question + "，将结果写入记录。", *unique[1:]]
        path = cls.root / "input.parquet"
        pq.write_table(pa.table({"text": rows, "ignored": ["元数据"] * len(rows)}), path, row_group_size=len(rows))
        source = {"repo": "local/fixture", "revision": "fixture", "selected": [
            {"path": path.name, "bytes": path.stat().st_size},
        ]}
        settings = CorpusSettings(shard_bytes=sum(len(row.encode("utf-8")) for row in unique),
                                  tokenizer_bytes_per_shard=300, vocab_size=280, min_train_tokens=2,
                                  min_han=4, batch_size=4)
        attempts = 0

        def fail_on_second_batch(*args):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise OSError("simulated interruption")
            return ingest_batch(*args)

        with patch("tinychat.stage1_data.ingest_batch", side_effect=fail_on_second_batch):
            try:
                build_pretrain(cls.base, source, eval_prompts=cls.eval_path, local_root=cls.root, settings=settings)
            except OSError as exc:
                if str(exc) != "simulated interruption":
                    raise
            else:
                raise AssertionError("Expected an interruption during fixture acquisition")
        cls.manifest = build_pretrain(cls.base, source, eval_prompts=cls.eval_path, local_root=cls.root, settings=settings)
        cls.tokenizer = Tokenizer.from_file(cls.manifest["tokenizer_path"])

    def test_pretraining_resume_dedup_screen_and_frozen_splits(self):
        source = self.manifest["sources"]["selected"][0]
        self.assertEqual(source["filter_counts"], {
            "accepted": 12, "exact_duplicate": 1, "normalized_duplicate": 1,
            "evaluation_exact_overlap": 0, "evaluation_substring_overlap": 1,
        })
        self.assertEqual(source["rows_scanned"], 15)
        self.assertLessEqual(self.manifest["tokenizer_sample_bytes"], 300)
        self.assertGreaterEqual(self.manifest["tokenizer_sample_bytes"], 297)
        self.assertTrue(self.manifest["frozen"])
        seen = set()
        for split in SPLITS:
            ids = np.fromfile(self.base / f"{split}.bin", dtype="<u2")
            rows = [json.loads(line) for line in (self.base / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(ids), self.manifest["splits"][split]["tokens"])
            eos = SPECIAL_IDS["<|eos|>"]
            expected = [token for row in rows for token in [*self.tokenizer.encode(row["text"]).ids, eos]]
            np.testing.assert_array_equal(ids, expected)
            self.assertFalse(seen.intersection(row["id"] for row in rows))
            seen.update(row["id"] for row in rows)

    def test_special_round_trip_and_role_labels(self):
        self.assertEqual([self.tokenizer.token_to_id(token) for token in SPECIALS], list(range(7)))
        sample = "天气晴朗，hello!\n空格 🧪𠀀"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(sample).ids), sample)
        messages = [{"role": "user", "content": "长江在哪里？"},
                    {"role": "assistant", "content": "长江位于中国。"},
                    {"role": "user", "content": "流向何处？"},
                    {"role": "assistant", "content": "流入东海。"}]
        encoded = encode_conversation(self.tokenizer, messages)
        turn = SPECIAL_IDS["<|end_turn|>"]
        expected = [*self.tokenizer.encode(messages[1]["content"]).ids, turn,
                    *self.tokenizer.encode(messages[3]["content"]).ids, turn, SPECIAL_IDS["<|eos|>"]]
        self.assertEqual([label for label in encoded["labels"] if label != IGNORE_LABEL], expected)
        for index, label in enumerate(encoded["labels"]):
            if label != IGNORE_LABEL:
                self.assertEqual(label, encoded["input_ids"][index])
        prefix = encode_conversation(self.tokenizer, messages[:-1], generation_prompt=True)
        self.assertEqual(prefix["input_ids"][-1], SPECIAL_IDS["<|assistant|>"])
        self.assertTrue(all(label == IGNORE_LABEL for label in prefix["labels"]))
        with self.assertRaises(ValueError):
            encode_conversation(self.tokenizer, [{"role": "user", "content": "<|assistant|>"}], generation_prompt=True)

    def test_multiturn_group_and_complete_final_answer(self):
        short = parse_conversation(multi_row(65), "multi")
        long = parse_conversation(multi_row(65, followup=True), "multi")
        self.assertEqual(conversation_split(short), conversation_split(long))
        self.assertEqual([message["role"] for message in long], ["user", "assistant"] * 3)
        self.assertEqual(long[-1]["content"], "整理审核记录，并提交复核。")
        shorter_size = len(encode_conversation(self.tokenizer, long[-4:])["input_ids"])
        selected, reason = select_window(self.tokenizer, long, "multi", max_tokens=shorter_size)
        self.assertIsNone(reason)
        self.assertEqual(selected[0], long[-4:])
        self.assertEqual(select_window(self.tokenizer, long, "multi", max_tokens=shorter_size - 1),
                         (None, "over_context_length"))

    def test_sft_build_counts_and_json_stream_formats(self):
        self.assertEqual(list(read_json_records(BytesIO(b'[{"a": 1}, {"a": 2}]'))), [{"a": 1}, {"a": 2}])
        source_entries = []
        for kind, repo, name in (("single", "BelleGroup/train_0.5M_CN", "single.json"),
                                 ("multi", "BelleGroup/multiturn_chat_0.8M", "multi.json")):
            groups = {}
            for index in range(10000):
                row = multi_row(index) if kind == "multi" else {
                    "instruction": f"怎样整理编号{index}的收据？", "input": "", "output": "先按日期整理，然后核对金额。",
                }
                split, _ = conversation_split(parse_conversation(row, kind))
                groups.setdefault(split, row)
                if len(groups) == 3:
                    break
            path = self.root / name
            rows = list(groups.values())
            if kind == "single":
                contaminated = {"instruction": self.eval_question, "input": "", "output": "好的。"}
                path.write_text("".join(json_line(row) for row in [contaminated, *rows]), encoding="utf-8")
            else:
                path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            source_entries.append({"repo": repo, "revision": "fixture", "usage": "research only",
                                   "files": [{"path": name, "bytes": path.stat().st_size}]})

        out = self.root / "sft"
        quotas = {kind: dict.fromkeys(SPLITS, 1) for kind in ("single", "multi")}
        result = build_sft(out, source_entries, self.base, self.eval_path, local_root=self.root, quotas=quotas)
        self.assertEqual(result["counts"], dict.fromkeys(SPLITS, 2))
        self.assertEqual(result["tokenizer_sha256"], self.manifest["tokenizer_sha256"])
        counts = Counter()
        for item in result["filter_stats"]:
            counts.update(item["counts"])
        self.assertEqual(counts, {"accepted": 6, "evaluation_exact_overlap": 1, "evaluation_substring_overlap": 0})
        for split in SPLITS:
            rows = [json.loads(line) for line in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["splits"][split]["assistant_tokens"],
                             sum(label != IGNORE_LABEL for row in rows for label in row["labels"]))


if __name__ == "__main__":
    unittest.main()
