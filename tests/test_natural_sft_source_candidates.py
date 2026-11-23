import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tokenizers import Tokenizer

from tinychat.natural_sft_source_candidates import (
    TOKENIZER,
    build_source,
    cmrc_candidate,
    moss_candidate,
    verify_candidate,
)


class NaturalSftSourceCandidatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER))

    def moss(self, *turns):
        text = "\n".join(f"[Human]: {question}<eoh>\n[MOSS]: {answer}<eoa>" for question, answer in turns)
        return moss_candidate(self.tokenizer, {"num_turns": len(turns), "plain_text": text, "id": 18},
                              "moss_helpfulness", 0)

    def test_moss_retains_only_contiguous_dependent_turns(self):
        pairs = (
            ("我想周末收拾一下书桌，该从哪下手？", "先把书桌上的文具和书本分开，再把常用的东西放近些。"),
            ("你说的文具应该怎么放？", "常用的笔可以放进一个笔筒，剪刀和订书机可以收在抽屉里。"),
            ("你觉得我明天穿什么去散步？", "穿方便活动的衣服和一双合脚的运动鞋。"),
        )
        result, reason = self.moss(*pairs)
        self.assertIsNone(reason)
        self.assertEqual(result["category"], "multi")
        self.assertEqual(result["source_evidence"]["dependency"], ["follow_up"])
        self.assertEqual([item["content"] for item in result["messages"]], [text for pair in pairs[:2] for text in pair])
        self.assertEqual(result["source_group_id"], self.moss(pairs[0])[0]["source_group_id"])
        verify_candidate(self.tokenizer, result)

    def test_moss_rejects_encyclopedic_natural_and_source_identity(self):
        result, reason = self.moss(("什么是盆地？", "四周高、中间低的地形称为盆地。"))
        self.assertIsNone(result)
        self.assertEqual(reason, "outside_intended_tasks")
        result, reason = self.moss(("你好！", "你好，我的名字是MOSS。"))
        self.assertIsNone(result)
        self.assertEqual(reason, "source_identity")

    def test_moss_prefix_keeps_original_file_metadata_and_source_index(self):
        row = {"id": 173, "num_turns": 1, "plain_text":
               "[Human]: 我想把衣柜收拾一下，有什么建议？<eoh>\n[MOSS]: 可以先把当季常穿的衣物放在最顺手的地方。<eoa>"}
        source_range = {"byte_start": 0, "byte_end_inclusive": 1023, "bytes": 1024,
                        "complete_original_records": True, "local_file": "prefix.jsonl"}
        with TemporaryDirectory() as directory:
            input_file = Path(directory) / "prefix.jsonl"
            input_file.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            output = io.StringIO()
            report = build_source(self.tokenizer, "moss_helpfulness", input_file, output, source_range=source_range)
        candidate = json.loads(output.getvalue())
        self.assertEqual(report["counts"]["accepted:natural"], 1)
        self.assertEqual(candidate["source_file"], "zh_helpfulness.json")
        self.assertEqual(candidate["source_row_id"], "173")
        self.assertEqual(candidate["source_evidence"]["source_row_index"], 0)
        self.assertEqual(candidate["source_evidence"]["source_range"], source_range)
        verify_candidate(self.tokenizer, candidate)

    def test_cmrc_crops_complete_sentences_and_rejects_wrong_offset(self):
        prefix = "大家散步时经常经过这座桥，桥上总能看到附近居民闲聊。" * 28
        answer = "梧桐树"
        sentence = "桥边种着一棵梧桐树，夏天树荫下很凉快。"
        raw = prefix + sentence + prefix
        start = len(prefix) + sentence.index(answer)
        row = {"context": raw, "id": "TRAIN_12_QUERY_3", "question": "桥边种着什么树？",
               "answers": {"text": [answer], "answer_start": [start]}}
        result, reason = cmrc_candidate(self.tokenizer, row, "cmrc_train", 0, evidence_v2=True)
        self.assertIsNone(reason)
        self.assertEqual(result["source_group_id"], "cmrc2018:document:TRAIN_12")
        self.assertLessEqual(result["tokens"], 512)
        source = result["source_evidence"]
        self.assertEqual(source["answer_start"], start)
        self.assertEqual(source["evidence_sentence"], sentence)
        self.assertIn(sentence, result["messages"][1]["content"])
        left, right = source["passage_start"], source["passage_end"]
        self.assertLess(right - left, len(raw))
        self.assertIn(sentence, raw[left:right])
        self.assertIn(raw[left:right], result["messages"][0]["content"])
        verify_candidate(self.tokenizer, result)
        row["answers"]["answer_start"][0] -= 1
        self.assertEqual(cmrc_candidate(self.tokenizer, row, "cmrc_train", 0), (None, "no_verified_answer_offset"))


if __name__ == "__main__":
    unittest.main()
