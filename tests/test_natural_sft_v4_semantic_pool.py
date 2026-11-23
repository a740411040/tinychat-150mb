"""Exercise source fidelity, complete prefix accounting and group deduplication."""

import unittest

from tokenizers import Tokenizer

from tinychat.chat_format import IGNORE_LABEL, SPECIAL_IDS, encode_conversation
from tinychat.natural_sft_v4_semantic_pool import (
    CMRC_PREFIX, CMRC_SEPARATOR, TOKENIZER, SourceInput, build_pool, cmrc_render,
    msg, parse_moss_exact, prefix_options, structural_reject,
)


def conversation(*contents):
    return [msg(("user", "assistant")[index % 2], content) for index, content in enumerate(contents)]


def source(name, locator, messages, origin=None):
    return SourceInput(name, {"source_row_id": locator}, messages, {"type": "test"}, origin)


class SemanticPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER))

    def test_prefix_options_match_frozen_no_eos_formatter(self):
        messages = [msg("system", "原系统提示。"), *conversation(
            "附近有什么好逛的？", "可以去社区图书馆。", "周末也想去。", "可以先查看周末开放时间。",
            "再具体些。", "这里是很长的回答。" * 500, "下个问题", "短回答",
        )]
        options = prefix_options(self.tokenizer, messages)
        self.assertEqual([option["message_count"] for option in options], [3, 5])
        for option in options:
            encoded = encode_conversation(self.tokenizer, messages[:option["message_count"]], final_eos=False)
            self.assertEqual(option["tokens"], len(encoded["input_ids"]))
            self.assertEqual(option["supervised_tokens"], sum(label != IGNORE_LABEL for label in encoded["labels"]))
            self.assertNotIn(SPECIAL_IDS["<|eos|>"], encoded["input_ids"])
            self.assertEqual(encoded["input_ids"][-1], SPECIAL_IDS["<|end_turn|>"])

    def test_overlong_or_structurally_broken_first_turn_never_skips_history(self):
        long = conversation("问题", "这里是很长的回答。" * 500, "下个问题", "短回答")
        bad = conversation("问题", "<|end_turn|>", "下个问题", "短回答")
        self.assertEqual(prefix_options(self.tokenizer, long), [])
        self.assertEqual(structural_reject(bad), "reserved_chat_token_in_content")
        rows, _, counts = build_pool(self.tokenizer, loaders=(lambda: iter([
            source("example", 1, long), source("example", 2, bad),
        ]),))
        self.assertFalse(rows)
        self.assertEqual(counts["totals"]["rejections"], {
            "first_complete_pair_over_context": 1, "reserved_chat_token_in_content": 1,
        })

    def test_exact_aliases_and_original_or_first_question_variants_share_group(self):
        pair = conversation("带什么去公园？", "带伞。")
        extended = [*pair, *conversation("为什么？", "下午可能下雨。")]
        rows, _, counts = build_pool(self.tokenizer, loaders=(lambda: iter([
            source("cache", 1, pair, ("original", 1)),
            source("legacy", 2, pair, ("original", 2)),
            source("cache", 3, extended, ("original", 1)),
            source("other", 4, conversation(" 带什么去公园?", "带水。"), ("original", 4)),
            source("other", 5, conversation("另一个原始开头。", "原始组相同。"), ("original", 2)),
        ]),))
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row["group_id"] for row in rows}), 1)
        self.assertEqual([record["source_row_id"] for record in rows[0]["sources"]], [1, 2])
        self.assertEqual(len(rows[1]["prefix_options"]), 2)
        self.assertEqual(counts["totals"]["groups_with_prefix_option"], {"one_turn": 1, "two_or_more_turns": 1})
        self.assertEqual(counts["source_overlap"]["exact_prefixes_shared_between_inputs"],
                         [{"sources": ["cache", "legacy"], "prefixes": 1}])

    def test_moss_removes_only_source_framing(self):
        raw = "[Human]:  我问。\r\n<eoh> [MOSS]:  答案！  <eoa>"
        messages, spans, reject = parse_moss_exact(raw, 1)
        self.assertIsNone(reject)
        self.assertEqual(messages, conversation(" 我问。\r\n", " 答案！  "))
        self.assertEqual([raw[start:end] for start, end in spans], [" 我问。\r\n", " 答案！  "])
        self.assertEqual(parse_moss_exact(raw + "残余内容", 1)[2], "malformed_moss_markup")

    def test_cmrc_uses_whole_context_and_original_verified_annotation(self):
        context, question = "甲在北京，乙在上海。", "题目原文？"
        annotations = {"text": ["上海", "北京", "上海"], "answer_start": [0, 2, 7]}
        messages, transform, reject = cmrc_render(context, question, annotations)
        self.assertIsNone(reject)
        self.assertEqual(messages, conversation(CMRC_PREFIX + context + CMRC_SEPARATOR + question, "北京"))
        self.assertEqual(transform["source_annotations"], annotations)
        self.assertEqual(transform["offset_exact_annotation_indices_zero_based"], [1, 2])
        self.assertEqual(transform["answer_chars_zero_based_half_open"], [2, 4])


if __name__ == "__main__":
    unittest.main()
