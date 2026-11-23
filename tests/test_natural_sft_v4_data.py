"""Focused behavior checks for complete source transforms and candidate formatting."""

import unittest

from tokenizers import Tokenizer

from tinychat.chat_format import SPECIAL_IDS, encode_conversation
from tinychat.natural_sft_v4_candidates import (
    TOKENIZER,
    SourceRow,
    cmrc_candidate,
    dialogue_candidate,
    make_candidate,
    msg,
)
from tinychat.natural_sft_v4_data import pilot_samples, user_keys


def dialogue(*turns):
    messages = [msg(("user", "assistant")[index % 2], content) for index, content in enumerate(turns)]
    return SourceRow("test_source", "original_group", {"source_row_id": 1}, {"messages": messages})


class NaturalSftV4DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER))

    def test_history_dependent_prefix_preserves_original_group_and_all_source_turns(self):
        source = dialogue(
            "我下周搬家，怎么整理书桌比较省事？",
            "整理书桌时，先把充电器和电线扎在一起，再用纸箱装常用的东西。",
            "谢谢你！",
            "不客气。",
            "把上面说的纸箱换成布袋。",
            "整理书桌时，先把充电器和电线扎在一起，再用布袋装常用的东西。",
            "你是谁？",
            "我是一个人工智能助手。",
        )
        row, reason = dialogue_candidate(self.tokenizer, source)
        self.assertIsNone(reason)
        self.assertEqual(row["category"], "multi")
        self.assertEqual(row["source_group_id"], source.group_id)
        self.assertEqual(row["group_id"], source.group_id)
        self.assertEqual(row["messages"], source.original["messages"][:6])
        self.assertEqual(row["original"], source.original)
        self.assertEqual(row["transform"]["message_range_zero_based_half_open"], [0, 6])
        dependency, = row["transform"]["dependency_checks"]
        self.assertEqual(dependency["prior_assistant_index"], 1)
        self.assertEqual(dependency["dependent_user_index"], 4)

    def test_brainstorm_can_support_a_referenced_followup_without_single_turn_category(self):
        source = dialogue(
            "帮我思考一个创意，可以促进人们健康的生活方式。",
            "可以尝试开发一款结合游戏和健身的app，用户通过完成健身动作获得游戏奖励。",
            "好主意！这个app需要包含哪些功能？",
            "可以放入健身动作的示范视频，并加上挑战打卡和组队练习。",
        )
        row, reason = dialogue_candidate(self.tokenizer, source)
        self.assertIsNone(reason)
        self.assertEqual(row["task_family"], "history_object_specifics")
        self.assertEqual(row["messages"], source.original["messages"])

    def test_generic_followup_and_identity_filler_do_not_qualify_as_multi(self):
        source = dialogue(
            "我下周搬家，怎么整理书桌比较省事？",
            "整理书桌时，先把充电器和电线扎在一起，再用纸箱装常用的东西。",
            "能具体一点吗？",
            "当然可以！",
        )
        row, reason = dialogue_candidate(self.tokenizer, source)
        self.assertIsNone(reason)
        self.assertEqual(row["category"], "natural")
        self.assertEqual(row["messages"], source.original["messages"][:2])
        identity = dialogue("你是谁？", "我是一个人工智能助手。")
        self.assertEqual(dialogue_candidate(self.tokenizer, identity), (None, "source_identity"))

    def test_encoded_candidate_keeps_supervised_end_turn_and_rejects_oversize(self):
        source = dialogue("早上好！", "早上好！今天过得怎么样？")
        messages = source.original["messages"]
        row, reason = make_candidate(self.tokenizer, source, messages, "natural", "greeting", {})
        self.assertIsNone(reason)
        encoded = encode_conversation(self.tokenizer, row["messages"], final_eos=False)
        end_turn = SPECIAL_IDS["<|end_turn|>"]
        self.assertEqual(row["token_count"], len(encoded["input_ids"]))
        self.assertEqual(encoded["input_ids"][-1], end_turn)
        self.assertEqual(encoded["labels"][-1], end_turn)
        self.assertNotIn(SPECIAL_IDS["<|eos|>"], encoded["input_ids"])
        long_messages = [messages[0], msg("assistant", "小心轻放。" * 600)]
        self.assertEqual(make_candidate(self.tokenizer, source, long_messages, "natural", "greeting", {}),
                         (None, "over_512_tokens"))

    def test_cmrc_checks_requested_relation_and_visible_original_offsets(self):
        context = "青河图书馆由社区志愿者筹建。青河图书馆于1998年开馆。"
        source = SourceRow("cmrc_train", "same_document", {"source_row_id": "doc_1"}, {
            "id": "doc_1",
            "context": context,
            "question": "青河图书馆是哪一年开馆的？",
            "answers": {"text": ["1998年"], "answer_start": [context.index("1998年")]},
        })
        row, reason = cmrc_candidate(self.tokenizer, source)
        self.assertIsNone(reason)
        self.assertEqual(row["category"], "grounded")
        self.assertEqual(row["group_id"], "same_document")
        begin, end = row["transform"]["passage_chars_zero_based_half_open"]
        self.assertIn(context[begin:end], row["messages"][0]["content"])
        self.assertEqual(row["messages"][1]["content"], "1998年")
        source.original["question"] = "青河图书馆是哪一年闭馆的？"
        self.assertEqual(cmrc_candidate(self.tokenizer, source), (None, "cmrc_no_explicit_local_relation"))
        source.original["answers"]["answer_start"] = [0]
        self.assertEqual(cmrc_candidate(self.tokenizer, source), (None, "invalid_cmrc_answer_offset"))

    def test_cmrc_keeps_named_antecedent_and_records_exact_render_edits(self):
        context = "张小兰（，）一直对绘画有兴趣。她先跟家人学习画画。后来住到了松河村。1989年她开始举办个人画展。"
        answer = "1989年"
        source = SourceRow("cmrc_train", "context-id", {"source_row_id": "doc_2"}, {
            "context": context, "question": "张小兰哪一年开始举办个人画展？",
            "answers": {"text": [answer], "answer_start": [context.index(answer)]},
        })
        row, reason = cmrc_candidate(self.tokenizer, source)
        self.assertIsNone(reason)
        transform = row["transform"]
        self.assertEqual(transform["subject_link"], "named_antecedent_in_contiguous_prefix")
        self.assertIn("张小兰一直对绘画有兴趣。", row["messages"][0]["content"])
        edit, = transform["wiki_render_edits"]
        start, end = edit["chars"]
        self.assertEqual(context[start:end], "（，）")
        self.assertEqual(edit["replacement"], "")

    def test_supplied_document_requires_visible_answer_and_matching_relation(self):
        material = "材料：青河图书馆由社区志愿者筹建，于1998年开馆，服务周边居民。\n问题："
        source = dialogue(material + "青河图书馆是哪一年开馆的？", "1998年。")
        row, reason = dialogue_candidate(self.tokenizer, source)
        self.assertIsNone(reason)
        self.assertEqual(row["task_family"], "document_date")
        start, end = row["transform"]["answer_chars_in_user_zero_based_half_open"]
        self.assertEqual(row["messages"][0]["content"][start:end], "1998年")
        source.original["messages"][0]["content"] = material + "青河图书馆是哪一年闭馆的？"
        row, _ = dialogue_candidate(self.tokenizer, source)
        self.assertIsNone(row)

    def test_natural_instances_allow_action_duration_but_exclude_statistics_and_task_wrapper(self):
        question = "我想把书桌收拾一下，怎么开始？"
        answer = "先拿出10分钟，把书桌上暂时用不着的东西放进一个盒子。"
        row, reason = dialogue_candidate(self.tokenizer, dialogue(question, answer))
        self.assertIsNone(reason)
        self.assertEqual(row["task_family"], "situated_daily_advice")
        row, _ = dialogue_candidate(self.tokenizer, dialogue(question, answer + "这样能提高效率30%。"))
        self.assertIsNone(row)
        row, _ = dialogue_candidate(self.tokenizer, dialogue("根据以下输入，提供一个建议。\n" + question, answer))
        self.assertIsNone(row)
        question = "不太确定怎么跟室友开口。"
        answer = "你想聊的是作息、卫生，还是别的事？"
        row, reason = dialogue_candidate(self.tokenizer, dialogue(question, answer))
        self.assertIsNone(reason)
        self.assertEqual(row["task_family"], "situated_clarifying_question")

    def test_instruction_compaction_preserves_day_and_both_attributes(self):
        question = "用更简洁的方式重新书写下面这句话：\n今天是一个充实而有意义的日子。"
        row, reason = dialogue_candidate(self.tokenizer, dialogue(question, "今日有意义而充实。"))
        self.assertIsNone(reason)
        self.assertEqual(row["task_family"], "meaning_preserving_day_compaction")
        for incorrect in ("明日有意义而充实。", "今日有意义而开心。", "今日很充实。"):
            with self.subTest(answer=incorrect):
                row, _ = dialogue_candidate(self.tokenizer, dialogue(question, incorrect))
                self.assertIsNone(row)

    def test_pilot_spans_sources_and_skips_same_group_variants(self):
        rows = [
            {"id": identifier, "source": source, "category": "grounded", "group_id": group, "token_count": count}
            for identifier, source, group, count in (("a", "x", "one", 40), ("b", "x", "one", 180),
                                                       ("c", "y", "two", 300))
        ]
        sample, counts, coverage = pilot_samples(rows)
        self.assertEqual(counts["grounded"], 2)
        self.assertEqual({row["source"] for row in sample}, {"x", "y"})
        self.assertEqual(len(coverage["grounded"]), 2)
        self.assertEqual(user_keys([msg("user", "第一轮！"), msg("assistant", "gold"), msg("user", "第二轮？")]),
                         ["第一轮", "第二轮", "第一轮|第二轮"])


if __name__ == "__main__":
    unittest.main()
