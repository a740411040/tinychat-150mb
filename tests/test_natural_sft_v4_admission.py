"""Check the neutral-prefix and admission structure, without assigning semantic quality."""

from collections import Counter
import unittest

from tokenizers import Tokenizer

from tinychat.chat_format import encode_conversation
from tinychat.natural_sft_v4_admission import (
    CMRC_PREFIX, CMRC_SEPARATOR, ROUND2_TEXT_DELIVERABLE, TOKENIZER, LexicalIndex, PriorFrame,
    admission_payload, candidate_prefix_options,
    cmrc_document_keys, cmrc_material_findings, guard_prefixes, make_micro, sampling_signals, source_key,
    round2_instruction_challenges, round2_multi_signal, validate_admission, validate_checker_context,
    validation_challenges,
)
from tinychat.stage1_data import normalized_content


def dialogue(*texts):
    return [{"role": ("user", "assistant")[index % 2], "content": text} for index, text in enumerate(texts)]


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER))

    def candidate(self, *texts):
        messages = dialogue(*texts)
        return {
            "id": "test",
            "source_group": "test-source-group",
            "sources": [{"repo": "test", "revision": "test", "local_file": "test"}],
            "original_messages": messages,
            "prefix_options": candidate_prefix_options(self.tokenizer, messages),
            "sampling_hint": "multi",
        }

    def response(self, category, end, **overrides):
        return {
            "id": "test", "decision": "accept", "category": category, "message_count": end,
            "reason": "跟随可核对的原题约束。", "evidence": [], "dependencies": [], "multi_subtype": None,
            "requirements": [{"assistant_message_index": index, "text": "提出适用于本轮的办法。"}
                             for index in range(1, end, 2)], **overrides,
        }

    def test_old_answer_length_does_not_remove_a_leading_prefix(self):
        row = self.candidate("半小时内怎样准备出门？", "先看看天气。" * 450, "要下雨，鞋子怎么选？", "选择防滑鞋。")
        self.assertGreater(len(encode_conversation(self.tokenizer, row["original_messages"][:2], final_eos=False)["input_ids"]), 512)
        self.assertEqual([opt["message_count"] for opt in row["prefix_options"]], [2, 4])
        result = validate_admission(row, self.response("natural", 2), self.tokenizer)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["admitted_seed"]["original_messages"], row["original_messages"][:2])

    def test_admission_can_see_only_explicit_role_content_and_bounded_options(self):
        row = self.candidate("今天好累。", "休息一下。")
        row["original_messages"][0]["debug"] = "internal"
        row["prefix_options"][0]["note"] = "internal"
        payload = admission_payload(row)
        self.assertEqual(set(payload), {"id", "original_messages", "prefix_options"})
        self.assertEqual(set(payload["original_messages"][0]), {"role", "content"})
        self.assertEqual(set(payload["prefix_options"][0]),
                         {"message_count", "fixed_tokens", "assistant_content_token_budget"})

    def cmrc_candidate(self, passage, question="这段材料说明了什么？"):
        row = self.candidate(CMRC_PREFIX + passage + CMRC_SEPARATOR + question, "原回复。")
        row["sources"][0].update({
            "input": "cmrc_train", "source_document_id": "DOC",
            "transform": {"passage_chars_zero_based_half_open": [0, len(passage)]},
        })
        row.update({"document_group": "cmrc-doc", "possible_sampling_hints": ["grounded"]})
        return row

    def test_cmrc_catches_only_explicit_markup_and_empty_prose_notes_in_material(self):
        row = self.cmrc_candidate("罂粟花（），此处有 13<nowiki>幺</nowiki>，以及 &lt;nowiki&gt; 标记。")
        original = row["original_messages"][0]["content"]
        findings = cmrc_material_findings(row)
        parts = [original[slice(*item["chars_zero_based_half_open"])] for item in findings]
        self.assertEqual(set(parts), {"（）", "<nowiki>", "</nowiki>", "&lt;nowiki&gt;"})
        self.assertEqual(row["original_messages"][0]["content"], original)
        self.assertEqual(cmrc_document_keys(row), {("test", "test", "DOC")})
        clean = self.cmrc_candidate("书名《植物志》。花（俗称小花），旁注“样本”。函数 f() 可无参数。", "字面符号 <nowiki> 怎么处理？")
        self.assertEqual(cmrc_material_findings(clean), [])
        self.assertIn("normal_parentheses", validation_challenges(clean))
        self.assertIn("normal_quote", validation_challenges(clean))

    def test_cmrc_refuses_a_passage_boundary_not_backed_by_full_source(self):
        row = self.cmrc_candidate("这是一段完整材料。")
        row["sources"][0]["transform"]["passage_chars_zero_based_half_open"][1] -= 1
        self.assertEqual(cmrc_material_findings(row), [
            {"reason": "cmrc_unverifiable_full_passage_boundary", "message_index": 0, "chars_zero_based_half_open": None},
        ])

    def test_rewrite_obligations_and_local_challenge_tags_survive_only_local_admission(self):
        row = self.candidate(
            "重写下面两段：\n会议确认了下周的活动安排和参与方式。\n报名仍需在周五前完成并保留确认邮件。",
            "旧的第一答。",
        )
        row["possible_sampling_hints"] = ["instruction"]
        row["validation_challenges"] = validation_challenges(row)
        requirements = [
            {"assistant_message_index": 1, "text": "保留会议确认的事。"},
            {"assistant_message_index": 1, "text": "保留报名限期和邮件要求。"},
        ]
        result = validate_admission(row, self.response("instruction", 2, requirements=requirements), self.tokenizer)
        self.assertNotIn("validation_challenges", admission_payload(row))
        self.assertEqual(result["admitted_seed"]["validation_challenges"], ["full_rewrite"])
        self.assertEqual(result["admitted_seed"]["admission_requirements"], requirements)

    def test_simplify_hint_requires_a_user_request_to_make_expression_easier(self):
        cases = (
            ("总结下面这段文本的摘要， 我们经常会听到人们谈论情商，但是什么是情商呢？"
             "情商是指一个人在处理人际交往、情感表达、自我管理等方面的能力。通俗地说，情商高的人擅长与人相处。", False),
            ("编写一个简单的日记条目，并将其转换为一篇文学散文。\n日期：2021年7月15日\n日记条目：今天早上在公园里跑步。", False),
            ("编辑下面这句话，使它更流畅和易于理解：“这个电子设备负责自动双向翻译。”\n这个电子设备负责自动的双向翻译。", True),
            ("用通俗易懂的语言重写下面的文章：\n我们怀着最大的诚意和尊重，正式邀请您出席即将到来的盛会。", True),
            ("把下面的文字总结成一段通俗易懂的摘要。", False),
            ("将这句话改写成简单日记。", False),
        )
        for user, should_tag in cases:
            with self.subTest(user=user):
                row = self.candidate(user, "旧的第一答。")
                row["possible_sampling_hints"] = ["instruction"]
                self.assertEqual("simplify" in validation_challenges(row), should_tag)

    def test_multi_can_be_admitted_without_a_prior_lexical_anchor(self):
        row = self.candidate("我下班太晚，来不及走有灯的大路。", "先看看哪条街还能走。",
                             "我已经有两种可选了。", "比较能走且照明好的。")
        view = self.response("multi", 4, multi_subtype="daily", dependencies=[
            {"user_message_index": 2, "history_message_index": 0, "history_quote": "来不及走有灯的大路",
             "missing_without_history": "无法判断两个路线要解决下班太晚且道路照明的困境。"},
        ])
        result = validate_admission(row, view, self.tokenizer)
        self.assertEqual(result["admitted_seed"]["rewrite_assistant_turn_indices"], [1, 3])
        self.assertEqual(result["admitted_seed"]["user_messages"], row["original_messages"][::2])
        self.assertEqual(result["admitted_seed"]["admission_dependencies"], view["dependencies"])
        self.assertNotIn("semantic_pass", result["admitted_seed"])

    def test_multi_rejects_no_history_non_user_target_or_empty_missing_reason(self):
        row = self.candidate("我要上晚班。", "先报备。", "那要带什么？", "带卡。")
        for position, prior, quote, missing in (
            (2, 2, "那要带什么", "说明缺失"), (1, 0, "晚班", "说明缺失"), (2, 0, "晚班", ""),
            (2, 0, "不存在的引用", "说明缺失"),
        ):
            view = self.response("multi", 4, multi_subtype="daily", dependencies=[
                {"user_message_index": position, "history_message_index": prior, "history_quote": quote,
                 "missing_without_history": missing},
            ])
            with self.assertRaises(ValueError):
                validate_admission(row, view, self.tokenizer)
        with self.assertRaises(ValueError):
            validate_admission(row, self.response("multi", 2, multi_subtype="daily"), self.tokenizer)

    def test_non_multi_cannot_keep_two_exchanges_under_any_category(self):
        row = self.candidate("我下周搬家，该怎么收拾？", "先整理常用物品。", "材料：周一关闭库房。", "周一关闭。")
        for category in ("natural", "grounded", "instruction"):
            with self.subTest(category=category), self.assertRaisesRegex(ValueError, "only the original first exchange"):
                evidence = [{"message_index": 2, "quote": "周一关闭库房"}] if category == "grounded" else []
                validate_admission(row, self.response(category, 4, evidence=evidence), self.tokenizer)

    def test_non_multi_first_exchange_cannot_keep_history_metadata(self):
        row = self.candidate("我下周搬家，该怎么收拾？", "先整理常用物品。", "那纸箱怎么分？", "按房间分。")
        view = self.response("natural", 2, dependencies=[
            {"user_message_index": 2, "history_message_index": 0, "history_quote": "下周搬家",
             "missing_without_history": "没有首问便不知道纸箱用来搬家。"},
        ])
        with self.assertRaisesRegex(ValueError, "cannot retain dependency metadata"):
            validate_admission(row, view, self.tokenizer)

    def test_grounded_quote_must_match_visible_user_material_and_exact_text(self):
        row = self.candidate("材料：水杯在左边。问：水杯在哪里？", "我觉得水杯在桌上。")
        for item in ({"message_index": 1, "quote": "水杯在桌上"}, {"message_index": 0, "quote": "水杯在  左边"}):
            with self.assertRaises(ValueError):
                validate_admission(row, self.response("grounded", 2, evidence=[item]), self.tokenizer)
        accepted = validate_admission(row, self.response("grounded", 2, evidence=[
            {"message_index": 0, "quote": "水杯在左边"},
        ]), self.tokenizer)
        self.assertEqual(accepted["admitted_seed"]["source_group"], "test-source-group")

    def test_reject_response_is_recorded_and_wrong_id_or_incomplete_coverage_fails(self):
        row = self.candidate("搬家后该怎么收拾？", "先整理厨房。")
        reject = {"id": "test", "decision": "reject", "reason": "原题无法保真。"}
        self.assertIsNone(validate_admission(row, reject, self.tokenizer)["admitted_seed"])
        for reply in (dict(reject, id="missing"), self.response("natural", 2, requirements=[]),
                      self.response("natural", True), dict(reject, evidence=[{}])):
            with self.assertRaises(ValueError):
                validate_admission(row, reply, self.tokenizer)

    def test_protected_second_user_removes_only_that_leading_option(self):
        second = "你能逐条检查这些课程练习表上面所有学生报名时填错的组别吗"
        row = self.candidate("下周怎么安排时间？", "先排课。", second, "我来逐条检查。")
        index = LexicalIndex()
        index.add(normalized_content(second), "old", "protected")
        prior = PriorFrame({}, {}, {}, LexicalIndex(), {}, Counter())
        opts, conflict = guard_prefixes(row["original_messages"], row["prefix_options"], index, {}, prior)
        self.assertEqual([option["message_count"] for option in opts], [2])
        self.assertEqual(conflict, ("protected_user_exact", "old"))

    def test_table_columns_are_not_sampled_as_grounded_by_material_alone(self):
        messages = dialogue("找出下面表格的姓名列，用逗号分隔：\n|姓名|地点|\n|甲|图书馆|", "甲")
        hints, _, _ = sampling_signals(messages, {"input": "local"})
        self.assertEqual(hints, ["instruction"])

    def test_same_numeric_source_id_in_another_moss_file_is_distinct(self):
        source = {"repo": "pinned", "revision": "same", "local_file": "helpfulness", "source_row_id": 7}
        self.assertNotEqual(source_key(source), source_key(dict(source, local_file="honesty")))

    def test_round2_checker_context_validates_zero_based_adjacent_reply_evidence(self):
        row = self.candidate("我周末只有二十分钟整理工作台。", "先清理桌面。", "那能再快点吗？",
                             "二十分钟内先收好桌面工具。")
        messages = [dict(message, message_index=index) for index, message in enumerate(row["original_messages"])]
        audit = {"user_message_index": 2, "history_message_index": 0, "history_quote": "只有二十分钟",
                 "missing_without_history": "不会知道要整理工作台以及可用时间",
                 "answer_quote": "二十分钟内先收好桌面工具"}
        response = {"category": "multi", "multi_subtype": "daily", "context_audits": [audit]}
        self.assertEqual(validate_checker_context(messages, response, "daily"), [])
        self.assertEqual(validate_checker_context(messages, response, "document"), ["checker_multi_subtype_disagreement"])
        for change in ({"history_message_index": 2}, {"user_message_index": 1}, {"history_quote": "二十分钟左右"},
                       {"missing_without_history": " "}, {"answer_quote": "先清理桌面"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_checker_context(messages, dict(response, context_audits=[dict(audit, **change)]), "daily")
        with self.assertRaisesRegex(ValueError, "cover every later user"):
            validate_checker_context(messages + dialogue("还要带走工具呢？", "分类收好。"),
                                     response, "daily")

    def test_round2_non_multi_checker_cannot_keep_context_fields(self):
        messages = dialogue("我想整理工作台。", "先清理桌面。")
        self.assertEqual(validate_checker_context(messages, {"category": "natural"}, None), [])
        self.assertEqual(validate_checker_context(messages, {
            "category": "natural", "multi_subtype": None, "context_audits": [],
        }, None), [])
        for fields in ({"multi_subtype": "daily"}, {"context_audits": [{}]}, {"context_audits": None}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_checker_context(messages, {"category": "natural", **fields}, None)

    def test_round2_local_cues_prioritize_actual_deliverable_and_allow_overlapping_styles(self):
        paragraph = "生成一个关于瑜伽冥想的段落，介绍它的好处以及如何开始练习。"
        self.assertIsNotNone(ROUND2_TEXT_DELIVERABLE.search(paragraph))
        doc = self.candidate(paragraph, "先在安静的角落坐下，放松肩膀，留意自然呼吸。"
                             "刚开始时可以只练几分钟，走神了就温和地把注意力带回呼吸。",
                             "请从这篇文章中提取练习步骤。", "找安静角落，坐下并关注呼吸。")
        self.assertEqual(round2_multi_signal(doc)[0], "document")
        daily = self.candidate("我周末要准备考试，但自习室人多，有什么办法？", "可以提前预约。",
                               "那如果不想预约，有没有更安静的办法？", "选择早一点的时段。")
        self.assertEqual(round2_multi_signal(daily)[0], "daily")
        simplify = self.candidate("用通俗易懂的语言重写下面一段：\n本次活动由本组负责组织。", "旧答。")
        simplify["possible_sampling_hints"] = ["instruction"]
        self.assertTrue({"simplify", "full_rewrite"} <= set(round2_instruction_challenges(simplify)))
        formalize = self.candidate("用更正式的语气重写下句：\n我们负责举办这次活动。", "旧答。")
        formalize["possible_sampling_hints"] = ["instruction"]
        self.assertTrue({"formalize", "full_rewrite"} <= set(round2_instruction_challenges(formalize)))

    def test_micro_competing_hint_choices_cannot_duplicate_a_group(self):
        rows = []
        for i in range(90):
            candidate = self.candidate(f"最近准备学习项目第{i}小节，有哪些办法？", "先读讲义。",
                                       f"我刚拿到新的第{i}小节讲义。", "先确定重点。")
            candidate.update({
                "id": str(i), "source_group": str(i // 3), "possible_sampling_hints": ["natural", "grounded", "multi", "instruction"],
                "multi_sampling_subtype": "daily", "multi_has_lexical_hint": False,
            })
            candidate["sources"][0]["input"] = "source"
            rows.append(candidate)
        sample, stats, _ = make_micro(rows)
        self.assertEqual(len(sample), len({row["source_group"] for row in sample}))
        self.assertLessEqual(len(sample), 64)
        self.assertTrue(stats["hint_shortfalls"])  # A near-identical family cannot pad a shortage.
        for i, row in enumerate(rows):
            row["possible_sampling_hints"] = ["multi"]
            row["multi_sampling_subtype"] = "daily" if i == 0 else "document"
        sample, stats, _ = make_micro(rows)
        self.assertEqual(stats["micro_multi_hints_by_subtype"]["daily"], 1)
        self.assertLessEqual(stats["micro_multi_hints_by_subtype"]["document"], 1)


if __name__ == "__main__":
    unittest.main()
