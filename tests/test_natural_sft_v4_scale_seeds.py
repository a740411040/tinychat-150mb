"""Exercise newly added structural routing, lexical conflict and probe behavior."""

from collections import Counter
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer

from tinychat.chat_format import IGNORE_LABEL, encode_conversation
from tinychat.natural_sft_v4_scale_seeds import (
    LexicalIndex, PROBE_QUOTAS, TOKENIZER, V2_EXPLICIT_TRANSFORMATION, VariantLedger, cmrc_duplicate,
    dependency, fixed_tokens, implicit_dependency, make_probe, material_from_user, reject_user, route_row,
    v2_user_shape,
)
from tinychat.stage1_data import normalized_content


def conversation(*contents):
    return [{"role": ("user", "assistant")[index % 2], "content": text} for index, text in enumerate(contents)]


class ScaleSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER))

    def test_user_material_is_an_exact_span_for_quote_or_table(self):
        quote = "根据这段材料回答：小陈为什么改坐公交？“早上开始下雨，小陈不想淋湿自行车，所以出门时改坐公交。”"
        table = "表中哪个教室周五开放？\n|星期|地点|\n|周五|二楼教室|\n"
        for user, mode in ((quote, "visible_quote_and_question"), (table, "visible_table_and_question")):
            material = material_from_user(user)
            self.assertIsNotNone(material)
            self.assertEqual(material.mode, mode)
            self.assertNotIn("？", user[slice(*material.span)])

    def test_later_material_requires_history_and_stays_in_origin_prefix(self):
        messages = conversation(
            "能帮我回答练习里的阅读问题吗？", "请把原文和问题发给我。",
            "根据下面的材料回答：谁把钥匙送回来了？“李岩在楼梯下发现一串钥匙，问清是谁的以后把钥匙送回了楼上。”", "是李岩。",
        )
        result, reason = route_row({
            "id": "test", "group_id": "original-test", "messages": messages,
            "sources": [{"input": "moss_helpfulness", "license": "test"}],
        }, self.tokenizer)
        self.assertIsNone(reason)
        row, _ = result
        self.assertEqual(row["category"], "grounded")
        self.assertEqual(row["grounded_evidence"]["user_message_index"], 2)
        self.assertEqual(row["original_messages"], messages)
        self.assertEqual(row["rewrite_assistant_turn_indices"], [1, 3])
        for anchor in row["dependency_anchors"][0]["source_spans"]:
            text = messages[anchor["message_index"]]["content"]
            self.assertEqual(text[slice(*anchor["chars_zero_based_half_open"])], anchor["text"])
        encoded = encode_conversation(self.tokenizer, messages, final_eos=False)
        assistant_content = sum(label != IGNORE_LABEL for label in encoded["labels"]) - 2
        self.assertEqual(fixed_tokens(self.tokenizer, messages), len(encoded["input_ids"]) - assistant_content)

    def test_dependency_requires_concrete_previous_content_or_new_constraint(self):
        first = conversation("没量勺怎么调味？", "可以用矿泉水瓶盖临时估量。")
        true, generic, switch = [conversation(text)[0] for text in (
            "你说的矿泉水瓶盖具体怎么量？", "还有别的建议吗？", "对了，朋友今天来我家喝茶。",
        )]
        subtype, reason, anchors = dependency(first, true)
        self.assertEqual((subtype, reason, anchors[0]["text"]), ("daily", "referent_verbatim_in_earlier_turn", "矿泉水瓶盖"))
        self.assertIsNone(dependency(first, generic))
        self.assertIsNone(dependency(first, switch))

    def test_implicit_comparison_anchors_the_previous_list_and_preserves_user_bytes(self):
        messages = conversation(
            "我想学习英语听力，有哪些办法？",
            "1. 每天听一段短音频，记下没听懂的句子。\n2. 拿文字稿对照听一遍，标记连读。\n3. 回顾上一周的练习笔记。",
            "上述方法中哪种比较有效？  ",
            "可以结合你的目标来比较。",
        )
        anchor, reason = implicit_dependency(messages)
        self.assertIsNone(reason)
        self.assertEqual(anchor["operation"], "compare_or_choose")
        self.assertEqual(len([span for span in anchor["source_spans"] if span["message_index"] == 1]), 3)
        for span in anchor["source_spans"]:
            self.assertEqual(messages[span["message_index"]]["content"][slice(*span["chars_zero_based_half_open"])],
                             span["text"])
        routed, reason = route_row({"id": "test", "group_id": "g", "messages": messages,
                                    "sources": [{"input": "moss_helpfulness", "license": "test"}]},
                                   self.tokenizer, implicit=anchor)
        self.assertIsNone(reason)
        row = routed[0]
        self.assertEqual(row["dependency_anchors"][0]["anchor_type"], "implicit_reference_candidate")
        self.assertEqual(row["user_messages"], messages[::2])
        self.assertEqual(row["rewrite_assistant_turn_indices"], [1, 3])

    def test_implicit_novelty_asks_for_another_actual_option(self):
        first, reply = "我想少忘带钥匙，该怎么做？", "可以把钥匙挂在门口；也可以在手机上设一个出门提醒。"
        anchor, reason = implicit_dependency(conversation(first, reply, "那还有其他办法吗？", "可以和已有做法比较。"))
        self.assertIsNone(reason)
        self.assertEqual(anchor["operation"], "avoid_repeating_options")
        self.assertEqual(implicit_dependency(conversation(first, "这个问题很常见。", "还有其他办法吗？", "可以。"))[1],
                         "implicit_no_locatable_prior_action_set")

    def test_implicit_route_rejects_closed_risk_and_self_contained_or_locked_questions(self):
        first, reply = "学摄影从哪里开始？", "先设置相机并练习对焦；也可以从观察自然光开始。"
        self.assertEqual(implicit_dependency(conversation(
            first, reply, "我的选择是拍照和画画，这两种方法哪种更合适？", "要看爱好。",
        ))[1], "implicit_topic_switch_or_self_contained_reference")
        self.assertEqual(implicit_dependency(conversation(
            "工作中有 burnout 应该怎么办？", reply, "上述办法具体怎么用？", "可以。",
        ))[1], "implicit_health_or_professional_risk")
        self.assertEqual(implicit_dependency(conversation(
            first, "我曾经去南方采风，后来我开始练习摄影。", "你亲自使用这些技巧的经历能具体说说吗？", "可以。",
        ))[1], "implicit_followup_locks_unsupported_personal_experience")

    def test_route_variant_keeps_existing_natural_group_and_rejects_second_multi(self):
        initial = {
            "id": "natural", "source_group": "group-1", "original_source_group": "group-1", "category": "natural",
            "deduplication": {"primary_user_normalized": "怎么安排周末的学习时间"},
        }
        with patch("tinychat.natural_sft_v4_scale_seeds.load_protected", return_value=(LexicalIndex(), {}, 0)):
            ledger = VariantLedger([initial], "unused")
        messages = conversation("怎么安排周末的学习时间？", "先把任务分类，再做简短练习。",
                                "那么这些建议具体怎么做？", "给这些措施配上执行步骤。")
        candidate = {"id": "multi-v2", "source_group": "group-2", "original_messages": messages}
        self.assertIsNone(ledger.add(candidate))
        self.assertEqual(candidate["source_group"], ledger.root(initial["source_group"]))
        self.assertEqual(candidate["competing_route_categories"], ["natural"])
        self.assertEqual(ledger.add({"id": "duplicate", "source_group": "group-3", "original_messages": messages}),
                         "existing_multi_variant_or_near_duplicate")

    def test_v2_makes_observable_packet_constraints_explicit(self):
        forms = {
            "写一篇文章，篇幅1000字以上。": "explicit_minimum_length_exceeds_response_budget",
            "最多能在一个微信公众平台绑定多少第三方平台？": "unverified_current_platform_limit",
            "请根据段落回答。\nAnswer: 之前的答案": "pseudo_answer_or_mixed_script_corruption",
            "最近在看《openhagen》的剧，你说好看吗？": "unresolved_work_reference_with_no_visible_context",
        }
        for user, reason in forms.items():
            self.assertEqual(v2_user_shape(conversation(user, "旧答。")), reason)
        self.assertIsNone(v2_user_shape(conversation("请用500字以内讲一个故事。", "旧答。")))
        self.assertIsNotNone(V2_EXPLICIT_TRANSFORMATION.search("用更轻松的语气重写下面的文章：旧文本。"))

    def test_lexical_index_reports_exact_containment_and_allowed_document_group(self):
        raw = normalized_content("旅行当天出发前检查背包和水杯还要留出赶车的时间安排午饭")
        index = LexicalIndex()
        index.add(raw, "existing", "group-a")
        self.assertEqual(index.match(raw), ("exact", "existing", "group-a"))
        self.assertEqual(index.match("第一段" + raw + "出行计划"), ("containment", "existing", "group-a"))
        self.assertIsNone(index.match(raw, allow_group="group-a"))
        index.add("另一条比较简单的材料", "another", "group-b")
        hit = cmrc_duplicate("为什么他要出门", [1, 3], [("她做了什么", [1, 3], "q-one")])
        self.assertEqual(hit, ("same_document_same_annotation_span", "q-one"))
        self.assertIsNone(cmrc_duplicate("现场哪一种运动", [100, 105], [("文章里说他从哪里来", [1, 3], "q-one")]))

    def test_exclusions_focus_on_unsafe_questions_and_broken_history(self):
        self.assertEqual(reject_user("你刚才说得对，我们继续吧。", first=True), "missing_origin_history")
        self.assertEqual(reject_user("我希望你替我支付这笔款。"), "requires_physical_or_external_action")
        self.assertEqual(reject_user("请你替我支付这笔款。"), "requires_physical_or_external_action")
        self.assertEqual(reject_user("帮我找出今天的实时股价。"), "time_sensitive_fact_or_price")
        self.assertIsNone(reject_user("根据材料回答。医生发现了一本旧书。", substantive="他发现了什么？"))

    def test_probe_keeps_quotas_and_withholds_previous_groups(self):
        rows = []
        for category in PROBE_QUOTAS:
            for number in range(64):
                source = number % 4
                identifier = f"{category}-{number}"
                row = {
                    "id": identifier, "source_group": identifier, "category": category,
                    "seen_in_previous_pilot": number == 0,
                    "route_reason": "test", "tokens_without_assistant_content": 35 + number,
                    "sources": [{"input": f"source-{source}", "repo": f"repo-{source}"}],
                }
                if category == "grounded":
                    row["grounded_evidence"] = {"source_type": "cmrc_complete_context" if number % 2 else "table"}
                if category == "multi":
                    row["multi_subtype"] = "document" if number % 2 else "daily"
                rows.append(row)
        probe = make_probe(rows)
        self.assertEqual([row["id"] for row in probe], [row["id"] for row in make_probe(rows)])
        self.assertEqual(Counter(row["category"] for row in probe), Counter(PROBE_QUOTAS))
        self.assertEqual(Counter(row["pilot_batch"] for row in probe), Counter({batch: 16 for batch in range(1, 7)}))
        self.assertEqual(len({row["source_group"] for row in probe}), 96)
        self.assertFalse(any(row["seen_in_previous_pilot"] for row in probe))
        self.assertLessEqual(sum(row.get("multi_subtype") == "document" for row in probe), 12)


if __name__ == "__main__":
    unittest.main()
