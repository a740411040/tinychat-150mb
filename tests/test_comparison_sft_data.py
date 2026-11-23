import unittest
from pathlib import Path

from tokenizers import Tokenizer

from tinychat.chat_format import IGNORE_LABEL, SPECIAL_IDS, encode_conversation
from tinychat.comparison_sft_data import (
    PROGRAM_QUOTAS, QUOTAS, ConflictIndex, belle_quality, comparison_quotas, grounded_answer_reason,
    narrow_groups, program_candidates, pt_chunks, render_program, row_kind, validate_labels,
)


class ComparisonSftDataTest(unittest.TestCase):
    def test_containment_both_directions_and_verified_jaccard(self):
        short = "中午出发前请先查看登记册确认库房位置并核对批次编号"
        long = "本周的提醒内容" + short + "登记完成以后请签字并把记录按时间顺序放回左侧抽屉"
        for keys in ((short, long), (long, short)):
            index = ConflictIndex(containment=True)
            index.add(keys[0])
            self.assertEqual(index.match(keys[0]), "exact")
            self.assertEqual(index.match(keys[1]), "substring")
        index = ConflictIndex()
        index.add(long)
        self.assertEqual(index.match(long.replace("午", "晚", 1)), "jaccard")
        self.assertIsNone(index.match("请从仓库取出新登记的展品并放回展柜"))

    def test_short_eval_containment_covers_every_anchor_boundary(self):
        prompt = "图书馆志愿者请在闭馆之前复核当天借出的全部书目"
        for count in (6, 11, 12, 23):
            with self.subTest(count=count):
                embedded = prompt[:count]
                longer = "我遇到一件事想请你帮忙" + embedded + "可以告诉我接下来如何操作吗"
                for needle, query in ((embedded, longer), (longer, embedded)):
                    index = ConflictIndex(containment=True, substring_minimum=6)
                    index.add(needle)
                    self.assertEqual(index.match(query), "substring")
        index = ConflictIndex(containment=True, substring_minimum=6)
        index.add(prompt[:5])
        self.assertIsNone(index.match("请看看" + prompt[:5] + "这个词汇"))

    def test_program_families_are_disjoint_and_composition_matches_quotas(self):
        used = set()
        for split, families in PROGRAM_QUOTAS.items():
            self.assertTrue(used.isdisjoint(families))
            used.update(families)
            counts = {}
            for family, count in families.items():
                candidate = next(program_candidates(family))
                self.assertEqual(candidate.row["messages"], render_program(family, 0))
                kind = row_kind(candidate.row)
                counts[kind] = counts.get(kind, 0) + count
            self.assertEqual(counts, {key: val for key, val in QUOTAS[split].items() if key.startswith("program_")})

    def test_assistant_labels_and_program_regeneration(self):
        tokenizer = Tokenizer.from_file(str(Path("data/stage1_v1/tokenizer.json")))
        candidate = next(program_candidates("slot_revision")).row
        row = {**candidate, **encode_conversation(tokenizer, candidate["messages"])}
        validate_labels(row)
        ii = row["input_ids"].index(SPECIAL_IDS["<|user|>"]) + 1
        self.assertEqual(row["labels"][ii], IGNORE_LABEL)
        row["labels"][ii] = row["input_ids"][ii]
        with self.assertRaisesRegex(ValueError, "assistant-only"):
            validate_labels(row)

    def test_multi_turn_reallocation_keeps_total_budget_and_adds_distinct_family(self):
        for belle_multi in (1_000, 1_473, 2_000):
            quotas, families = comparison_quotas(belle_multi)
            train = quotas["train"]
            self.assertEqual(sum(train.values()), 20_000)
            self.assertEqual(train["belle_single"] + train["belle_multi"], 16_400)
            self.assertEqual(train["belle_multi"] + train["program_multi"], 3_000)
            self.assertEqual(sum(families["train"].values()), 3_600)
            self.assertLessEqual(max(families["train"].values()), 1_000)
        candidate = next(program_candidates("filter_then_select"))
        self.assertEqual(candidate.kind, "program_multi")
        self.assertEqual(len(candidate.row["messages"]), 4)

    def test_unverifiable_arithmetic_and_missing_context_are_rejected(self):
        messages = [{"role": "user", "content": "根据上面给出的文本提取关键字。"},
                    {"role": "assistant", "content": "主题和时间。"}]
        self.assertEqual(belle_quality(messages)[0], "missing_initial_context")
        messages[0]["content"] = "请计算32与19的总和是多少。"
        self.assertEqual(belle_quality(messages)[0], "unverified_constraint_or_external_action")

    def test_echo_and_unverifiable_budget_recommendation_are_rejected(self):
        messages = [{"role": "user", "content": "写一段关于印章设计的短文。"},
                    {"role": "assistant", "content": "写一段关于印章设计的短文。印章设计重视图案与留白。"}]
        messages[0]["content"] = "请你写一段关于印章设计元素的短文。"
        messages[1]["content"] = "请你写一段关于印章设计元素的短文。印章设计重视图案与留白。"
        self.assertEqual(belle_quality(messages)[0], "echoed_request")
        messages[0]["content"] = "根据我的实际需求，推荐一辆适合通勤的车。\n\n我上下班需要用，预算不超过15万元。"
        messages[1]["content"] = "我建议您选择微型车，其售价在您的预算范围内。"
        self.assertEqual(belle_quality(messages)[0], "unverified_current_price")

    def test_extraction_cannot_add_unsupported_sentences_or_claim_exhaustiveness(self):
        query = "从文章中提取有关书籍修复的信息。"
        source = "修复员用软刷清理书页上的浮尘，再把掉落的封页放在原书旁边。"
        answer = "修复员用软刷清理书页上的浮尘，再把掉落的封页放在原书旁边。"
        self.assertIsNone(grounded_answer_reason(query, answer, source))
        answer += "修复随后使用紫外线处理了手稿，还派专车送去外地再装订。"
        self.assertEqual(grounded_answer_reason(query, answer, source), "weakly_grounded_extraction")
        self.assertEqual(grounded_answer_reason("提取所有修复措施。", answer, source),
                         "unverified_exhaustive_extraction")

    def test_quoted_entities_and_template_group(self):
        left = narrow_groups([{"content": "请写一个关于《日光书店》的故事", "role": "user"}])
        right = narrow_groups([{"content": "请写一个关于《静候木屋》的故事", "role": "user"}])
        self.assertTrue(any(key.startswith("template:") for key in set(left) & set(right)))
        self.assertIn("entity:日光书店", left)

    def test_old_pt_chunks_ignore_short_sentence_and_cover_long_passage(self):
        paragraph = (
            "图书馆志愿者在闭馆之前复核了当天借出的全部书目，并把有明显破损的书籍单独放到一楼东侧的修复台。"
            "等记录核对完毕以后，值班室把当天的登记册交给第二天接班的同事。"
        )
        expected = "图书馆志愿者在闭馆之前复核了当天借出的全部书目并把有明显破损的书籍单独放到一楼东侧的修复台"
        self.assertIn(expected, tuple(pt_chunks(paragraph)))
        self.assertEqual(list(pt_chunks("请查看登记册。")), [])


if __name__ == "__main__":
    unittest.main()
