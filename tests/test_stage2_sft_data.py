from pathlib import Path
from types import SimpleNamespace
import unittest

from opencc import OpenCC

from tinychat.chat_format import SPECIAL_IDS, encode_conversation
from tinychat.stage2_sft_data import (
    CMRC_FILE, PromptIndex, Source, answer_issue, assign_splits, clean_messages, deduplicate,
    edit_progresses, practical_without_reference, prompt_key, select_conversation, select_mixture,
    sentence_constraint_met, stratified_sample,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(character) + len(SPECIAL_IDS) for character in text])


def conversation(*pairs):
    return [{"role": role, "content": text}
            for pair in pairs for role, text in zip(("user", "assistant"), pair)]


class Stage2SftDataTest(unittest.TestCase):
    def setUp(self):
        self.convert = OpenCC("t2s").convert
        self.source = Source(Path("everyday.parquet"), "test/source", "0" * 40, "everyday.parquet",
                             "test", "", "https://example.test")
        self.eval = PromptIndex(0.75)
        self.tokenizer = CharacterTokenizer()

    def select(self, messages):
        return select_conversation({"messages": messages}, self.source, 5, self.tokenizer, self.convert, self.eval)

    def test_keeps_whole_linked_turns_without_eos(self):
        main = conversation(("我的陽台很小，該怎麼種小番茄？", "可以選矮生品種，用帶排水孔的大盆，放在日照充足的位置。"),
                            ("那個大盆需要多深？", "盆深可以選三十厘米左右，讓根系有足夠空間向下生長。"))
        too_long = conversation(("澆水方面又該怎麼做呢？", "先確認表土乾燥，再從盆沿慢慢澆水。" * 50))
        row, problem = self.select(main + too_long)
        self.assertIsNone(problem)
        self.assertEqual(row["messages"], clean_messages(main, self.convert))
        self.assertIn("阳台", row["messages"][0]["content"])
        expected = encode_conversation(self.tokenizer, row["messages"], final_eos=False)
        self.assertEqual(row["input_ids"], expected["input_ids"])
        self.assertEqual(row["labels"], expected["labels"])
        self.assertEqual(row["input_ids"][-1], SPECIAL_IDS["<|end_turn|>"])
        self.assertNotIn(SPECIAL_IDS["<|eos|>"], row["input_ids"])

    def test_rejects_high_stakes_and_fake_multiturn(self):
        unsafe = conversation(("孩子發燒，應該吃什麼藥？", "我認為可以先自行安排處方用藥，再去就診。"))
        self.assertEqual(self.select(unsafe)[1], "high_stakes")
        disconnected = conversation(("陽台很小，怎麼種小番茄？", "可以選矮生品種，再找採光充足的位置。"),
                                    ("能解釋光合作用嗎？", "植物可以利用光能合成有機物並釋放氧氣。"))
        row, reason = self.select(disconnected)
        self.assertIsNone(reason)
        self.assertEqual(row["messages"], clean_messages(disconnected[:2], self.convert))
        self.assertEqual(row["category"], "singleturn_natural")

    def test_rejects_evaluation_overlap_from_later_turn(self):
        target = "阳台种植番茄时应该如何选择花盆大小和放置位置？"
        self.eval.add(prompt_key(target), 0)
        messages = conversation(("陽台很小，怎麼種小番茄？", "可以選矮生品種，再找採光充足的位置。"),
                                ("陽台種植番茄時應該如何選擇花盆大小和放置位置呢？", "盆要寬而深，可以放在日照充足的位置。"))
        self.assertEqual(self.select(messages)[1], "evaluation_overlap")

    def test_rejects_known_synthetic_prompt_artifacts(self):
        answer = "先把几个关键词按主题归类，再用一句话写出中心意思。"
        self.assertEqual(answer_issue("ช่วยหน่อย请问怎样写中文短文？", answer), "non_chinese_or_short")
        self.assertEqual(answer_issue("您好，在这个角色下，我 将尽我所能提供写作帮助。", answer), "user_role_swap")
        self.assertEqual(answer_issue("感冒了，我是不是需要开始去健身房锻炼？", answer), "high_stakes")

    def test_checks_hard_sentence_bounds(self):
        two = "先把鞋子收好。再整理门口的雨伞。"
        self.assertFalse(sentence_constraint_met("总结不少于15句话。", two))
        self.assertTrue(sentence_constraint_met("请概括为二至三句话。", two))
        self.assertTrue(sentence_constraint_met("请总结到5句话以内。", two))
        self.assertFalse(sentence_constraint_met("请用一句话总结。", two))
        self.assertTrue(sentence_constraint_met("请用一句话总结。", "他问：“行吗？”"))

    def test_requires_real_edit_of_the_quoted_target(self):
        quote = "目前尚不能确定这位工程师是否需承担责任"
        before = f"调查仍在进行中，{quote}。公司禁止员工进入公司并查阅相关资料。"
        after = before.replace("进入", "进出")
        user = f"请修改‘工程’的措辞，调整“{quote}”这句。"
        self.assertFalse(edit_progresses(user, before, before))
        self.assertFalse(edit_progresses(user, before, after))
        self.assertTrue(edit_progresses(user, before, before.replace("这位工程师", "当事人")))
        self.assertTrue(edit_progresses("请补上感叹号。", "欢迎回家。", "欢迎回家！"))
        self.assertTrue(edit_progresses("请原样复述，不做改动。", before, before))

    def test_information_requires_self_contained_practical_ask(self):
        answer = "先按季节分好衣物，再把常穿的挂在开门就能拿到的位置。"
        self.assertTrue(practical_without_reference("小房間裡的衣物如何收納？".replace("收納", "收纳"), answer))
        self.assertFalse(practical_without_reference("现在最新上市的整理设备有哪些？", answer))
        self.assertFalse(practical_without_reference("汉弗莱阿格纽是谁？", answer))

    def test_cmrc_full_passage_and_human_offsets(self):
        cmrc = Source(Path(CMRC_FILE), "hfl/cmrc2018", "0" * 40, CMRC_FILE, "cc-by-sa-4.0", "", "https://example.test")
        context = "這座博物館 位於青島。展出品包括老地圖和城市照片。"
        first = {"id": "TRAIN_1_QUERY_1", "context": context, "question": "博物館在哪裡？",
                 "answers": {"text": ["青島"], "answer_start": [context.index("青島")]}}
        second = {**first, "id": "TRAIN_1_QUERY_2", "question": "館內展出哪一類照片？",
                  "answers": {"text": ["城市照片"], "answer_start": [context.index("城市照片")]}}
        a, problem = select_conversation(first, cmrc, 0, self.tokenizer, self.convert, self.eval)
        b, _ = select_conversation(second, cmrc, 1, self.tokenizer, self.convert, self.eval)
        self.assertIsNone(problem)
        self.assertIn(self.convert(context), a["messages"][0]["content"])
        self.assertEqual(a["messages"][1]["content"], "青岛")
        self.assertEqual(a["split_group"], b["split_group"])
        self.assertNotEqual(a["source_group"], b["source_group"])
        broken = {**first, "answers": {"text": ["青島"], "answer_start": [0]}}
        self.assertEqual(select_conversation(broken, cmrc, 0, self.tokenizer, self.convert, self.eval)[1], "annotation_offset")

    def test_cross_source_and_source_group_deduplication(self):
        a = "阳台种植番茄时应该如何选择花盆大小和放置位置"
        b = a + "呢"
        c = "如何避免多肉植物的根系在雨季腐烂"

        def row(identifier, source, group, question, turns=1):
            return {"id": identifier, "source": source, "source_group": group, "turns": turns,
                    "messages": conversation((question, "测试回答。")), "category": "question_explanation"}

        rows = [row("a", "x", "group-a", a), row("b", "y", "group-b", b, turns=2),
                row("c", "y", "group-b", c), row("d", "x", "group-d", "常用的资料夹格式有哪些")]
        result = deduplicate(rows)
        self.assertEqual({row["id"] for row in result}, {"b", "d"})
        self.assertEqual(next(row for row in result if row["id"] == "b")["cluster_id"], "a")
        self.assertEqual(stratified_sample(result, 2, 42), stratified_sample(result, 2, 42))

    def test_mixture_caps_summary_and_keeps_splits_disjoint(self):
        sources = {"summary": 24, "everyday": 12, "information-seeking": 15, "creative-writing": 12, "editing": 10}
        population = [{"id": f"{name}-{index}", "source": f"publisher/repo:{name}.parquet", "category": name,
                       "source_metadata": {"practical_writing": index % 2 == 0}}
                      for name, count in sources.items() for index in range(count)]
        population.extend({"id": f"cmrc-{index}", "source": f"hfl/cmrc2018:{CMRC_FILE}", "category": "grounded",
                           "split_group": f"cmrc-document-{index // 3}", "source_metadata": {}}
                          for index in range(24))
        result = select_mixture(population, 30)
        self.assertEqual(len(result), 30)
        self.assertLessEqual(sum(row["category"] == "summary" for row in result), 6)
        self.assertLessEqual(sum(row["category"] == "creative-writing" for row in result), 1)
        self.assertLessEqual(sum(row["category"] in {"summary", "grounded"} for row in result), 10)
        self.assertGreaterEqual(sum(row["category"] == "everyday" for row in result), 8)
        splits = assign_splits(result, {"train": 20, "validation": 5, "test": 5})
        self.assertEqual({key: len(rows) for key, rows in splits.items()}, {"train": 20, "validation": 5, "test": 5})
        self.assertEqual(len({row["id"] for rows in splits.values() for row in rows}), 30)
        group_splits = {}
        for split, rows in splits.items():
            for row in rows:
                if group := row.get("split_group"):
                    self.assertEqual(group_splits.setdefault(group, split), split)


if __name__ == "__main__":
    unittest.main()
