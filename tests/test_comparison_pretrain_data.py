import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tinychat import comparison_pretrain_data as data


def article(seed, length=260):
    randomizer = random.Random(seed)
    return "".join(chr(randomizer.randrange(0x4E00, 0x5400)) for _ in range(length)) + "。"


def document(record_id, text, source="s1", quality=1, old_holdout=False):
    return data.Document({"id": record_id, "text": text, "source": source}, quality, old_holdout)


class ComparisonPretrainingDataTest(unittest.TestCase):
    def test_filter_removes_navigation_and_rejects_mechanical_content(self):
        prose = article(1)
        reason, cleaned, quality, removed = data.quality_filter(f"首页\n{prose}\n免责声明：转载来源。")
        self.assertIsNone(reason)
        self.assertEqual(cleaned, prose)
        self.assertEqual(removed, len("首页免责声明：转载来源。"))
        self.assertLess(quality, 1)

        cases = {
            "insufficient_chinese": "标题。很短的演示文章。",
            "repeated_lines": "\n".join([prose] * 6),
            "repeated_characters": prose + "同" * 40,
            "malformed_text": prose + "锟斤拷锟斤拷",
            "keyword_stuffing": prose + "\n关键词：词一\n关键词：词二\n关键词：词三",
            "unsupported_software_fields": prose + "\n功能：虚词\n使用说明：残句\n软件介绍：情节\n软件名称：引语",
            "repeated_reference_replies": "参考回复：\n" + prose + "\n其它回复\n" + prose[3:] + "独立内容。",
            "malformed_question_sequences": prose + "问题?吗?随后?如何?继续。",
            "unrelated_news_blurb": "某地区职业篮球赛揭晓半决赛时间\n二〇二六年十一月\n来源：报社\n"
                                     + "\n“".join([prose[i:i + 21] for i in range(0, 210, 21)]),
            "deposit_return_ad": "每位投资者可获得服务。每位投资者每日可获得固定收益三元。"
                                 "投资者在注册当天充值。" + prose,
        }
        for expected, raw in cases.items():
            with self.subTest(reason=expected):
                self.assertEqual(data.quality_filter(raw)[0], expected)
        news = "警方通报称有人以每日固定收益为名要求注册充值，相关报道提醒投资者查验资质。" + prose
        self.assertIsNone(data.quality_filter(news)[0])

    def test_second_pass_detects_structural_and_promotional_noise(self):
        prose = article(50)
        cases = {
            "seo_synonym_spinning": prose + "并异国相关记载。宠喜欢者在此。首点虽矮可以改变。",
            "malformed_foreign_names": prose + "AB？XZ和CD？EF。还???没有结束。",
            "gambling_promotion": prose + "推广博彩和在线投注，注册领取返利。",
            "locked_exam_fragment": prose + "试题查看价格为十个金币，请先登录。",
            "serious_illness_product_promotion": prose + "神奇消费群体和使用者宣称本品延长癌症存活期。",
            "adult_media_seo": prose + "成人影片免费在线观看。",
            "sports_lottery_splice": "球队甲vs球队乙\n" + prose + "热代码十二、冷代码三四与预报无关。",
        }
        for expected, raw in cases.items():
            with self.subTest(reason=expected):
                self.assertEqual(data.quality_filter(raw)[0], expected)
        valid_contexts = (
            "警方调查网络赌场，案情涉及在线投注；涉案网页要求填写资料注册。",
            "癌症论文使用置信区间对照使用者与未使用者的有效率，不能支持治愈结论。",
            "消费群体和使用者讨论所谓神奇产品，临床研究不能支持治愈癌症。",
        )
        for context in valid_contexts:
            with self.subTest(context=context):
                self.assertIsNone(data.quality_filter(prose + context)[0])
        press = f"{prose[:12]}\n2023-11-08 11:00:00来源：通讯稿\ntext_black_16_YaHei\">{prose}"
        reason, cleaned, _, removed = data.quality_filter(press)
        self.assertIsNone(reason)
        self.assertEqual(cleaned, f"{prose[:12]}\n{prose}")
        self.assertGreater(removed, 30)

    def test_topic_screen_requires_distinct_unrelated_article_restarts(self):
        title = "森林资源与物种监测方法详细说明"
        lead = title + article(1, length=710)
        stray_story = "近日" + article(2, length=150)
        second_story = "日前" + article(3, length=150)
        normal = "\n".join((title, lead, "章节索引", stray_story, "结语"))
        spliced = f"{normal}\n{second_story}"
        self.assertIsNone(data.quality_filter(normal)[0])
        self.assertEqual(data.quality_filter(spliced)[0], "unrelated_spliced_sections")
        self.assertIsNone(data.quality_filter("各部门消息汇总" + spliced)[0])

    def test_exact_and_verified_near_duplicates_keep_one_component(self):
        base = article(2)
        near = base[:80] + "新词段" + base[80:]
        exact = "  " + base.replace("。", "！")
        index = data.DuplicateIndex()
        index.add(document("format", exact, quality=0.8))
        index.add(document("near", near, quality=0.9))
        index.add(document("original", base))
        index.add(document("different", article(3)))

        selected, discarded, groups = index.select()
        self.assertEqual({doc.id for docs in selected.values() for doc in docs}, {"original", "different"})
        self.assertEqual(index.counts["exact"], 1)
        self.assertEqual(index.counts["near"], 1)
        self.assertGreaterEqual(index.examples["near"][0]["jaccard"], 0.8)
        self.assertEqual(discarded["s1"]["duplicate_component_nonrepresentative"], 2)
        self.assertEqual(groups[0]["representative"], "original")

    def test_legacy_holdout_and_eval_conflicts_exclude_connected_groups(self):
        first, second, third = article(1), article(2), article(3)
        index = data.DuplicateIndex()
        index.add(document("old-holdout", first, "heldout", old_holdout=True))
        index.add(document("near-holdout", first[:40] + "重要" + first[40:]))
        index.add(document("raw", second))
        evaluation = document("eval-blocked", second[:90] + "必查" + second[90:])
        evaluation.evaluation_conflict = "evaluation_substring_overlap"
        index.add(evaluation)
        index.add(document("clean", third))

        selected, discarded, _ = index.select()
        self.assertEqual({doc.id for docs in selected.values() for doc in docs}, {"clean"})
        self.assertEqual(index.counts["old_holdout_near"], 1)
        self.assertEqual(discarded["s1"]["old_holdout_duplicate_component"], 1)
        self.assertEqual(discarded["s1"]["evaluation_conflict_component"], 2)

    def test_old_holdout_navigation_is_stripped_for_duplicate_screening(self):
        body = article(11)
        old_holdout = body + "\n版权声明：" + article(12, length=74)
        raw_grams = data.shingles(data.normalized_content(old_holdout))
        body_grams = data.shingles(data.normalized_content(body))
        self.assertLess(len(raw_grams & body_grams) / len(raw_grams | body_grams), 0.8)
        self.assertEqual(data.quality_filter(old_holdout)[1], body)

        build = data.CorpusBuild()
        build.add(document("heldout", old_holdout, "old").record, old_holdout=True)
        build.add(document("new", body).record)

        selected, discarded, _ = build.index.select()
        self.assertFalse(any(selected.values()))
        self.assertEqual(build.index.documents[0].text, old_holdout)
        self.assertEqual(build.index.counts["old_holdout_exact"], 1)
        self.assertEqual(build.counts["old"]["old_holdout_navigation_stripped_views"], 1)
        self.assertEqual(discarded["s1"]["old_holdout_duplicate_component"], 1)

    def test_short_complete_eval_prompt_is_screened_inside_a_document(self):
        keys = {"鲸鱼用什么呼吸", "继续", "1234567"}
        matches = data.comparison_eval_matches(keys)
        self.assertEqual(matches("确切问题鲸鱼用什么呼吸出现在正文中"), "evaluation_short_substring_overlap")
        self.assertEqual(matches("继续"), "evaluation_exact_overlap")
        self.assertIsNone(matches("这里继续介绍编码1234567还有其他正文内容"))

    def test_global_order_is_stable_and_prefix_counts_partial_documents(self):
        docs = [document(f"doc-{number}", article(number), f"source-{number % 2}") for number in range(6)]
        for doc in docs:
            doc.token_bytes = b"\x02\x00" * 11
        ordered = data.ordered_documents(docs)
        expected_ids = [doc.id for doc in ordered]
        self.assertEqual(expected_ids, [doc.id for doc in data.ordered_documents(reversed(docs))])

        with TemporaryDirectory() as temp, patch.multiple(data, PREFIX_TOKENS=50, COVERAGE_WINDOW=10):
            out = Path(temp)
            stats, _ = data.write_splits(out, {"train": docs, "validation": [], "test": []}, lambda message: None)
            self.assertEqual((out / "train.bin.partial").stat().st_size, 132)
            self.assertEqual(stats["train"]["tokens"], 66)
            prefix = stats["train"]["first_100m_source_tokens"]
            expected_prefix = {source: 11 * sum(doc.source == source for doc in ordered[:4]) for source in prefix}
            expected_prefix[ordered[4].source] += 6
            self.assertEqual(prefix, expected_prefix)
            targets = expected_prefix.copy()
            targets[ordered[0].source] -= 1
            targets[ordered[4].source] += 1
            self.assertEqual(stats["train"]["first_100m_target_source_tokens"], targets)
            self.assertEqual([sum(window.values()) for window in stats["train"]["first_100m_10m_window_source_tokens"]], [10] * 5)
            self.assertEqual([json.loads(line)["id"] for line in
                              (out / "train.jsonl.partial").read_text(encoding="utf-8").splitlines()], expected_ids)


if __name__ == "__main__":
    unittest.main()
