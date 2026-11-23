"""Focused Wikipedia preparation and publish-boundary tests."""

import json
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from tinychat.comparison_pretrain_data import CorpusBuild, Document
from tinychat.comparison_wiki_data import (
    add_old_holdouts, article_body, build_prompt_screen, ingest_wiki, pending_group_batches,
    publish, read_wiki_group, stratified_samples,
)
from tinychat.stage1_data import file_identity


ARTICLE = "\n".join((
    "滨海湿地处于陆地和海洋之间，受到潮汐和河流共同影响。这里既包括潮间带泥滩，也包括河口草甸以及盐沼。地形的轻微变化会影响水盐条件，改变植物群落的分布。",
    "生境特点",
    "有些候鸟在春秋迁徙途中经过这些地区，以弹涂鱼、甲壳动物和其他小生物为食。研究者通常利用样线调查记录鸟类种类与活动时间，再将记录与不同季节的水位变化比较。",
    "在受人为干扰的区域，恢复项目有时先疏通潮沟，并减少机械进入的频率。不同措施的成效需要持续观测，例如比较恢复前后的沉积物厚度、土壤含盐量以及植物盖度。",
    "有些监测项目使用固定观测点，分别在大潮、小潮和风暴过后检查潮沟是否保持连通；长期记录有助于区分年际变化和工程影响。",
))


def row(identifier, text, source="wiki/fixture.parquet"):
    return {"id": identifier, "title": "滨海湿地", "source": source, "text": text,
            "row_group": 0, "row_in_group": 0, "url": "https://zh.wikipedia.org/wiki/example"}


class WikiCorpusTests(unittest.TestCase):
    def test_article_strips_reference_tail_and_lists_but_keeps_prose(self):
        raw = ARTICLE + "\n* 第一条列表之外的内容。\n外部連結\n" + ARTICLE.replace("湿地", "山地")
        reason, body, removed = article_body(raw, "滨海湿地")
        self.assertIsNone(reason)
        self.assertEqual(body, ARTICLE)
        self.assertGreater(removed, 0)
        self.assertEqual(article_body(ARTICLE, "湖泊列表")[0], "list_or_disambiguation")
        self.assertEqual(article_body("这是一条短定义。", "湖泊")[0], "stub_or_nonprose")
        self.assertEqual(article_body(ARTICLE + "锟斤拷", "湖泊")[0], "garbled_or_reserved")

    def test_screen_checks_short_whole_matches_and_six_character_substrings(self):
        with TemporaryDirectory() as tmp:
            prompts = Path(tmp) / "prompts.jsonl"
            prompts.write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": text}]}, ensure_ascii=False)
                                         for text in ("你好", "甲乙丙丁戊己", "123456")), encoding="utf-8")
            screen, identities = build_prompt_screen([prompts])
        self.assertEqual(identities[0]["substring_min_chars"], 6)
        self.assertEqual(screen(" 你，好。"), "evaluation_exact_overlap")
        self.assertIsNone(screen("你好，今天的天气不错。"))
        self.assertEqual(screen("插入甲乙丙丁戊己的句子。"), "evaluation_substring_overlap")
        self.assertEqual(screen("号码１２３４５６可见。"), "evaluation_substring_overlap")

    def test_cleaned_old_view_remains_in_component_and_rejects_wiki_copy(self):
        original = ARTICLE + "\n参考文献\n" + "这段文献标题与正文无关，包含额外文字和署名。" * 25
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "validation.jsonl").write_text(json.dumps(row("legacy_validation", original), ensure_ascii=False), encoding="utf-8")
            (root / "test.jsonl").write_text("", encoding="utf-8")
            build = CorpusBuild()
            add_old_holdouts(build, root, lambda _: None)
            ingest_wiki(build, [row("training_candidate", ARTICLE)], lambda _: None)
            kept, discarded, _ = build.index.select()
        self.assertFalse(any(kept.values()))
        self.assertEqual(discarded["wiki/fixture.parquet"]["old_holdout_duplicate_component"], 1)
        self.assertEqual(build.index.counts["old_holdout_exact"], 1)

    def test_samples_and_publish_require_expected_inputs(self):
        docs = [Document(row(str(index), "清洗正文。" * (index + 1), f"wiki/s{index % 2}")) for index in range(48)]
        samples = stratified_samples({"train": docs})
        self.assertEqual(len(samples), len({sample["id"] for sample in samples}))
        self.assertEqual(len(samples), 32)
        self.assertEqual({sample["length_band"] for sample in samples}, {"short", "middle", "long"})
        self.assertEqual({sample["source"] for sample in samples}, {"wiki/s0", "wiki/s1"})
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            output = root / "data"
            output.mkdir()
            marker = root / "approved.txt"
            with self.assertRaisesRegex(FileNotFoundError, "independent quality review marker"):
                publish(output, marker)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("fixture", encoding="utf-8")
            report = root / "report.json"
            report.write_text("{}", encoding="utf-8")
            part = output / "train.bin.partial"
            part.write_bytes(b"\x01\x00")
            manifest = {"tokenizer_path": str(tokenizer), "quality_report_path": str(report),
                        "files": {"tokenizer.json": file_identity(tokenizer), "train.bin": file_identity(part)}}
            (output / "prepared.json").write_text(json.dumps({"proposed_manifest": manifest, "parts": ["train.bin"]}), encoding="utf-8")
            marker.write_text("PASS", encoding="utf-8")
            publish(output, marker)
            self.assertEqual((output / "train.bin").read_bytes(), b"\x01\x00")
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["status"], "frozen")

    def test_parallel_parquet_reads_keep_interleaved_order_and_skip_committed(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.parquet"
            pq.write_table(pa.table({"id": ["a", "b"], "url": ["url/a", "url/b"],
                                    "title": ["标题甲", "标题乙"], "text": ["正文甲", "正文乙"]}),
                           path, row_group_size=1)
            committed = ({"path": "a.parquet", "row_group": 1, "documents": 1},)
            with closing(pq.ParquetFile(path)) as first, closing(pq.ParquetFile(path)) as second, \
                    patch("tinychat.comparison_wiki_data.read_wiki_group", wraps=read_wiki_group) as read_group:
                files = [("a.parquet", first, iter((1, 0))), ("b.parquet", second, iter((0, 1)))]
                loaded = list(pending_group_batches(files, committed, lambda _: None))
        self.assertEqual([(ordinal, group["path"], group["row_group"]) for ordinal, group, _ in loaded],
                         [(2, "b.parquet", 0), (3, "a.parquet", 0), (4, "b.parquet", 1)])
        self.assertEqual([batch[0]["article_id"] for _, _, batch in loaded], ["a", "a", "b"])
        self.assertEqual(read_group.call_count, 3)


if __name__ == "__main__":
    unittest.main()
