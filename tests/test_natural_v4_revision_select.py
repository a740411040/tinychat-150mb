import unittest
from unittest.mock import patch

import scripts.natural_v4_revision_select as selector
from scripts.natural_v4_revision_select import balanced_required, requested_rewrite_stratum, balanced_multi
from scripts.natural_v4_staged_queue import validate_round2_writer_selection


class RevisionSelectionTests(unittest.TestCase):
    def rows(self):
        return [{"id": str(i), "sources": [{"repo": "A" if i < 5 else "B"}],
                 "assistant_content_token_budget": 300, "family": "rewrite" if i == 7 else "simplify" if i == 6 else "other"}
                for i in range(8)]

    def test_required_families_survive_source_balancing_without_record_changes(self):
        rows = self.rows()
        required = [(kind, lambda row, kind=kind: row["family"] == kind) for kind in ("rewrite", "simplify")]
        selected = balanced_required(rows, 4, required)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({row["id"] for row in selected}), 4)
        self.assertTrue({"rewrite", "simplify"}.issubset({row["family"] for row in selected}))
        self.assertTrue(all(any(row is original for original in rows) for row in selected))
        self.assertEqual(selected, balanced_required(rows, 4, required))

    def test_absent_family_is_not_filled_with_an_unrelated_row(self):
        with self.assertRaisesRegex(ValueError, "Missing prespecified"):
            balanced_required(self.rows(), 4, [("missing", lambda row: row["family"] == "missing")])

    def test_simplification_is_a_request_with_style_evidence_not_an_exclusive_enum(self):
        row = {"messages": [{"role": "user", "content": "编辑下面这句话，使它更流畅和易于理解：设备负责双向翻译。"}],
               "automated_checker_response": {"turn_audits": [{"task_kind": "rewrite", "style_verdict": "satisfied", "coverage": [{"verdict": "covered"}]}]}}
        self.assertTrue(requested_rewrite_stratum(row, simplify=True))
        self.assertFalse(requested_rewrite_stratum(row, simplify=False))
        audit = row["automated_checker_response"]["turn_audits"][0]
        for bad_style in ("failed", "uncertain", "not_applicable"):
            audit["style_verdict"] = bad_style
            self.assertFalse(requested_rewrite_stratum(row, simplify=True))
        audit["style_verdict"] = "satisfied"
        for unrelated in ("总结下面这段文本的摘要：\n通俗地说，这是一项技术。", "编写一个简单的日记条目，并将其转换为一篇文学散文。"):
            row["messages"][0]["content"] = unrelated
            self.assertFalse(requested_rewrite_stratum(row, simplify=True))
        row["messages"][0]["content"] = "重写下面的段落：\n这是一段需保留完整意思的正文。"
        audit["style_verdict"] = "not_applicable"
        self.assertTrue(requested_rewrite_stratum(row, simplify=False))
        audit["task_kind"] = "summary"
        self.assertFalse(requested_rewrite_stratum(row, simplify=False))

    def test_document_dialogues_cannot_fill_the_daily_quota(self):
        rows = [{"id": str(i), "multi_subtype": "document" if i > 1 else "daily",
                 "sources": [{"repo": "A" if i % 2 else "B"}], "assistant_content_token_budget": 300,
                 "automated_checker_response": {"multi_subtype": "document" if i > 1 else "daily"}}
                for i in range(8)]
        selected = balanced_multi(rows, 4, checked=True)
        self.assertEqual([row["multi_subtype"] for row in selected].count("daily"), 2)
        self.assertEqual([row["multi_subtype"] for row in selected].count("document"), 2)
        self.assertTrue(all(any(row is original for original in rows) for row in selected))
        with self.assertRaisesRegex(ValueError, "daily"):
            balanced_multi(rows[1:], 8)

    def test_checked_multi_sampling_requires_subtype_agreement(self):
        rows = [{"id": str(i), "multi_subtype": "daily" if i < 2 else "document",
                 "sources": [{"repo": "A"}], "assistant_content_token_budget": 300,
                 "automated_checker_response": {"multi_subtype": "document"}}
                for i in range(4)]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            balanced_multi(rows, 4, checked=True)

    def round2_writer_results(self, challenges):
        def row(category, index, repo="BelleGroup/train_0.5M_CN", subtype=None, tags=()):
            identifier = f"{category}-{index}"
            return {"status": "accepted", "next_seed": {
                "id": identifier, "category": category, "source_group": identifier,
                "multi_subtype": subtype, "validation_challenges": list(tags),
                "sources": [{"repo": repo, "input": repo}], "assistant_content_token_budget": 480,
            }}

        return (
            [row("natural", index) for index in range(4)]
            + [row("grounded", 0, repo="hfl/cmrc2018")]
            + [row("grounded", index) for index in range(1, 4)]
            + [row("multi", index, subtype="daily" if index < 2 else "document") for index in range(4)]
            + [row("instruction", index, tags=challenges.get(index, ())) for index in range(9)]
        )

    def test_round2_writer_reserves_distinct_challenges_at_capacity(self):
        for single_tag in ("full_rewrite", "simplify"):
            with self.subTest(single_tag=single_tag):
                outcomes = self.round2_writer_results({
                    2: (single_tag,), 3: ("full_rewrite", "simplify"),
                })
                outcomes.extend({"id": f"rejected-{i}", "status": "rejected"} for i in range(27))
                with (
                    patch.object(selector, "result_rows", return_value=outcomes),
                    patch.object(selector, "phase_status", return_value={
                        "submitted": 64, "completed": 48, "failed_no_result": 16,
                    }),
                    patch.object(selector, "save_jsonl") as save,
                    patch.object(selector, "write_new") as report,
                ):
                    result = selector.select.__wrapped__("round2_writer")
                selected = save.call_args.args[1]
                self.assertEqual(result["selected_categories"]["instruction"], 8)
                validate_round2_writer_selection(selected)
                self.assertTrue({"instruction-2", "instruction-3"} <= {row["id"] for row in selected})
                recorded = report.call_args.args[1]
                self.assertEqual(tuple(recorded[key] for key in (
                    "parent_submitted", "parent_completed", "parent_failed_no_result", "parent_accepted",
                )), (64, 48, 16, 21))

    def test_round2_writer_reports_insufficient_when_only_one_row_covers_both(self):
        outcomes = self.round2_writer_results({3: ("full_rewrite", "simplify")})
        with (
            patch.object(selector, "result_rows", return_value=outcomes),
            patch.object(selector, "phase_status", return_value={
                "submitted": 21, "completed": 21, "failed_no_result": 0,
            }),
            patch.object(selector, "save_jsonl") as save,
            patch.object(selector, "write_new") as report,
        ):
            with self.assertRaisesRegex(ValueError, "distinct full-rewrite and simplify"):
                selector.select.__wrapped__("round2_writer")
        save.assert_not_called()
        self.assertEqual(report.call_args.args[1]["status"], "INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
