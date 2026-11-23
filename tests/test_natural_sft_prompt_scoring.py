import unittest

from scripts.freeze_natural_sft_prompts import TARGET, load_bank, score_machine


class NaturalSftPromptScoringTests(unittest.TestCase):
    def test_open_answer_has_no_machine_decision(self):
        self.assertIsNone(score_machine({"rubric": {"success": "方形的"}}, "方形的"))

    def test_single_line_uses_the_complete_output(self):
        record = {"machine_check": {"mode": "single_line", "value": "03,11"}}
        self.assertTrue(score_machine(record, "03,11\r\n"))
        for text in ("03,11，也可以选08", "其中是03,11", "03, 11", "3,11", "03,11\n\n"):
            with self.subTest(text=text):
                self.assertFalse(score_machine(record, text))

    def test_line_sequences_require_order_and_no_extras(self):
        record = {"machine_check": {"mode": "lines", "value": ["←还书处", "借书处→"]}}
        self.assertTrue(score_machine(record, "←还书处\r\n借书处→\r\n"))
        self.assertFalse(score_machine(record, "借书处→\n←还书处"))
        self.assertFalse(score_machine(record, "1. ←还书处\n2. 借书处→"))

    def test_json_checks_structure_types_duplicate_keys_and_full_input(self):
        record = {"machine_check": {"mode": "json", "value": {"青门": 3, "灰门": 0}}}
        self.assertTrue(score_machine(record, '{ "灰门": 0, "青门": 3 }'))
        for text in ('{"青门":3,"灰门":false}', '{"青门":3.0,"灰门":0}',
                     '{"青门":3,"灰门":0,"灰门":0}', '{"青门":3,"灰门":0,"别的":1}',
                     '```json\n{"青门":3,"灰门":0}\n```', '{"青门":3,"灰门":0}后面还有字'):
            with self.subTest(text=text):
                self.assertFalse(score_machine(record, text))

    def test_catalog_record_rejects_multiline_valid_json(self):
        records = load_bank(TARGET / "prompt_bank.tsv")
        catalog = next(row for row in records if row["scenario_family"] == "format_catalog_codes")
        self.assertTrue(score_machine(catalog, '["雨桥", "谷芽"]\r\n'))
        self.assertFalse(score_machine(catalog, '[\n  "雨桥",\n  "谷芽"\n]'))
        self.assertFalse(score_machine(catalog, '[\r  "雨桥", "谷芽"\r]'))
        control = next(row for row in records if row["scenario_family"] == "seating_grid_json")
        self.assertTrue(score_machine(control, '[\n  2,\n  5\n]'))


if __name__ == "__main__":
    unittest.main()
