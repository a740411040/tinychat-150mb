from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tinychat import comparison_verified_sft as generator
from tinychat import verified_sft_checks as checker
from tinychat.chat_format import IGNORE_LABEL, SPECIAL_IDS, encode_conversation
from tinychat.stage1_data import file_identity


TEST_SEED = 771944


def first_multi(family):
    for ordinal in range(100):
        row = generator.render_row(family, "train", 0, True, ordinal, seed=TEST_SEED)
        if row is not None:
            return row
    raise AssertionError(f"No multi-turn test row for {family}")


def text_pair(user, answer):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def task_prompt(family, payload):
    return "\n".join((checker.STARTS["train"][0], checker.RULES[family], checker.RESOURCE_LINE,
                      *generator.fields(payload), checker.ENDS["train"][0]))


def follow_up(family, patch):
    lead = checker.UPDATES["train"][0]
    if family == "constraints":
        lead += "原有的限制仍有效。"
    return "\n".join((lead, checker.PATCH_LINE, *generator.fields(patch)))


def prepared_fixture(directory):
    root = Path(directory)
    output, base = root / "prepared", root / "base"
    output.mkdir()
    base.mkdir()
    tokenizer = base / "tokenizer.json"
    tokenizer.write_bytes(b"{}")
    files = {"tokenizer.json": file_identity(tokenizer)}
    splits = {}
    for split, count in generator.RESULT_COUNTS.items():
        name = f"{split}.jsonl"
        path = output / name
        path.write_bytes(b"{}\n")
        files[name] = file_identity(path)
        splits[split] = {"documents": count, "tokens": count * 2, "assistant_tokens": count}
    manifest = {
        "schema_version": 1, "corpus_id": generator.CORPUS_ID, "frozen": True,
        "vocab_size": 16_384, "max_tokens": 512, "tokenizer_path": str(tokenizer),
        "tokenizer_sha256": sha256(tokenizer.read_bytes()).hexdigest(),
        "special_token_ids": SPECIAL_IDS, "counts": generator.RESULT_COUNTS, "splits": splits, "files": files,
    }
    content = (json.dumps(manifest, indent=4) + "\n").encode("utf-8")
    (output / "manifest.prepared.json").write_bytes(content)
    marker = root / "verified_sft_review_ready.txt"
    approval = {"decision": "pass", "corpus_id": generator.CORPUS_ID, "prepared_sha256": sha256(content).hexdigest()}
    return output, base, marker, approval, content


class VerifiedSFTTests(unittest.TestCase):
    def test_every_family_rejects_forged_answers_at_each_assistant_turn(self):
        for family in checker.RULES:
            with self.subTest(family=family):
                row = first_multi(family)
                messages = row["messages"]
                self.assertEqual(checker.verify_dialogue(messages, "train", family, 0), (family, 0))
                for changed, other in ((1, 3), (3, 1)):
                    tampered = deepcopy(messages)
                    tampered[changed]["content"] = messages[other]["content"]
                    with self.assertRaises(ValueError):
                        checker.verify_dialogue(tampered, "train", family, 0, row["prompt_identity"])

    def test_every_family_recomputes_answer_after_visible_input_changes(self):
        for family in checker.RULES:
            with self.subTest(family=family):
                original = generator.render_row(family, "train", 0, False, 0, seed=TEST_SEED)
                for ordinal in range(1, 100):
                    alternate = generator.render_row(family, "train", 0, False, ordinal, seed=TEST_SEED)
                    if original["messages"][1] != alternate["messages"][1]:
                        break
                else:
                    self.fail(f"Did not find a different {family} answer")
                checker.verify_dialogue(alternate["messages"], "train", family, 0)
                tampered = [alternate["messages"][0], original["messages"][1]]
                with self.assertRaises(ValueError):
                    checker.verify_dialogue(tampered, "train", family, 0)
                with self.assertRaisesRegex(ValueError, "Visible prompt identity"):
                    checker.verify_dialogue(alternate["messages"], "train", family, 0, original["prompt_identity"])

    def test_corrections_are_replayed_in_turn_order(self):
        question = task_prompt("correction", {"记录": {"负责人": "柳宁", "地点": "东楼"},
                                             "已更正": {"负责人": "夏珂"}, "字段": "负责人"})
        a = follow_up("correction", {"改写": {"负责人": "陆文"}})
        b = follow_up("correction", {"改写": {"负责人": "顾泉"}})
        messages = text_pair(question, "夏珂") + text_pair(a, "陆文") + text_pair(b, "顾泉")
        self.assertEqual(checker.verify_dialogue(messages, "train"), ("correction", 0))
        messages[2], messages[4] = messages[4], messages[2]
        with self.assertRaisesRegex(ValueError, "assistant turn 2"):
            checker.verify_dialogue(messages, "train")

    def test_appended_constraints_keep_earlier_stronger_bounds(self):
        rows = [["甲", "北", 10, 1], ["乙", "北", 90, 16], ["丙", "北", 90, 4]]
        question = task_prompt("constraints", {"表": rows, "组": "北", "最低分": 0, "最多件数": 20})
        a = follow_up("constraints", {"最低分": 80})
        b = follow_up("constraints", {"最低分": 5, "最多件数": 6})
        messages = text_pair(question, '["甲","乙","丙"]') + text_pair(a, '["乙","丙"]') + text_pair(b, '["丙"]')
        checker.verify_dialogue(messages, "train")
        messages[-1]["content"] = '["甲","丙"]'
        with self.assertRaisesRegex(ValueError, "assistant turn 3"):
            checker.verify_dialogue(messages, "train")

    def test_full_encoding_rejects_tampered_user_and_end_marker_labels(self):
        tokenizer, _ = generator.load_frozen_tokenizer(Path("data/stage1_v1"))
        row = first_multi("arithmetic")
        row.update(encode_conversation(tokenizer, row["messages"]))
        checker.verify_encoded_row(row, tokenizer, 512)
        altered = deepcopy(row)
        user_start = altered["input_ids"].index(SPECIAL_IDS["<|user|>"]) + 1
        altered["labels"][user_start] = altered["input_ids"][user_start]
        with self.assertRaises(ValueError):
            checker.verify_encoded_row(altered, tokenizer, 512)
        row["labels"][-1] = IGNORE_LABEL
        with self.assertRaises(ValueError):
            checker.verify_encoded_row(row, tokenizer, 512)

    def test_template_and_entity_partition_quotas(self):
        inventory = generator.track_entity_partitions()
        self.assertEqual(len(set.union(*(set(names) for names in inventory.values()))), sum(map(len, inventory.values())))
        for split, expected in generator.RESULT_COUNTS.items():
            templates, dialogues = Counter(), Counter()
            multi = 0
            for index, family in enumerate(checker.RULES):
                for style, single, multiple in generator.plan_slots(split, index, probe=False):
                    templates[(family, style)] += single + multiple
                    dialogues[family] += single + multiple
                    multi += multiple
            self.assertEqual(len(templates), 400)
            self.assertLessEqual(max(templates.values()), 50)
            self.assertEqual(set(dialogues.values()), {expected // 20})
            self.assertEqual(sum(dialogues.values()), expected)
            self.assertEqual(multi, generator.MULTI_COUNTS[split])

    def test_publication_requires_review_bound_to_unmodified_prepared_data(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            output, base, marker, approval, _ = prepared_fixture(directory)
            published = output / "manifest.json"
            with self.assertRaises(FileNotFoundError):
                generator.publish_prepared(output, base, marker)
            marker.write_text(json.dumps(approval | {"prepared_sha256": "0" * 64}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not approve this prepared"):
                generator.publish_prepared(output, base, marker)
            marker.write_text(json.dumps(approval), encoding="utf-8")
            (output / "test.jsonl").write_bytes(b"{}\n{}\n")
            with self.assertRaisesRegex(ValueError, "Prepared file changed"):
                generator.publish_prepared(output, base, marker)
            self.assertFalse(published.exists())

    def test_approved_publication_preserves_manifest_bytes_and_refuses_overwrite(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            output, base, marker, approval, content = prepared_fixture(directory)
            marker.write_text(json.dumps(approval), encoding="utf-8")
            manifest = generator.publish_prepared(output, base, marker)
            self.assertEqual(manifest["counts"], generator.RESULT_COUNTS)
            self.assertEqual((output / "manifest.json").read_bytes(), content)
            self.assertEqual((output / "manifest.prepared.json").read_bytes(), content)
            with self.assertRaisesRegex(RuntimeError, "frozen manifest already exists"):
                generator.publish_prepared(output, base, marker)


if __name__ == "__main__":
    unittest.main()
