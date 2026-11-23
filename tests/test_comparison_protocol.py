import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tinychat.chat_format import SPECIAL_IDS
from tinychat.training_state import file_identity, read_frozen_manifest
from scripts.blind_dialogue_review import prepare, aggregate
from scripts.summarize_comparison_pairs import paired_statistics


class ComparisonProtocolTests(unittest.TestCase):
    def test_paired_effect_is_order_independent_and_reverses_with_models(self):
        rows = []
        for i in range(100):
            for model in ("old_v2", "clean_v2"):
                correct = model == "clean_v2" and i < 50
                rows.append({"id": f"t{i}", "model": model, "category": f"c{i % 10}",
                             "score": 4 if correct else 0, "readable": correct, "severe_failure": not correct})
        forward = paired_statistics(list(reversed(rows)), replicates=20)["success"]
        backward = paired_statistics(rows, old="clean_v2", new="old_v2", replicates=20)["success"]
        self.assertEqual(forward["new_minus_old"], 0.5)
        self.assertEqual(backward["new_minus_old"], -0.5)
        self.assertEqual(forward["new_higher_cases"], 50)
        self.assertEqual(forward["mcnemar_exact_two_sided_p"], 2 / 2 ** 50)
        self.assertEqual(backward["mcnemar_exact_two_sided_p"], forward["mcnemar_exact_two_sided_p"])
        with self.assertRaisesRegex(ValueError, "uniquely paired"):
            paired_statistics(rows[:-1] + [rows[0]], replicates=20)

    def test_new_corpus_accepts_only_frozen_unchanged_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}", encoding="utf-8")
            manifest = {"schema_version": 1, "frozen": True, "corpus_id": "pretrain_clean_v2",
                        "vocab_size": 16384, "special_token_ids": SPECIAL_IDS,
                        "tokenizer_path": str(tokenizer.resolve()), "tokenizer_sha256": "0" * 64,
                        "files": {tokenizer.name: file_identity(tokenizer)}, "splits": {}}
            for split in ("train", "validation", "test"):
                binary = root / f"{split}.bin"
                binary.write_bytes(b"\x00\x00\x01\x00")
                rows = root / f"{split}.jsonl"
                rows.write_text('{}\n', encoding="utf-8")
                manifest["splits"][split] = {"documents": 1, "tokens": 2, "binary_bytes": 4}
                manifest["files"].update({p.name: file_identity(p) for p in (binary, rows)})
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(read_frozen_manifest(root, "pretrain", 16384)[0]["corpus_id"], "pretrain_clean_v2")
            manifest["corpus_id"] = "unapproved"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "supported frozen corpus"):
                read_frozen_manifest(root, "pretrain", 16384)
            manifest["corpus_id"] = "pretrain_clean_v2"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "train.bin").write_bytes(b"\x00" * 6)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                read_frozen_manifest(root, "pretrain", 16384)

    def test_custom_labels_survive_blinding_and_aggregation(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            prompts = [{"id": f"t{i}", "split": "test", "category": "arithmetic",
                        "messages": [{"role": "user", "content": str(i)}], "rubric": "correct"} for i in range(100)]
            payload = "".join(json.dumps(r) + "\n" for r in prompts)
            (root / "prompts.jsonl").write_text(payload, encoding="utf-8")
            digest = hashlib.sha256((root / "prompts.jsonl").read_bytes()).hexdigest()
            (root / "prompt_manifest.json").write_text(json.dumps({"prompts_sha256": digest}), encoding="utf-8")
            for label in ("old_v2", "clean_v2"):
                summary = {"prompts_sha256": digest, "decoding": {"strategy": "greedy"}, "tokenizer_sha256": "same", "batch_size": 1}
                (root / f"{label}_outputs.summary.json").write_text(json.dumps(summary), encoding="utf-8")
                outputs = [{"id": r["id"], "model_label": label, "text": label} for r in prompts]
                (root / f"{label}_outputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in outputs), encoding="utf-8")
            prepare(root, ("old_v2", "clean_v2"))
            folder = root / "blind_review"
            for number in range(1, 6):
                packet = json.loads((folder / f"input_{number:02d}.txt").read_text(encoding="utf-8"))
                scores = []
                for case in packet["cases"]:
                    row = {"id": case["id"]}
                    for side in ("A", "B"):
                        correct = case[side] == "clean_v2"
                        row[side] = {"score": 4 if correct else 0, "readable": correct, "relevant": correct,
                                     "main_instruction": correct, "factual_error": False, "multi_turn_correct": None,
                                     "severe_failure": not correct, "reason": "fixture"}
                    scores.append(row)
                (folder / f"scores_{number:02d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores), encoding="utf-8")
            aggregate(root)
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["clean_v2"]["all"]["success"], 100)
            self.assertEqual(summary["old_v2"]["all"]["success"], 0)


if __name__ == "__main__":
    unittest.main()
