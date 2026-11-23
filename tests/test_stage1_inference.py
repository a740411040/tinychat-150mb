import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer, models

from tinychat.chat_format import SPECIAL_IDS
from tinychat.inference import check_checkpoint, checked_tokenizer


class FrozenInferenceTests(unittest.TestCase):
    def test_identity_and_completed_stage_are_required(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            first = Tokenizer(models.WordLevel({**SPECIAL_IDS, "甲": 7, "乙": 8}))
            second = Tokenizer(models.WordLevel({**SPECIAL_IDS, "乙": 7, "甲": 8}))
            path, other = root / "tokenizer.json", root / "other.json"
            first.save(str(path))
            second.save(str(other))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            identity = {"tokenizer_sha256": digest, "files": {"tokenizer.json": {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}}}
            state = {"stage": "pretrain", "config": {"vocab_size": 9}, "tokenizer_sha256": digest, "data_identity": identity, "progress": {"tokens": 100_000_000, "updates": 3052}, "settings": {"planned_tokens": 100_000_000, "planned_updates": 3052}, "validations": {name: {"step": 3052, "nonfinite": False} for name in ("final", "final_comparable")}}
            self.assertEqual(check_checkpoint(state, path, completed_stage="pretrain")[1], digest)
            with self.assertRaisesRegex(ValueError, "identity"):
                check_checkpoint(state, other)
            state["progress"]["tokens"] = state["settings"]["planned_tokens"] = 10
            with self.assertRaisesRegex(ValueError, "100M"):
                check_checkpoint(state, path, completed_stage="pretrain")
            state["stage"] = "pilot"
            with self.assertRaisesRegex(ValueError, "provisional"):
                check_checkpoint(state, path)
            # A self-consistent hash still cannot legitimize changed role IDs.
            altered = Tokenizer(models.WordLevel({**SPECIAL_IDS, "<|user|>": 5, "<|assistant|>": 4, "甲": 7, "乙": 8}))
            altered.save(str(other))
            with self.assertRaisesRegex(ValueError, "special token IDs"):
                checked_tokenizer(other, hashlib.sha256(other.read_bytes()).hexdigest(), 9)


if __name__ == "__main__":
    unittest.main()
