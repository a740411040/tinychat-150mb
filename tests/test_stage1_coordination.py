import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.blind_dialogue_review import import_result
from scripts.monitor_stage1 import main as monitor_main


class CoordinationBoundaryTests(unittest.TestCase):
    def test_score_cannot_override_blind_identity(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            folder = root / "blind_review"
            folder.mkdir()
            (folder / "input_01.txt").write_text(json.dumps({"cases": [{"id": "test_001", "category": "factual_basic"}]}), encoding="utf-8")
            score = {"score": 3, "readable": True, "relevant": True, "main_instruction": True, "factual_error": False, "multi_turn_correct": None, "severe_failure": False, "reason": "回答正确"}
            row = {"id": "test_001", "A": {**score, "model": "sft"}, "B": score}
            result = {"taskCompleted": True, "executionPass": True, "providerErrors": [], "taskId": "unit-test-only", "finalText": json.dumps(row)}
            path = root / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate score fields"):
                import_result(root, path, 1)
            self.assertFalse((folder / "scores_01.jsonl").exists())
            row["A"] = score
            result["finalText"] = json.dumps(row)
            result["taskCompleted"] = False
            path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not complete"):
                import_result(root, path, 1)
            result["taskCompleted"] = True
            path.write_text(json.dumps(result), encoding="utf-8")
            import_result(root, path, 1)
            self.assertEqual(json.loads((folder / "scores_01.jsonl").read_text(encoding="utf-8")), row)

    def test_monitor_survives_timeout_and_empty_query(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            path = Path(tmp) / "telemetry.csv"
            process = Mock()
            process.create_time.return_value = 42
            process.is_running.side_effect = [True, True, True, False]
            results = [subprocess.TimeoutExpired("nvidia-smi", 15), SimpleNamespace(returncode=0, stdout=""), SimpleNamespace(returncode=0, stdout="2048,8151,90,70,51\n")]
            with patch("scripts.monitor_stage1.psutil.Process", return_value=process), patch("scripts.monitor_stage1.subprocess.run", side_effect=results), patch("scripts.monitor_stage1.time.sleep"), patch("sys.argv", ["monitor", "--pid", "123", "--output", str(path)]):
                monitor_main()
            summary = json.loads(path.with_suffix(".summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["query_failures"], 2)
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["peaks"]["memory_used_MiB"], 2048)


if __name__ == "__main__":
    unittest.main()
