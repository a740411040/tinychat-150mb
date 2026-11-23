from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import run_special_ablation as runner


class LauncherLifecycleTests(unittest.TestCase):
    def test_state_write_error_reaps_only_owned_child(self):
        with tempfile.TemporaryDirectory() as temp:
            child = MagicMock()
            child.pid = 123
            child.poll.return_value = None
            child.wait.return_value = 0
            state = {"completed_phases": []}
            with patch.object(runner, "REPORT", Path(temp)), patch.object(runner, "LOGS", Path(temp)), \
                    patch.object(runner.subprocess, "Popen", return_value=child), \
                    patch.object(runner, "save_state", side_effect=OSError("disk write failed")):
                with self.assertRaises(OSError):
                    runner.run_child(state, "train", "test", ["owned-process"])
            child.terminate.assert_called_once()
            child.wait.assert_called_once_with(timeout=15)
            self.assertIsNone(state["child_pid"])

    def test_stop_request_is_observed_while_child_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            child = MagicMock()
            child.pid = 124
            child.poll.return_value = None
            calls = []

            def waiting(timeout):
                calls.append(timeout)
                if len(calls) == 1:
                    (Path(temp) / "STOP_REQUESTED").write_text("pause", encoding="utf-8")
                    raise subprocess.TimeoutExpired(["owned-process"], timeout)
                return 0

            child.wait.side_effect = waiting
            state = {"completed_phases": []}
            with patch.object(runner, "REPORT", Path(temp)), patch.object(runner, "LOGS", Path(temp)), \
                    patch.object(runner.subprocess, "Popen", return_value=child), patch.object(runner, "save_state"):
                with self.assertRaisesRegex(RuntimeError, "stop requested"):
                    runner.run_child(state, "train", "test", ["owned-process"])
            child.terminate.assert_called_once()
            self.assertEqual(calls, [30, 15])
            self.assertIsNone(state["child_pid"])


if __name__ == "__main__":
    unittest.main()
