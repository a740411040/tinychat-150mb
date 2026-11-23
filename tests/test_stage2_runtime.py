import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tinychat.stage2_runtime import DiskBudgetError, GIB, ensure_disk_budget


class DiskGuardTest(unittest.TestCase):
    def test_atomic_allocation_keeps_reserve_and_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("tinychat.stage2_runtime.shutil.disk_usage", return_value=SimpleNamespace(free=33 * GIB)):
                self.assertEqual(ensure_disk_budget(directory, required_bytes=GIB), 33 * GIB)
                with self.assertRaises(DiskBudgetError):
                    ensure_disk_budget(directory, required_bytes=GIB + 1)

    def test_missing_child_uses_existing_filesystem(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            with patch("tinychat.stage2_runtime.shutil.disk_usage", return_value=SimpleNamespace(free=32 * GIB - 1)) as usage:
                with self.assertRaises(DiskBudgetError):
                    ensure_disk_budget(Path(directory) / "not-created" / "checkpoint.pt")
                self.assertEqual(usage.call_args.args[0], Path(directory).resolve())

    def test_rejects_invalid_allocation(self):
        with self.assertRaises(ValueError):
            ensure_disk_budget(".", required_bytes=-1)
        with self.assertRaises(ValueError):
            ensure_disk_budget(".", min_free_gib=float("nan"))


if __name__ == "__main__":
    unittest.main()
