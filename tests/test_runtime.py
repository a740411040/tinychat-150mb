from io import StringIO
import json
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tinychat.benchmark import (
    SequentialBatcher,
    capture_rng,
    log_json,
    read_data_identity,
    restore_rng,
    train,
    validate_resume_data,
)


class BatcherTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.token_path = Path(directory.name) / "train.bin"
        np.arange(7, dtype="<u2").tofile(self.token_path)

    def test_wrap_preserves_pair_alignment_across_rows(self):
        batcher = SequentialBatcher(self.token_path, micro_batch=2, seq_len=2)
        inputs, targets = batcher.next_batch()
        self.assertEqual(inputs.tolist(), [[0, 1], [2, 3]])
        self.assertEqual(targets.tolist(), [[1, 2], [3, 4]])
        self.assertEqual(batcher.state_dict(), {"cursor": 4, "wraps": 0})

        inputs, targets = batcher.next_batch()
        self.assertEqual(inputs.tolist(), [[4, 5], [6, 0]])
        self.assertEqual(targets.tolist(), [[5, 6], [0, 1]])
        self.assertEqual(batcher.state_dict(), {"cursor": 1, "wraps": 1})

    def test_resume_replays_stream_and_random_states(self):
        original_rng = capture_rng()
        self.addCleanup(restore_rng, original_rng)
        batcher = SequentialBatcher(self.token_path, micro_batch=1, seq_len=4)
        batcher.next_batch()
        state = batcher.state_dict()
        rng = capture_rng()
        expected_inputs, _ = batcher.next_batch()
        expected_rng = random.random(), np.random.random(), torch.rand(3)
        expected_position = batcher.state_dict()
        batcher.next_batch()

        resumed = SequentialBatcher(self.token_path, micro_batch=1, seq_len=4)
        resumed.load_state_dict(state)
        restore_rng(rng)
        replay_inputs, _ = resumed.next_batch()
        self.assertTrue(torch.equal(replay_inputs, expected_inputs))
        self.assertEqual(resumed.state_dict(), expected_position)
        self.assertEqual((random.random(), np.random.random()), expected_rng[:2])
        self.assertTrue(torch.equal(torch.rand(3), expected_rng[2]))

    def test_time_limit_discards_partial_update(self):
        batcher = SequentialBatcher(self.token_path, micro_batch=1, seq_len=2)
        model = torch.nn.Sequential(torch.nn.Embedding(7, 4), torch.nn.Dropout(), torch.nn.Linear(4, 7))
        parameters_before = [param.detach().clone() for param in model.parameters()]
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        args = SimpleNamespace(seconds=1, grad_accum=4)
        steps = StringIO()
        elapsed = 0.0

        def finish_microbatch(_device):
            nonlocal elapsed
            elapsed = 2.0

        rng_at_start = torch.get_rng_state()
        with (
            patch("tinychat.benchmark.perf_counter", side_effect=lambda: elapsed),
            patch("torch.cuda.synchronize", side_effect=finish_microbatch),
        ):
            stats = train(model, optimizer, scheduler, scaler, batcher, torch.float16, args, steps, 0)

        self.assertEqual(stats["stop_reason"], "time_limit")
        self.assertEqual(stats["consumed_tokens"], 2)
        self.assertEqual(stats["discarded_tokens"], 2)
        self.assertEqual(stats["trained_tokens"], 0)
        self.assertEqual(stats["effective_tokens_per_second"], 0.0)
        self.assertEqual(batcher.state_dict(), {"cursor": 0, "wraps": 0})
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_at_start))
        self.assertEqual(optimizer.state, {})
        self.assertEqual(steps.getvalue(), "")
        for param, before in zip(model.parameters(), parameters_before):
            torch.testing.assert_close(param, before)
            self.assertIsNone(param.grad)

    def test_resume_rejects_same_length_rewrite(self):
        data_dir = self.token_path.parent
        tokenizer_path = data_dir / "tokenizer.json"
        tokenizer_path.write_text("{}", encoding="utf-8")
        meta = {
            "repo": "example/pilot",
            "revision": "example-revision",
            "file": "data/shard.parquet",
            "seed": 42,
            "splits": {"train": {"tokens": 7}},
            "vocab_size": 7,
            "tokenizer_path": str(tokenizer_path),
        }
        (data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        checkpoint = {"data_identity": read_data_identity(data_dir)}
        validate_resume_data(checkpoint, read_data_identity(data_dir))

        original_stat = self.token_path.stat()
        np.arange(6, -1, -1, dtype="<u2").tofile(self.token_path)
        os.utime(
            self.token_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 2_000_000_000)
        )
        self.assertEqual(self.token_path.stat().st_size, original_stat.st_size)
        with self.assertRaisesRegex(SystemExit, "data identity does not match"):
            validate_resume_data(checkpoint, read_data_identity(data_dir))


class LogTests(unittest.TestCase):
    def test_nonfinite_values_are_null_and_failure_details_are_preserved(self):
        record = {
            "validation_loss_after": float("nan"),
            "other_values": (float("inf"), np.float32("-inf"), 0.5),
            "nan": True,
            "nonfinite": True,
            "oom": True,
            "validation_errors": {"after": "CUDA out of memory"},
        }
        decoded = json.loads(log_json(record))
        self.assertEqual(
            decoded,
            {**record, "validation_loss_after": None, "other_values": [None, None, 0.5]},
        )


if __name__ == "__main__":
    unittest.main()
