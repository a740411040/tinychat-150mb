from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from tinychat.chat_format import IGNORE_LABEL, SPECIAL_IDS, encode_conversation
from tinychat.model import ModelConfig
from tinychat.train_stage1 import (
    PretrainStream,
    Trainer,
    accumulate_gradients,
    loss_sum,
    main,
    sft_batch,
)
from tinychat.training_state import SPLITS, file_identity, read_json


CPU = torch.device("cpu")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class Stage1TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        directory = TemporaryDirectory(prefix=".stage1-test-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def pretrain_fixture(self):
        data = self.root / "pretrain"
        data.mkdir()
        tokenizer_path = data / "tokenizer.json"
        vocabulary = {**SPECIAL_IDS, "[UNK]": 7, "你好": 8, "回答": 9, "补充": 10}
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.save(str(tokenizer_path))
        config = ModelConfig(vocab_size=len(vocabulary), n_layers=1, d_model=8, n_heads=2,
                             ffn_dim=16, max_seq_len=16)
        config_path = self.root / "config.json"
        write_json(config_path, asdict(config))
        manifest = {"schema_version": 1, "corpus_id": "stage1_v1", "frozen": True,
                    "vocab_size": len(vocabulary), "special_token_ids": SPECIAL_IDS,
                    "tokenizer_path": str(tokenizer_path),
                    "tokenizer_sha256": sha256(tokenizer_path.read_bytes()).hexdigest(),
                    "splits": {}, "files": {"tokenizer.json": file_identity(tokenizer_path)}}
        for split, count in zip(SPLITS, (121, 20, 14)):
            binary = data / f"{split}.bin"
            (np.arange(count, dtype="<u2") % len(vocabulary)).tofile(binary)
            jsonl = data / f"{split}.jsonl"
            jsonl.write_text('{"id":"document"}\n', encoding="utf-8")
            manifest["splits"][split] = {"documents": 1, "tokens": count, "binary_bytes": 2 * count}
            manifest["files"][binary.name] = file_identity(binary)
            manifest["files"][jsonl.name] = file_identity(jsonl)
        write_json(data / "manifest.json", manifest)
        return data, config_path, manifest, tokenizer

    def sft_fixture(self, base_manifest, tokenizer):
        data = self.root / "sft"
        data.mkdir()
        manifest = {**base_manifest, "corpus_id": "sft_stage1_v1", "splits": {}, "counts": {},
                    "files": {"tokenizer.json": base_manifest["files"]["tokenizer.json"]}}
        for split, count in zip(SPLITS, (5, 2, 1)):
            rows = []
            for index in range(count):
                messages = [{"role": "user", "content": "你好"},
                            {"role": "assistant", "content": "回答" if index % 2 else "回答 补充"}]
                rows.append({"id": f"{split}-{index}", "source": "fixture", "messages": messages,
                             **encode_conversation(tokenizer, messages)})
            jsonl = data / f"{split}.jsonl"
            jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            manifest["splits"][split] = {
                "documents": count, "tokens": sum(len(row["input_ids"]) for row in rows),
                "assistant_tokens": sum(label != IGNORE_LABEL for row in rows for label in row["labels"]),
            }
            manifest["counts"][split] = count
            manifest["files"][jsonl.name] = file_identity(jsonl)
        write_json(data / "manifest.json", manifest)
        return data

    def run_cli(self, stage, data, config, run, *options):
        args = ["--stage", stage, "--data-dir", str(data), "--run-dir", str(run), "--config", str(config),
                "--device", "cpu", "--micro-batch", "2", "--grad-accum", "2", "--eval-every", "1",
                "--checkpoint-every", "1", *options]
        with redirect_stdout(StringIO()) as output:
            code = main(args)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        return code, records

    def test_formal_tail_and_no_wrap(self):
        # The first 3,051 complete updates consume 99,975,168 targets.
        tail_tokens = 100_000_000 - 3_051 * (8 * 8 * 512)
        self.assertEqual(tail_tokens, 24_832)
        path = self.root / "tail.bin"
        tokens = np.arange(tail_tokens + 1, dtype="<u2") % 997
        tokens.tofile(path)
        stream = PretrainStream(path, 512, 997)
        batches = list(stream.batches(0, tail_tokens, 8))
        self.assertEqual([batch.targets for batch in batches], [4096] * 6 + [256])
        self.assertEqual(batches[-1].inputs.shape, (1, 256))
        labels = torch.cat([batch.labels.flatten() for batch in batches])
        torch.testing.assert_close(labels, torch.from_numpy(tokens[1:].astype(np.int64)))
        with self.assertRaisesRegex(ValueError, "exceed"):
            list(stream.batches(1, tail_tokens, 8))

    def test_sft_micrograd_matches_joint_assistant_only_cross_entropy(self):
        rows = [
            (torch.tensor([1, 4, 8, 6, 5, 9, 6, 2]), torch.tensor([-100] * 5 + [9, 6, 2]), 3),
            (torch.tensor([1, 4, 8, 6, 5, 9, 10, 6, 2]), torch.tensor([-100] * 5 + [9, 10, 6, 2]), 4),
            (torch.tensor([1, 4, 8, 6, 5, 9, 6, 2]), torch.tensor([-100] * 5 + [9, 6, 2]), 3),
        ]
        model = torch.nn.Embedding(11, 11)
        batch_a, batch_b = sft_batch(rows[:2]), sft_batch(rows[2:])
        self.assertEqual((batch_a.targets, batch_b.targets, batch_a.padding_tokens), (7, 3, 1))
        self.assertEqual(batch_a.labels[0].tolist(), [-100] * 4 + [9, 6, 2, -100])
        loss = accumulate_gradients(model, [batch_a, batch_b], 10, CPU)
        gradient = model.weight.grad.clone()
        model.zero_grad()
        reference = loss_sum(model, sft_batch(rows), CPU) / 10
        reference.backward()
        self.assertAlmostEqual(loss, reference.item(), places=6)
        torch.testing.assert_close(gradient, model.weight.grad)

    def test_resume_rolls_back_interrupted_micro_then_sft_finishes_one_epoch(self):
        data, config, manifest, tokenizer = self.pretrain_fixture()
        run = self.root / "base_run"
        real_loss = loss_sum

        def interrupt_after_a_micro(model, batch, device):
            if model.training:
                interrupt_after_a_micro.training_micros += 1
                if interrupt_after_a_micro.training_micros == 4:
                    random.random()
                    np.random.random()
                    torch.rand(1)
                    raise KeyboardInterrupt("fixture interruption after one tail micro")
            return real_loss(model, batch, device)

        interrupt_after_a_micro.training_micros = 0
        with (
            patch("tinychat.train_stage1.loss_sum", side_effect=interrupt_after_a_micro),
            patch("tinychat.train_stage1.VALIDATION_TOKENS", 7),
        ):
            code, records = self.run_cli("pretrain", data, config, run, "--target-tokens", "117")
        self.assertEqual(code, 130, records)
        paused = read_json(run / "metrics.json")
        self.assertEqual((paused["stop_reason"], paused["completed_updates"], paused["cumulative_valid_loss_tokens"]),
                         ("interrupted", 1, 64))
        self.assertFalse(paused["optimizer_state_unsafe"])
        saved_rng = torch.load(paused["checkpoint"], weights_only=False)["rng"]
        self.assertEqual(random.getstate(), saved_rng["python"])
        np.testing.assert_array_equal(np.random.get_state()[1], saved_rng["numpy"][1])
        self.assertEqual(np.random.get_state()[2:], saved_rng["numpy"][2:])
        torch.testing.assert_close(torch.get_rng_state(), saved_rng["torch_cpu"])
        with (
            patch("tinychat.train_stage1.VALIDATION_TOKENS", 7),
            patch("tinychat.train_stage1.FINAL_VALIDATION_TOKENS", 13),
        ):
            code, records = self.run_cli("pretrain", data, config, run, "--target-tokens", "117",
                                         "--resume", str(run / "latest.json"))
        self.assertEqual(code, 0, records)
        [tail] = [event for event in records if event["event"] == "update"]
        self.assertEqual((tail["step"], tail["tokens_this_update"], tail["padding_tokens"], tail["lr"]),
                         (2, 53, 11, 3e-5))
        pretrain = read_json(run / "metrics.json")
        self.assertEqual((pretrain["stop_reason"], pretrain["cumulative_valid_loss_tokens"]),
                         ("target_tokens_reached", 117))
        base = Path(pretrain["checkpoint"])
        base_identity = file_identity(base)
        self.assertEqual([pretrain["validation"][phase]["valid_loss_tokens"]
                          for phase in ("start", "final_comparable", "final")], [7, 7, 13])
        self.assertEqual([record["phase"] for record in records if record["event"] == "validation"][-2:],
                         ["final", "final_comparable"])
        self.assertEqual(torch.load(base, weights_only=False)["validations"], pretrain["validation"])
        cuda_trainer = SimpleNamespace(device=torch.device("cuda"))
        with self.assertRaisesRegex(ValueError, "full 100,000,000-token budget"):
            Trainer.initialize_from_base(cuda_trainer, base)

        sft = self.sft_fixture(manifest, tokenizer)
        sft_run = self.root / "sft_run"
        code, records = self.run_cli("sft", sft, config, sft_run, "--init-checkpoint", str(base))
        self.assertEqual(code, 0, records)
        self.assertEqual([event["examples_this_update"] for event in records if event["event"] == "update"], [4, 1])
        metrics = read_json(sft_run / "metrics.json")
        self.assertEqual(metrics["stop_reason"], "epoch_complete")
        self.assertEqual((metrics["cumulative_examples"], metrics["cumulative_valid_loss_tokens"]), (5, 18))
        self.assertEqual(metrics["validation"]["final"]["valid_loss_tokens"], 7)
        final, comparable = (metrics["validation"][phase] for phase in ("final", "final_comparable"))
        self.assertEqual(comparable["loss"], final["loss"])
        self.assertEqual(comparable["validation_examples"], metrics["validation"]["start"]["validation_examples"])
        self.assertEqual((comparable["reused_from"], comparable["evaluation_seconds"]), ("final", 0.0))
        self.assertEqual(file_identity(base), base_identity)

    def test_frozen_file_rewrite_refuses_resume(self):
        data, config, _, _ = self.pretrain_fixture()
        run = self.root / "run"
        code, records = self.run_cli("pretrain", data, config, run, "--target-tokens", "85", "--max-updates", "1")
        self.assertEqual(code, 0, records)
        old_metrics = read_json(run / "metrics.json")
        binary = data / "train.bin"
        with binary.open("r+b") as file:
            file.write(b"\x02\x00")
        code, records = self.run_cli("pretrain", data, config, run, "--target-tokens", "85",
                                     "--resume", str(run / "latest.json"))
        self.assertEqual(code, 1, records)
        self.assertIn("Frozen file identity changed", records[-1]["error"])
        self.assertEqual(read_json(run / "metrics.json"), old_metrics)

    def test_optimizer_failure_preserves_last_published_boundary(self):
        data, config, _, _ = self.pretrain_fixture()
        run = self.root / "run"
        original_step = torch.optim.AdamW.step
        attempts = 0

        def fail_inside_step(optimizer):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                with torch.no_grad():
                    optimizer.param_groups[0]["params"][0].add_(1)
                raise RuntimeError("fixture failure after optimizer mutation")
            return original_step(optimizer)

        with patch.object(torch.optim.AdamW, "step", fail_inside_step):
            code, records = self.run_cli("pretrain", data, config, run, "--target-tokens", "117")
        self.assertEqual(code, 1, records)
        metrics = read_json(run / "metrics.json")
        self.assertEqual((metrics["stop_reason"], metrics["completed_updates"], metrics["checkpoint_step"]),
                         ("failure", 1, 1))
        self.assertTrue(metrics["optimizer_state_unsafe"])
        self.assertEqual(read_json(run / "latest.json")["checkpoint"], metrics["checkpoint"])
        self.assertEqual([path.name for path in run.glob("step_*.pt")], ["step_00000001.pt"])


if __name__ == "__main__":
    unittest.main()
