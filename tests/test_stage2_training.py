from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from tinychat import train_stage2 as stage2
from tinychat.chat_format import IGNORE_LABEL, SPECIAL_IDS, encode_conversation
from tinychat.inference import load_model
from tinychat.model import ModelConfig
from tinychat.train_stage1 import MicroBatch
from tinychat.training_state import SPLITS, file_identity, read_json


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class Stage2TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        directory = TemporaryDirectory(prefix=".stage2-test-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        vocabulary = {**SPECIAL_IDS, "[UNK]": 7, "中文": 8, "问题": 9, "回答": 10}
        self.tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        self.tokenizer_path = self.root / "tokenizer.json"
        self.tokenizer.save(str(self.tokenizer_path))
        self.config = ModelConfig(vocab_size=len(vocabulary), n_layers=1, d_model=8, n_heads=2,
                                  ffn_dim=16, max_seq_len=16)
        self.config_path = self.root / "config.json"
        write_json(self.config_path, asdict(self.config))
        patcher = patch.multiple(stage2, PILOT_UPDATES=3, PILOT_WARMUP_UPDATES=2, WARMUP_TOKENS=1011,
                                 STABLE_TOKENS=1503, FINAL_TOKENS=3123)
        patcher.start()
        self.addCleanup(patcher.stop)

    def common_manifest(self):
        return {"schema_version": 1, "frozen": True, "vocab_size": self.config.vocab_size,
                "special_token_ids": SPECIAL_IDS, "tokenizer_path": str(self.tokenizer_path),
                "tokenizer_sha256": sha256(self.tokenizer_path.read_bytes()).hexdigest(),
                "splits": {}, "files": {"tokenizer.json": file_identity(self.tokenizer_path)}}

    def pretrain_data(self):
        data = self.root / "pretrain"
        data.mkdir()
        manifest = {**self.common_manifest(), "corpus_id": stage2.PRETRAIN_CORPUS, "dtype": "<u2"}
        for split, count in zip((*SPLITS, "general_validation"), (3300, 37, 19, 25)):
            filename = f"{split}.bin"
            binary = data / filename
            tokens = np.resize(np.array([8, 9, 7, 10, SPECIAL_IDS["<|eos|>"]], dtype="<u2"), count)
            tokens.tofile(binary)
            entry = {"documents": count // 5, "tokens": count, "binary_bytes": 2 * count}
            if split == "general_validation":
                manifest["stage2"] = {"general_validation": {**entry, "path": filename}}
            else:
                jsonl = data / f"{split}.jsonl"
                write_json(jsonl, {"id": split})
                manifest["files"][jsonl.name] = file_identity(jsonl)
                manifest["splits"][split] = entry
            manifest["files"][filename] = file_identity(binary)
        manifest["splits"]["train"]["special_token_counts"] = {
            token: 660 if token == "<|eos|>" else 0 for token in SPECIAL_IDS
        }
        write_json(data / "manifest.json", manifest)
        return data

    def sft_data(self):
        data = self.root / "sft"
        data.mkdir()
        manifest = {**self.common_manifest(), "corpus_id": "sft_stage2_natural",
                    "chat_format": {"final_eos": False}, "counts": {}}
        for split, count in zip(SPLITS, (37, 7, 3)):
            input_tokens, assistant_tokens = 0, 0
            path = data / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as file:
                for index in range(count):
                    messages = [{"role": "user", "content": "中文 问题"},
                                {"role": "assistant", "content": " ".join(["回答"] * (index % 4 + 1))}]
                    encoded = encode_conversation(self.tokenizer, messages, final_eos=False)
                    input_tokens += len(encoded["input_ids"])
                    assistant_tokens += sum(label != IGNORE_LABEL for label in encoded["labels"][1:])
                    file.write(json.dumps({"id": f"{split}-{index}", "source": "fixture", "messages": messages,
                                           **encoded}, ensure_ascii=False) + "\n")
            manifest["splits"][split] = {"documents": count, "tokens": input_tokens,
                                         "assistant_tokens": assistant_tokens}
            manifest["counts"][split] = count
            manifest["files"][path.name] = file_identity(path)
        write_json(data / "manifest.json", manifest)
        return data

    def command(self, phase, data, run, *options):
        return ["--phase", phase, "--data-dir", str(data), "--run-dir", str(run), "--config", str(self.config_path),
                "--device", "cpu", "--checkpoint-every", "1", "--eval-every", "1", *map(str, options)]

    def run_cli(self, command, status=0):
        with redirect_stdout(StringIO()) as output:
            code = stage2.main(command)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(code, status, records[-1].get("error") if records else "No training events")
        return records

    def test_pilot_interruption_restores_rng_and_exact_result(self):
        data = self.pretrain_data()
        baseline, interrupted = self.root / "baseline", self.root / "interrupted"
        args = self.command("pilot", data, interrupted, "--lr", "0.001", "--residual-init", "original")
        self.run_cli(self.command("pilot", data, baseline, "--lr", "0.001", "--residual-init", "original"))
        real_loss = stage2.pretrain_loss_sum
        batches = 0

        def interrupt_mid_update(model, batch, device, **options):
            nonlocal batches
            if model.training:
                batches += 1
                if batches == 10:
                    random.random()
                    np.random.random()
                    torch.rand(1)
                    raise KeyboardInterrupt("mid-update interruption")
            return real_loss(model, batch, device, **options)

        with patch.object(stage2, "pretrain_loss_sum", side_effect=interrupt_mid_update):
            self.run_cli(args, status=130)
        latest = interrupted / "latest.pt"
        paused = stage2.load_checkpoint(latest)
        self.assertEqual((paused["progress"]["updates"], paused["global_tokens"], paused["train_cursor"]), (1, 1024, 1024))
        self.assertEqual(paused["optimizer"]["param_groups"][0]["lr"], 0.0005)
        self.assertEqual(random.getstate(), paused["rng"]["python"])
        np.testing.assert_array_equal(np.random.get_state()[1], paused["rng"]["numpy"][1])
        torch.testing.assert_close(torch.get_rng_state(), paused["rng"]["torch_cpu"])
        resume = [*args, "--resume", str(latest)]
        self.run_cli([*resume, "--lr", "0.0006"], status=1)
        self.run_cli(resume)
        actual, expected = stage2.load_checkpoint(latest), stage2.load_checkpoint(baseline / "latest.pt")
        self.assertTrue(actual["completed"])
        for name in expected["model"]:
            torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=0, atol=0)
        self.assertEqual(actual["optimizer"]["param_groups"][0]["lr"], 0.001)
        self.assertEqual({path.name for path in interrupted.glob("*.pt")}, {"latest.pt"})

    def test_stable_explicit_decay_preserves_optimizer_cursor_and_exports(self):
        data = self.pretrain_data()
        stable, decay = self.root / "stable", self.root / "decay"
        self.run_cli(self.command("stable", data, stable))
        stable_path = stable / "stable.pt"
        stable_state = stage2.load_checkpoint(stable_path)
        self.assertEqual(stable_state["global_tokens"], 1503)
        self.assertEqual(stable_state["settings"]["masked_token_ids"], [0, 1, 3, 4, 5, 6])
        self.assertEqual(stable_state["optimizer"]["param_groups"][0]["lr"], 6e-4)
        self.assertEqual(set(stable_state["validations"]["final"]["scopes"]), {"general", "mixed", "general_full_vocab"})
        self.run_cli(self.command("decay", data, decay, "--resume", stable_path), status=1)
        self.run_cli(self.command("decay", data, decay, "--branch-from", stable_path, "--max-updates", 1))
        latest = decay / "latest.pt"
        decay_state = stage2.load_checkpoint(latest)
        self.assertEqual(decay_state["global_tokens"], 2527)
        self.assertEqual(decay_state["branch_checkpoint"]["global_tokens"], 1503)
        self.assertEqual({value["step"].item() for value in decay_state["optimizer"]["state"].values()}, {3})
        self.assertLess(decay_state["optimizer"]["param_groups"][0]["lr"], 6e-4)
        self.run_cli(self.command("decay", data, decay, "--resume", latest))
        final = stage2.load_checkpoint(decay / "final.pt")
        self.assertEqual((final["global_tokens"], final["progress"]["padding_tokens"]), (3123, 13))
        self.assertEqual(final["optimizer"]["param_groups"][0]["lr"], 0)
        self.assertEqual(stage2.load_checkpoint(stable_path)["global_tokens"], 1503)
        self.assertEqual({path.name for path in decay.glob("*.pt")}, {"latest.pt", "final.pt"})
        artifact = self.root / "artifact"
        self.run_cli(["export", "--checkpoint", str(decay / "final.pt"), "--output", str(artifact)])
        model, _ = load_model(artifact, device="cpu")
        torch.testing.assert_close(model.token_embedding.weight,
                                   final["model"]["token_embedding.weight"].bfloat16().float())

    def test_sft_uses_same_bucketed_data_and_unchanged_base_embeddings(self):
        pretrain = self.pretrain_data()
        stable, decay = self.root / "stable", self.root / "decay"
        self.run_cli(self.command("stable", pretrain, stable))
        self.run_cli(self.command("decay", pretrain, decay, "--branch-from", stable / "stable.pt"))
        new_base = decay / "final.pt"
        legacy_base = self.root / "legacy.pt"
        new_state = stage2.load_checkpoint(new_base)
        torch.save({**new_state, "schema_version": 1,
                    "model": {name: weight + 0.01 for name, weight in new_state["model"].items()},
                    "progress": {**new_state["progress"], "tokens": 100_000_000},
                    "settings": {**new_state["settings"], "planned_tokens": 100_000_000},
                    "validations": {"final": {"nonfinite": False, "loss": 1.0,
                                              "step": new_state["progress"]["updates"]}}}, legacy_base)
        sft = self.sft_data()
        states = []
        for name, base in [("legacy_sft", legacy_base), ("new_sft", new_base)]:
            run = self.root / name
            command = self.command("sft", sft, run, "--init-checkpoint", base)
            trainer = stage2.Trainer(stage2.parse_args(command))
            base_weights = stage2.load_checkpoint(base)["model"]
            torch.testing.assert_close(trainer.model.token_embedding.weight, base_weights["token_embedding.weight"],
                                       rtol=0, atol=0)
            self.assertEqual(trainer.optimizer.state_dict()["state"], {})
            self.assertEqual([group["weight_decay"] for group in trainer.optimizer.param_groups], [0.01, 0.0])
            self.run_cli(command)
            self.assertTrue((run / "best.pt").is_file())
            states.append(stage2.load_checkpoint(run / "final.pt"))
        old, new = states
        for field in ("sft_order", "data_identity", "settings", "progress"):
            self.assertEqual(old[field], new[field])
        self.assertEqual([old["base_checkpoint"]["pretrain_tokens"], new["base_checkpoint"]["pretrain_tokens"]],
                         [100_000_000, 3123])
        self.assertEqual(new["progress"]["examples"], 37)
        self.assertEqual(sorted(new["sft_order"]), list(range(37)))

    def test_mask_excludes_gradients_checks_inputs_and_ignores_padding(self):
        model = torch.nn.Embedding(self.config.vocab_size, self.config.vocab_size)
        excluded = torch.zeros(self.config.vocab_size, dtype=torch.bool)
        excluded[0], excluded[1] = True, True
        batch = MicroBatch(torch.tensor([[8, 0]]), torch.tensor([[2, IGNORE_LABEL]]), 1, 1, 1)
        loss = stage2.pretrain_loss_sum(model, batch, torch.device("cpu"), masked=torch.tensor([0, 1]), excluded=excluded)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(model.weight.grad[:, :2], torch.zeros_like(model.weight.grad[:, :2])))
        for inputs, labels in [(torch.tensor([[0]]), torch.tensor([[2]])),
                               (torch.tensor([[8]]), torch.tensor([[1]]))]:
            with self.assertRaisesRegex(ValueError, "masked control token"):
                stage2.pretrain_loss_sum(model, MicroBatch(inputs, labels, 1, 1), torch.device("cpu"),
                                         masked=torch.tensor([0, 1]), excluded=excluded)

    def test_disk_pause_preserves_committed_run_and_rejects_data_drift(self):
        data = self.pretrain_data()
        run = self.root / "disk"
        command = self.command("pilot", data, run)
        writes = 0

        def guard(path, required_bytes=0):
            nonlocal writes
            if required_bytes:
                writes += 1
                if writes > 1:
                    raise stage2.DiskBudgetError("fixture disk reserve")

        with patch.object(stage2, "ensure_disk_budget", side_effect=guard):
            self.run_cli(command, status=75)
        paused = read_json(run / "metrics.json")
        self.assertEqual((paused["stop_reason"], paused["completed_updates"], paused["checkpoint_step"]),
                         ("paused_disk", 1, 0))
        latest = run / "latest.pt"
        self.assertEqual(stage2.load_checkpoint(latest)["global_tokens"], 0)
        self.run_cli([*command, "--resume", str(latest)])
        with (data / "general_validation.bin").open("ab") as file:
            file.write(b"00")
        records = self.run_cli([*command, "--resume", str(latest)], status=1)
        self.assertIn("Frozen file identity changed", records[-1]["error"])

    def test_pause_file_stops_at_update_boundary_and_can_resume(self):
        data = self.pretrain_data()
        run = self.root / "paused_run"
        pause = self.root / "PAUSE"
        command = self.command("pilot", data, run, "--pause-file", pause)
        real_update = stage2.Trainer.update

        def request_pause(trainer):
            real_update(trainer)
            pause.touch()

        with patch.object(stage2.Trainer, "update", request_pause):
            self.run_cli(command)
        metrics = read_json(run / "metrics.json")
        self.assertEqual((metrics["stop_reason"], metrics["pause_trigger"], metrics["checkpoint_step"]),
                         ("paused", "file", 1))
        self.assertTrue(pause.exists())
        resume = [*command, "--resume", str(run / "latest.pt")]
        self.run_cli(resume)
        self.assertEqual(read_json(run / "metrics.json")["checkpoint_step"], 1)
        pause.unlink()
        self.run_cli(resume)
        self.assertEqual(read_json(run / "metrics.json")["stop_reason"], "complete")


if __name__ == "__main__":
    unittest.main()
