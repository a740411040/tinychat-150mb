import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import tests.test_stage1_training as fixtures
from tinychat.chat_format import encode_conversation
from tinychat.export_stage1 import export_chat_metadata, export_manifest
from tinychat.special_token_ablation import EOS, PAD, RESET_IDS, prepare_no_eos_data, reset_unseen_specials
from tinychat.train_stage1 import read_sft_data
from tinychat.training_state import SPLITS, read_frozen_manifest, read_json


class SpecialTokenAblationTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.Stage1TrainingTests.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.Stage1TrainingTests.tearDownClass.__func__)
    setUp = fixtures.Stage1TrainingTests.setUp
    pretrain_fixture = fixtures.Stage1TrainingTests.pretrain_fixture
    sft_fixture = fixtures.Stage1TrainingTests.sft_fixture
    run_cli = fixtures.Stage1TrainingTests.run_cli

    def test_legacy_format_and_generation_prefix(self):
        _, _, _, tokenizer = self.pretrain_fixture()
        messages = [
            {"role": "user", "content": "你好"}, {"role": "assistant", "content": "回答"},
            {"role": "user", "content": "补充"}, {"role": "assistant", "content": "回答 补充"},
        ]
        legacy = encode_conversation(tokenizer, messages)
        self.assertEqual(legacy["input_ids"], [1, 3, 7, 6, 4, 8, 6, 5, 9, 6, 4, 10, 6, 5, 9, 10, 6, 2])
        self.assertEqual(legacy["labels"], [-100] * 8 + [9, 6] + [-100] * 4 + [9, 10, 6, 2])
        self.assertEqual(encode_conversation(tokenizer, messages, final_eos=True), legacy)
        self.assertEqual(encode_conversation(tokenizer, messages, final_eos=False),
                         {key: value[:-1] for key, value in legacy.items()})
        prefix = encode_conversation(tokenizer, messages[:-1], generation_prompt=True)
        self.assertEqual(encode_conversation(tokenizer, messages[:-1], generation_prompt=True, final_eos=False), prefix)
        self.assertEqual(prefix["input_ids"], legacy["input_ids"][:-4])

    def test_derived_manifest_and_frozen_rows(self):
        _, config, base_manifest, tokenizer = self.pretrain_fixture()
        source = self.sft_fixture(base_manifest, tokenizer)
        output = self.root / "derived"
        prepare_no_eos_data(source, output)
        original = read_json(source / "manifest.json")
        derived, identity = read_frozen_manifest(output, "sft", original["vocab_size"])
        self.assertIs(derived["chat_format"]["final_eos"], False)
        self.assertEqual(derived["derived_from"]["corpus_id"], original["corpus_id"])
        self.assertEqual(derived["files"]["tokenizer.json"], original["files"]["tokenizer.json"])
        self.assertEqual(derived["counts"], original["counts"])
        for split in SPLITS:
            rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            expected = [json.loads(line) for line in (source / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            for row in expected:
                del row["input_ids"][-1]
                del row["labels"][-1]
            self.assertEqual(rows, expected)
            lengths = [len(row["input_ids"]) for row in rows]
            stats = derived["splits"][split]
            self.assertEqual((stats["tokens"], stats["min_context"], stats["max_context"]),
                             (sum(lengths), min(lengths), max(lengths)))
            self.assertEqual(stats["assistant_tokens"], original["splits"][split]["assistant_tokens"] - len(rows))
        model_config = SimpleNamespace(**read_json(config))
        read_sft_data(output, derived, model_config)
        with self.assertRaises(FileExistsError):
            prepare_no_eos_data(source, output)
        state = {"config": read_json(config), "data_identity": identity}
        with self.assertRaisesRegex(ValueError, "identity differs"):
            export_manifest(state, source, "sft")
        with self.assertRaisesRegex(ValueError, "disagree"):
            export_chat_metadata(state, derived)

    def test_reset_preserves_rng_and_only_changes_unseen_rows(self):
        first = SimpleNamespace(token_embedding=torch.nn.Embedding(21, 12))
        second = SimpleNamespace(token_embedding=torch.nn.Embedding(21, 12))
        with torch.no_grad():
            first.token_embedding.weight[7:].add_(3.0)
            second.token_embedding.weight.copy_(first.token_embedding.weight)
        old = first.token_embedding.weight.detach().clone()
        rng_before = torch.get_rng_state()
        record = reset_unseen_specials(first)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        torch.rand(4)
        self.assertEqual(reset_unseen_specials(second), record)
        weight = first.token_embedding.weight.detach()
        self.assertTrue(torch.equal(weight, second.token_embedding.weight))
        self.assertTrue(torch.equal(old[EOS], weight[EOS]))
        self.assertTrue(torch.equal(old[7:], weight[7:]))
        self.assertEqual(record["changed_ids"], [PAD, *RESET_IDS])
        self.assertEqual(float(weight[PAD].norm()), 0)
        target = old[7:].norm(dim=1).mean()
        torch.testing.assert_close(weight[list(RESET_IDS)].norm(dim=1), target.expand(len(RESET_IDS)))
        self.assertAlmostEqual(record["common_norm"], 0.1 * float(target))
        cosine = torch.tensor(record["after"]["pairwise_cosines"])[list(RESET_IDS)][:, list(RESET_IDS)]
        self.assertLessEqual(float((cosine - torch.eye(len(RESET_IDS))).abs().max()), 0.010001)

    def test_reset_applied_once_and_resume_checks_factor(self):
        base_data, config, manifest, tokenizer = self.pretrain_fixture()
        base_run = self.root / "base"
        code, events = self.run_cli("pretrain", base_data, config, base_run, "--target-tokens", "64")
        self.assertEqual(code, 0, events)
        base = read_json(base_run / "metrics.json")["checkpoint"]
        source = self.sft_fixture(manifest, tokenizer)
        derived = self.root / "derived"
        prepare_no_eos_data(source, derived)
        run = self.root / "reset-noeos"
        reset_options = ("--reset-unseen-specials", "--special-ablation-diagnostics")
        code, events = self.run_cli("sft", derived, config, run, "--init-checkpoint", base,
                                    "--max-updates", "1", *reset_options)
        self.assertEqual(code, 0, events)
        first_update = next(event for event in events if event["event"] == "update")
        self.assertEqual(len(first_update["embedding_gradient_norms"]["special_rows"]), 7)
        saved = Path(read_json(run / "metrics.json")["checkpoint"])
        checkpoint = torch.load(saved, weights_only=False)
        self.assertEqual(checkpoint["special_token_initialization"]["changed_ids"], [PAD, *RESET_IDS])
        self.assertIs(checkpoint["settings"]["special_token_ablation"]["final_eos"], False)
        derived_manifest = read_json(derived / "manifest.json")
        export_format, metadata = export_chat_metadata(checkpoint, derived_manifest)
        self.assertIs(export_format["final_eos"], False)
        self.assertEqual(metadata["special_token_initialization"], checkpoint["special_token_initialization"])
        provenance = checkpoint["special_token_initialization"]
        missing_provenance = (
            {"seed": provenance["seed"], "recipe": provenance["recipe"]},
            {**provenance, "changed_ids": [PAD, EOS]},
            {**provenance, "before": {"pairwise_cosines": provenance["before"]["pairwise_cosines"]}},
            {**provenance, "after": {"row_norms": provenance["after"]["row_norms"]}},
        )
        for incomplete in missing_provenance:
            invalid_checkpoint = {**checkpoint, "special_token_initialization": incomplete}
            with self.assertRaisesRegex(ValueError, "reset provenance"):
                export_chat_metadata(invalid_checkpoint, derived_manifest)
            with patch("tinychat.train_stage1.load_checkpoint", return_value=invalid_checkpoint):
                code, events = self.run_cli("sft", derived, config, run, "--resume", str(saved), *reset_options)
            self.assertEqual(code, 1, events)
            self.assertIn("reset provenance", events[-1]["error"])
        for data, options in ((derived, ()), (derived, (*reset_options, "--reset-special-seed", "9")),
                              (source, reset_options)):
            code, events = self.run_cli("sft", data, config, run, "--resume", str(saved), *options)
            self.assertEqual(code, 1, events)
            self.assertIn("differs", events[-1]["error"])
        with patch("tinychat.train_stage1.reset_unseen_specials", side_effect=AssertionError("reset on resume")):
            code, events = self.run_cli("sft", derived, config, run, "--resume", str(saved), *reset_options)
        self.assertEqual(code, 0, events)
        final = read_json(run / "metrics.json")
        self.assertEqual((final["stop_reason"], final["completed_updates"]), ("epoch_complete", 2))
        self.assertEqual(final["special_token_initialization"], checkpoint["special_token_initialization"])

        legacy = self.root / "legacy"
        code, events = self.run_cli("sft", source, config, legacy, "--init-checkpoint", base, "--max-updates", "1")
        self.assertEqual(code, 0, events)
        legacy_path = Path(read_json(legacy / "metrics.json")["checkpoint"])
        state = torch.load(legacy_path, weights_only=False)
        del state["settings"]["special_token_ablation"]
        state.pop("special_token_ablation", None)
        torch.save(state, legacy_path)
        code, events = self.run_cli("sft", source, config, legacy, "--resume", str(legacy_path))
        self.assertEqual(code, 0, events)


if __name__ == "__main__":
    unittest.main()
