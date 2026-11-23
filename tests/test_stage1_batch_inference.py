from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from tinychat.chat_format import SPECIALS, SPECIAL_IDS
from tinychat.evaluate_stage1 import main as evaluate
from tinychat.inference import generate, generate_batch
from tinychat.model import ModelConfig, TinyLM


def tokenizer_fixture():
    vocab = {**SPECIAL_IDS, "[UNK]": 7, "快": 8, "慢": 9, "答": 10, "再": 11}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(list(SPECIALS))
    return tokenizer


def single_turn(content):
    return [{"role": "user", "content": content}]


class ScriptedModel(torch.nn.Module):
    def __init__(self, vocab_size, context=64):
        super().__init__()
        self.device_anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_seq_len=context)
        self.tokenizer_sha256 = "fixture"
        self.vocab_size = vocab_size
        self.input_shapes = []

    def forward(self, inputs):
        self.input_shapes.append(tuple(inputs.shape))
        fast = inputs.eq(8).cumsum(dim=1).gt(0)
        following = torch.full_like(inputs, SPECIAL_IDS["<|pad|>"])
        following = torch.where(inputs.eq(5) & fast, 2, following)
        following = torch.where(inputs.eq(5) & ~fast, 10, following)
        following = torch.where(inputs.eq(10), 11, following)
        following = torch.where(inputs.eq(11), 6, following)
        logits = torch.zeros((*inputs.shape, self.vocab_size))
        return logits.scatter_(-1, following.unsqueeze(-1), 1.0)


class Stage1BatchInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tokenizer = tokenizer_fixture()

    def test_tiny_lm_matches_unpadded_greedy_for_different_lengths_and_turns(self):
        multi = [{"role": "user", "content": "慢"}, {"role": "assistant", "content": "答"},
                 {"role": "user", "content": "再 答"}]
        prompts = [single_turn("答"), multi, single_turn("答 慢 再")]
        config = ModelConfig(vocab_size=self.tokenizer.get_vocab_size(), n_layers=1, d_model=16,
                             n_heads=2, ffn_dim=32, max_seq_len=32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(83)
            model = TinyLM(config, activation_checkpointing=False).eval()
        sequential = [generate(model, self.tokenizer, conversation, 5) for conversation in prompts]
        batched = generate_batch(model, self.tokenizer, prompts, 5)
        self.assertGreater(len({result["input_tokens"] for result in batched}), 1)
        for actual, expected in zip(batched, sequential, strict=True):
            self.assertGreaterEqual(actual["elapsed_seconds"], 0.0)
            self.assertEqual({key: value for key, value in actual.items() if key != "elapsed_seconds"},
                             {key: value for key, value in expected.items() if key != "elapsed_seconds"})

    def test_early_eos_does_not_stop_right_padded_peer(self):
        model = ScriptedModel(self.tokenizer.get_vocab_size())
        results = generate_batch(model, self.tokenizer, [single_turn("慢"), single_turn("快 快 快")], 6)
        self.assertEqual([row["generated_ids"] for row in results], [[10, 11, 6], [2]])
        self.assertEqual([row["stop_reason"] for row in results], ["end_token", "end_token"])
        self.assertEqual([rows for rows, _ in model.input_shapes], [2, 1, 1])
        self.assertLessEqual(results[1]["elapsed_seconds"], results[0]["elapsed_seconds"])
        capped = generate_batch(model, self.tokenizer, [single_turn("慢")], 2)
        self.assertEqual((capped[0]["generated_ids"], capped[0]["stop_reason"]), ([10, 11], "max_new_tokens"))

    def test_whole_batch_fails_before_forward_when_one_prompt_exceeds_context(self):
        model = ScriptedModel(self.tokenizer.get_vocab_size(), context=13)
        prompts = [single_turn("答"), single_turn("慢 " * 20)]
        with self.assertRaisesRegex(ValueError, "Batch prompt 2.*exceeds context"):
            generate_batch(model, self.tokenizer, prompts, 2)
        self.assertEqual(model.input_shapes, [])

    def test_evaluate_records_order_batch_ids_and_wall_time(self):
        model = ScriptedModel(self.tokenizer.get_vocab_size())
        inputs = [{"id": f"dev-{index}", "split": "development", "category": "fixture",
                   "messages": single_turn(text)} for index, text in enumerate(("慢", "快 快 快", "慢"))]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.jsonl"
            output = root / "results.jsonl"
            content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in inputs).encode("utf-8")
            prompts.write_bytes(content)
            (root / "prompt_manifest.json").write_text(json.dumps({"frozen": True,
                "prompts_sha256": sha256(content).hexdigest()}), encoding="utf-8")
            args = ["evaluate_stage1", "--model", str(root), "--prompts", str(prompts), "--output", str(output),
                    "--label", "fixture", "--device", "cpu", "--batch-size", "2", "--max-new-tokens", "6"]
            with (
                patch("sys.argv", args),
                patch("tinychat.evaluate_stage1.load_model", return_value=(model, self.tokenizer)),
                redirect_stdout(StringIO()),
            ):
                evaluate()
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in inputs])
        self.assertEqual([row["batch_id"] for row in rows], [1, 1, 2])
        self.assertEqual({row["timing_scope"] for row in rows}, {"concurrent_batch_wall_until_row_finished"})
        self.assertEqual(summary["batch_size"], 2)
        self.assertGreaterEqual(summary["elapsed_seconds_total"], max(row["elapsed_seconds"] for row in rows))
        self.assertEqual(summary["metrics"]["all"]["prompts"], 3)


if __name__ == "__main__":
    unittest.main()
