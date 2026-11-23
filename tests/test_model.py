import unittest

import torch
import torch.nn.functional as F

from tinychat.model import ModelConfig, TinyLM


class ModelBehaviorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.config = ModelConfig(vocab_size=64, n_layers=2, d_model=32,
                                  n_heads=4, ffn_dim=64, max_seq_len=16)

    def test_future_tokens_cannot_change_prefix_logits(self):
        model = TinyLM(self.config, activation_checkpointing=False).eval()
        tokens = torch.randint(0, 64, (2, 16))
        changed = tokens.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 64
        with torch.no_grad():
            original = model(tokens)
            altered = model(changed)
        torch.testing.assert_close(original[:, :8], altered[:, :8], atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(original[:, 8:], altered[:, 8:]))

    def test_checkpoint_recomputation_preserves_gradients(self):
        direct = TinyLM(self.config, activation_checkpointing=False)
        recompute = TinyLM(self.config, activation_checkpointing=True)
        recompute.load_state_dict(direct.state_dict())
        tokens = torch.randint(0, 64, (2, 16))
        targets = torch.randint(0, 64, (2, 16))
        for model in (direct, recompute):
            F.cross_entropy(model(tokens).reshape(-1, 64), targets.reshape(-1)).backward()
        for (name, left), (_, right) in zip(direct.named_parameters(), recompute.named_parameters()):
            self.assertIsNotNone(left.grad, name)
            torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-4)
        self.assertNotIn("lm_head.weight", direct.state_dict())


if __name__ == "__main__":
    unittest.main()
