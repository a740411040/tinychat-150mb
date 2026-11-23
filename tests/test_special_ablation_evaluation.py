import unittest
import torch

from scripts.evaluate_special_ablation import loss_masks, score_verified, read_jsonl, ROOT


class SpecialAblationEvaluationTests(unittest.TestCase):
    def test_content_and_stop_components_are_disjoint_and_shift_aligned(self):
        inputs = torch.tensor([[1, 4, 9, 6, 5, 10, 11, 6, 0]])
        labels = torch.tensor([[-100, -100, -100, -100, 10, 11, 6, 2, -100]])
        masks = loss_masks(inputs, labels)
        self.assertEqual(masks["content"].nonzero()[:, 1].tolist(), [4, 5])
        self.assertEqual(masks["first_content"].nonzero()[:, 1].tolist(), [4])
        self.assertEqual(masks["end_turn"].nonzero()[:, 1].tolist(), [6])
        self.assertEqual(masks["final_eos"].nonzero()[:, 1].tolist(), [7])
        self.assertEqual(int((masks["content"] | masks["end_turn"] | masks["final_eos"]).sum()), 4)

    def test_answer_recomputation_checks_actual_output_and_keeps_source(self):
        row = read_jsonl(ROOT / "data/sft_verified_v2/test.jsonl")[0]
        answer = row["messages"][-1]["content"]
        self.assertTrue(score_verified(row, answer))
        self.assertFalse(score_verified(row, ""))
        self.assertFalse(score_verified(row, "这是一段与题目无关的回答。"))
        self.assertTrue(score_verified(row, answer, [17, 21, 6]))
        self.assertFalse(score_verified(row, answer, [3, 17, 21, 6]))
        self.assertFalse(score_verified(row, answer, [17, 0, 21, 2]))
        self.assertEqual(row["messages"][-1]["content"], answer)


if __name__ == "__main__":
    unittest.main()
