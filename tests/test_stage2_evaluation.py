import unittest

from tokenizers import Tokenizer

from scripts.evaluate_stage2 import LABELS, LEGACY_DATA, blind_review, natural_prompts, plain_prefixes
from tinychat.chat_format import SPECIAL_IDS, encode_conversation


class Stage2EvaluationTests(unittest.TestCase):
    def test_frozen_chat_splits_use_the_recorded_system_tokens(self):
        tokenizer = Tokenizer.from_file(str(LEGACY_DATA / "tokenizer.json"))
        for split, expected_count in (("dev", 100), ("test", 200)):
            prompts, source = natural_prompts(split)
            self.assertEqual(len(prompts), expected_count)
            system_prefix = [SPECIAL_IDS["<|bos|>"], SPECIAL_IDS["<|system|>"],
                             *tokenizer.encode(source["system"], add_special_tokens=False).ids,
                             SPECIAL_IDS["<|end_turn|>"]]
            for prompt in prompts:
                tokens = encode_conversation(tokenizer, prompt["messages"], generation_prompt=True)["input_ids"]
                self.assertEqual(tokens[:len(system_prefix)], system_prefix)
                self.assertEqual(len(tokens), prompt["prompt_tokens"])

    def test_blind_packet_round_trips_labels_and_keeps_machine_scores_private(self):
        semantic = [{"id": f"semantic-{index}", "category": "social",
                     "messages": [{"role": "user", "content": "你好"}],
                     "rubric": {"success": "回应问候", "failure": "忽略问候"}} for index in range(8)]
        strict = {"id": "strict", "category": "constraint", "machine_check": {"mode": "single_line", "value": "好"}}
        answers = {label: {prompt["id"]: f"第{index}种回应" for prompt in semantic}
                   for index, label in enumerate(LABELS, 1)}
        answers[LABELS[0]][strict["id"]] = "好"
        answers[LABELS[1]][strict["id"]] = "结果是好"

        packet, key = blind_review([*semantic, strict], answers)
        self.assertEqual({row["id"] for row in packet}, {prompt["id"] for prompt in semantic})
        self.assertEqual(key["strict"]["machine_success"], dict(zip(LABELS, (True, False), strict=True)))
        self.assertEqual({key[row["id"]]["answer_labels"]["A"] for row in packet}, set(LABELS))
        for row in packet:
            self.assertEqual(set(row), {"id", "messages", "rubric", "answers"})
            self.assertEqual(set(row["answers"]), {"A", "B"})
            for anonymous, text in row["answers"].items():
                self.assertEqual(text, answers[key[row["id"]]["answer_labels"][anonymous]][row["id"]])

    def test_plain_prefixes_keep_the_two_legacy_sources(self):
        prefixes = plain_prefixes()
        self.assertEqual(len({prefix for _, prefix in prefixes}), 12)
        self.assertEqual(sum(case.startswith("diagnosis_v1:") for case, _ in prefixes), 8)
        self.assertEqual(sum(case.startswith("diagnosis_v2:") for case, _ in prefixes), 4)


if __name__ == "__main__":
    unittest.main()
