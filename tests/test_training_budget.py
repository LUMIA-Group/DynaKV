import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainingBudgetTest(unittest.TestCase):
    def test_reference_batch_settings(self):
        llama = json.loads((ROOT / "configs/deepspeed_llama.json").read_text())
        qwen = json.loads((ROOT / "configs/deepspeed_qwen.json").read_text())
        self.assertEqual(llama["train_micro_batch_size_per_gpu"], 4)
        self.assertEqual(llama["gradient_accumulation_steps"], 8)
        self.assertEqual(qwen["train_micro_batch_size_per_gpu"], 1)
        self.assertEqual(qwen["gradient_accumulation_steps"], 16)

    def test_equal_documented_token_budgets(self):
        llama_tokens = 500 * 1 * 4 * 8 * 8192
        qwen_tokens = 250 * 1 * 1 * 16 * 32768
        self.assertEqual(llama_tokens, 131_072_000)
        self.assertEqual(qwen_tokens, llama_tokens)

    def test_reference_wrapper_defaults_to_one_gpu(self):
        script = (ROOT / "train_8B.sh").read_text()
        self.assertIn('NUM_GPUS="${NUM_GPUS:-1}"', script)

    def test_qwen_wrapper_keeps_original_schedule(self):
        script = (ROOT / "train_8B.sh").read_text()
        qwen_case = script.split("qwen)", 1)[1].split(";;", 1)[0]
        self.assertIn('SEQUENCE_LENGTH="32768"', qwen_case)
        self.assertIn('MAX_STEPS="250"', qwen_case)


if __name__ == "__main__":
    unittest.main()
