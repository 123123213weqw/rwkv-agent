from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.train_fitgen_lora import (
    adapter_input_manifest,
    encode_example,
    load_records,
    stratified_eval_rows,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, value: str, *, add_special_tokens: bool):
        del add_special_tokens
        return {"input_ids": list(value.encode("utf-8"))}


class TrainFitGenLoRATests(unittest.TestCase):
    def test_response_survives_left_prompt_truncation(self) -> None:
        row = {"id": "x", "prompt": "p" * 100, "response": "ANSWER"}
        encoded = encode_example(FakeTokenizer(), row, max_length=20)
        supervised = [token for token, label in zip(encoded["input_ids"], encoded["labels"], strict=True) if label != -100]
        self.assertEqual(bytes(supervised).decode(), "ANSWER")
        self.assertEqual(len(encoded["input_ids"]), 20)

    def test_locked_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            locked = Path(temporary) / "locked" / "test.jsonl"
            locked.parent.mkdir()
            locked.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n")
            with self.assertRaises(ValueError):
                load_records(locked)

    def test_dev_selection_is_stratified_and_deterministic(self) -> None:
        rows = [
            {"id": f"a-{index}", "dataset": "a", "task": "answer"}
            for index in range(5)
        ] + [
            {"id": f"b-{index}", "dataset": "b", "task": "tool"}
            for index in range(5)
        ]
        selected = stratified_eval_rows(list(reversed(rows)), 4)
        self.assertEqual(
            [row["id"] for row in selected],
            ["a-0", "b-0", "a-1", "b-1"],
        )

    def test_adapter_continuation_input_is_validated_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "checkpoint"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"r": 8, "lora_alpha": 16})
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            manifest = adapter_input_manifest(adapter)
            self.assertEqual(manifest["r"], 8)
            self.assertEqual(manifest["lora_alpha"], 16)
            self.assertEqual(manifest["weights_file"], "adapter_model.safetensors")
            self.assertEqual(len(manifest["weights_sha256"]), 64)

    def test_adapter_continuation_rejects_incomplete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "checkpoint"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            with self.assertRaises(ValueError):
                adapter_input_manifest(adapter)


if __name__ == "__main__":
    unittest.main()
