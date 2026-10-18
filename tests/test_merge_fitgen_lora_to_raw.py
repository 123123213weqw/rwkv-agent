from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.merge_fitgen_lora_to_raw import merge, raw_key


class MergeFitGenLoRAToRawTests(unittest.TestCase):
    def test_translates_supported_hf_projection_names(self) -> None:
        prefix = "base_model.model.model.layers.7"
        self.assertEqual(
            raw_key(f"{prefix}.attn.r_proj.lora_A.weight"),
            ("blocks.7.att.receptance.weight", "A"),
        )
        self.assertEqual(
            raw_key(f"{prefix}.attn.o_proj.lora_B.weight"),
            ("blocks.7.att.output.weight", "B"),
        )
        self.assertEqual(
            raw_key(f"{prefix}.ffn.key.lora_A.weight"),
            ("blocks.7.ffn.key.weight", "A"),
        )

    def test_rejects_unknown_adapter_names(self) -> None:
        self.assertIsNone(raw_key("base_model.model.lm_head.lora_A.weight"))

    def test_merge_preserves_compact_shared_raw_storage(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.pth"
            backing = torch.arange(8, dtype=torch.bfloat16)
            torch.save(
                {
                    "blocks.0.att.key.weight": backing[:4].view(2, 2),
                    "head.weight": backing[4:].view(2, 2),
                },
                raw,
            )
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"r": 1, "lora_alpha": 1}),
                encoding="utf-8",
            )
            prefix = "base_model.model.model.layers.0.attn.k_proj"
            torch.save(
                {
                    f"{prefix}.lora_A.weight": torch.ones((1, 2)),
                    f"{prefix}.lora_B.weight": torch.ones((2, 1)),
                },
                adapter / "adapter_model.bin",
            )
            output = root / "merged.pth"
            manifest = root / "manifest.json"
            merge(raw, adapter, output, manifest)
            merged = torch.load(output, map_location="cpu", weights_only=True)
            self.assertTrue(
                torch.equal(
                    merged["blocks.0.att.key.weight"].float(),
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    merged["head.weight"].float(),
                    torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
                )
            )
            self.assertEqual(
                merged["blocks.0.att.key.weight"].untyped_storage().data_ptr(),
                merged["head.weight"].untyped_storage().data_ptr(),
            )
            self.assertLess(output.stat().st_size, raw.stat().st_size + 4096)


if __name__ == "__main__":
    unittest.main()
