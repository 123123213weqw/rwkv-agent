from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.blend_fitgen_lora_to_raw import blend


class BlendFitGenLoRAToRawTests(unittest.TestCase):
    @staticmethod
    def make_adapter(root: Path, name: str, a, b) -> Path:
        import torch

        value = root / name
        value.mkdir()
        (value / "adapter_config.json").write_text(
            json.dumps({"r": 1, "lora_alpha": 2}),
            encoding="utf-8",
        )
        prefix = "base_model.model.model.layers.0.attn.k_proj"
        torch.save(
            {
                f"{prefix}.lora_A.weight": torch.tensor(a, dtype=torch.float32),
                f"{prefix}.lora_B.weight": torch.tensor(b, dtype=torch.float32),
            },
            value / "adapter_model.bin",
        )
        return value

    def test_blends_in_update_space_and_preserves_shared_storage(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.pth"
            backing = torch.zeros(8, dtype=torch.bfloat16)
            torch.save(
                {
                    "blocks.0.att.key.weight": backing[:4].view(2, 2),
                    "head.weight": backing[4:].view(2, 2),
                },
                raw,
            )
            reference = self.make_adapter(root, "reference", [[1.0, 1.0]], [[1.0], [1.0]])
            candidate = self.make_adapter(root, "candidate", [[2.0, 2.0]], [[2.0], [2.0]])
            output = root / "blended.pth"
            manifest = root / "manifest.json"
            value = blend(raw, reference, candidate, 0.25, output, manifest)
            merged = torch.load(output, map_location="cpu", weights_only=True)
            # scale=2: 0.75*(1@1)*2 + 0.25*(2@2)*2 = 3.5
            self.assertTrue(
                torch.equal(
                    merged["blocks.0.att.key.weight"].float(),
                    torch.full((2, 2), 3.5),
                )
            )
            self.assertEqual(value["candidate_weight"], 0.25)
            self.assertEqual(value["reference_weight"], 0.75)
            self.assertEqual(
                merged["blocks.0.att.key.weight"].untyped_storage().data_ptr(),
                merged["head.weight"].untyped_storage().data_ptr(),
            )

    def test_rejects_extrapolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                blend(root / "raw", root / "a", root / "b", 1.01, root / "o", root / "m")


if __name__ == "__main__":
    unittest.main()
