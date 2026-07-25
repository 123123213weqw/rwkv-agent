from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = (
    Path(__file__).resolve().parents[1]
    / "bench"
    / "baselines"
    / "long_knowledge"
    / "finewiki-hybrid-v1"
)


class LongKnowledgeHybridBaselineTests(unittest.TestCase):
    def test_public_files_are_hash_locked_and_safe(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                expected,
            )
        public_text = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in manifest["files"]
        )
        for forbidden in (
            "/home/",
            "/Users/",
            "127.0.0.1",
            "100.70.",
            '"query":',
            '"hits":',
        ):
            self.assertNotIn(forbidden, public_text)
        self.assertFalse(manifest["raw_traces_included"])
        self.assertFalse(manifest["model_weights_included"])
        self.assertFalse(manifest["production_changed"])

    def test_headline_records_large_bilingual_gain_without_threshold_switch(self) -> None:
        comparison = json.loads(
            (ROOT / "comparison.json").read_text(encoding="utf-8")
        )
        self.assertGreater(
            comparison["headline"]["miracl_zh_hit_at_10_gain"],
            0.20,
        )
        self.assertGreater(
            comparison["headline"]["miracl_en_hit_at_10_gain"],
            0.30,
        )
        threshold = json.loads(
            (ROOT / "threshold.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            threshold["status"],
            "offline_candidate_not_production_calibrated",
        )


if __name__ == "__main__":
    unittest.main()
