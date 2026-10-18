from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = (
    Path(__file__).resolve().parents[1]
    / "bench"
    / "baselines"
    / "realtime_retrieval"
    / "agent-web-recall-5h-v1"
)


class AgentWebRecallBaselineTests(unittest.TestCase):
    def test_public_manifest_is_hash_locked_and_sanitized(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema_version"],
            "public-baseline-manifest.v1",
        )
        for item in manifest["files"]:
            path = ROOT / item["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                item["sha256"],
            )
            value = path.read_text(encoding="utf-8")
            for forbidden in (
                "/home/",
                "/Users/",
                "127.0.0.1",
                "100.70.",
                '"query":',
                '"url":',
            ):
                self.assertNotIn(forbidden, value)
        self.assertFalse(manifest["contains_full_urls"])
        self.assertFalse(manifest["contains_queries"])
        self.assertFalse(manifest["contains_page_content"])
        self.assertFalse(manifest["production_changed"])

    def test_repair_gate_has_exact_headline_metrics(self) -> None:
        value = json.loads(
            (ROOT / "comparison.json").read_text(encoding="utf-8")
        )
        decision = value["decision"]
        self.assertTrue(decision["enhanced_retrieval_gate_passed"])
        self.assertFalse(decision["enhanced_default_switch_passed"])
        self.assertFalse(decision["durable_searxng_pool_validated"])
        self.assertFalse(decision["production_enabled"])
        self.assertFalse(value["benchmark"]["answer_model_called"])

        legacy = value["arms"]["legacy"]
        enhanced = value["arms"]["enhanced"]
        self.assertEqual(legacy["candidate_domain_hit_at_10_rate"], 0.3)
        self.assertEqual(enhanced["candidate_domain_hit_at_10_rate"], 0.5)
        self.assertEqual(legacy["nonempty_result_rate"], 0.72)
        self.assertEqual(enhanced["nonempty_result_rate"], 0.88)
        self.assertEqual(legacy["garbage_result_rate"], 0.0915)
        self.assertEqual(enhanced["garbage_result_rate"], 0.0112)
        self.assertEqual(
            value["paired"]["candidate_domain_hit_at_10"],
            {"enhanced_wins": 10, "legacy_wins": 0},
        )
        self.assertEqual(
            value["paired"]["nonempty_result"],
            {"enhanced_wins": 8, "legacy_wins": 0},
        )


if __name__ == "__main__":
    unittest.main()
