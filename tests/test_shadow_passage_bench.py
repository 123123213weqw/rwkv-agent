from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from bench.run_shadow_passage_bench import summarize


class ShadowPassageBenchTests(unittest.TestCase):
    def test_summary_preserves_paired_retrieval_semantics(self) -> None:
        rows = [
            {
                "page_order_identical": True,
                "legacy_hit_at_8": True,
                "hydrated_hit_at_8": True,
                "compared_count": 2,
                "changed_text_count": 2,
                "character_delta": 500,
                "legacy_page_ids": ["42"],
                "passage_hydration": {
                    "status": "ok",
                    "latency_ms": 300.0,
                    "query_latency_ms": 100.0,
                    "rerank_latency_ms": 200.0,
                },
            },
            {
                "page_order_identical": True,
                "legacy_hit_at_8": False,
                "hydrated_hit_at_8": False,
                "compared_count": 1,
                "changed_text_count": 0,
                "character_delta": 0,
                "legacy_page_ids": [],
                "passage_hydration": {
                    "status": "fallback_legacy",
                    "latency_ms": 0.0,
                    "query_latency_ms": 0.0,
                    "rerank_latency_ms": 0.0,
                },
            },
        ]
        summary = summarize(rows)
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["page_order_identity_rate"], 1.0)
        self.assertEqual(summary["legacy_hit_at_8"], 0.5)
        self.assertEqual(summary["hydrated_hit_at_8"], 0.5)
        self.assertEqual(summary["changed_evidence_rate"], 2 / 3)
        self.assertEqual(
            summary["hydration_statuses"],
            {"fallback_legacy": 1, "ok": 1},
        )
        self.assertEqual(summary["latency_ms"]["cold_start"], 300.0)
        self.assertEqual(summary["latency_ms"]["warm_mean"], 0.0)
        self.assertEqual(summary["empty_legacy_evidence_cases"], 1)
        self.assertFalse(summary["visible_output_changed"])
        self.assertFalse(summary["answer_generation_executed"])

    def test_empty_summary_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize([])

    def test_public_frozen_summary_is_hash_locked_and_safe(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "bench"
            / "baselines"
            / "long_knowledge"
            / "shadow-passage-v1"
        )
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema_version"],
            "shadow-passage-manifest.v1",
        )
        for item in manifest["files"]:
            path = root / item["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                item["sha256"],
            )
            value = path.read_text(encoding="utf-8")
            self.assertNotIn("/home/wzu/", value)
            self.assertNotIn("127.0.0.1:19220", value)
        comparison = json.loads(
            (root / "comparison.json").read_text(encoding="utf-8")
        )
        self.assertFalse(comparison["paired_result"]["visible_output_changed"])
        self.assertFalse(comparison["scope"]["answer_generation_executed"])


if __name__ == "__main__":
    unittest.main()
