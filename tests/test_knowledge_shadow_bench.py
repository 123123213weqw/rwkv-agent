from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from benchmarks.run_knowledge_hybrid_shadow import (
    _percentile,
    load_cases,
    summarize,
)


class KnowledgeShadowBenchTests(unittest.TestCase):
    def test_percentile_uses_nearest_rank(self) -> None:
        self.assertEqual(_percentile(range(1, 21), 0.95), 19.0)

    def test_frozen_case_set_is_hash_locked(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "data"
            / "knowledge_shadow_cases_v1.jsonl"
        )
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "b46e93f087308c4dce78c273636a58e959c81315482c977a04b7684604358270",
        )
        rows = load_cases(path)
        self.assertEqual(len(rows), 24)
        self.assertEqual({row["language"] for row in rows}, {"zh", "en"})

    def test_summary_reports_paired_wins_and_languages(self) -> None:
        rows = [
            {
                "language": "zh",
                "legacy": {
                    "status": "ok",
                    "page_ids": ["1"],
                    "hit_at_1": False,
                    "hit_at_5": False,
                    "latency_ms": 10,
                },
                "hybrid": {
                    "status": "ok",
                    "page_ids": ["2"],
                    "hit_at_1": True,
                    "hit_at_5": True,
                    "latency_ms": 20,
                    "hydrated_text_changed": True,
                },
            },
            {
                "language": "en",
                "legacy": {
                    "status": "ok",
                    "page_ids": ["3"],
                    "hit_at_1": True,
                    "hit_at_5": True,
                    "latency_ms": 11,
                },
                "hybrid": {
                    "status": "fallback_legacy",
                    "page_ids": ["3"],
                    "hit_at_1": True,
                    "hit_at_5": True,
                    "latency_ms": 21,
                    "hydrated_text_changed": False,
                },
            },
        ]
        value = summarize(rows)
        self.assertEqual(value["paired"]["hybrid_hit_at_1_wins"], 1)
        self.assertEqual(value["paired"]["hybrid_fallbacks"], 1)
        self.assertEqual(value["by_language"]["zh"]["hybrid"]["hit_at_1"], 1)
        self.assertEqual(value["by_language"]["en"]["legacy"]["hit_at_1"], 1)


if __name__ == "__main__":
    unittest.main()
