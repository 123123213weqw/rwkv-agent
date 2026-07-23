from __future__ import annotations

import unittest

from bench.searxng_engine_bench import (
    enabled_search_engines,
    engine_is_unresponsive,
    evaluate_engine_record,
    public_case_matrix,
    rank_general_engines,
    summarize_records,
)
from bench.run_searxng_engine_bench import serialize_candidate
from rwkv_search.realtime.types import DiscoveredURL


CASE = {
    "id": "retrieval-en-001",
    "language": "en",
    "category": "software_release",
    "expected_domains_any": ["python.org"],
    "target_url_patterns_any": ["/downloads/"],
    "forbidden_result_types": ["search_homepage", "empty_content"],
}


class SearxngEngineBenchTests(unittest.TestCase):
    def test_discovered_url_serialization_preserves_fields(self) -> None:
        value = DiscoveredURL(
            url="https://python.org/downloads/",
            title="Python",
            snippet="Release",
            engine="mwmbl",
        )
        item = serialize_candidate(value, 1)
        self.assertEqual("https://python.org/downloads/", item["url"])
        self.assertEqual("Python", item["title"])
        self.assertEqual("mwmbl", item["engine"])

    def test_enabled_engine_inventory(self) -> None:
        config = {
            "engines": [
                {"name": "good", "enabled": True, "categories": ["general"]},
                {"name": "off", "enabled": False, "categories": ["general"]},
            ]
        }
        self.assertEqual(["good"], [row["name"] for row in enabled_search_engines(config)])

    def test_unresponsive_shapes(self) -> None:
        self.assertTrue(engine_is_unresponsive("mwmbl", [["mwmbl", "timeout"]]))
        self.assertTrue(engine_is_unresponsive("mwmbl", [{"engine": "mwmbl"}]))
        self.assertFalse(engine_is_unresponsive("mwmbl", [["360search", "timeout"]]))

    def test_record_metrics_cover_recall_garbage_and_diversity(self) -> None:
        candidates = [
            {
                "url": "https://www.python.org/downloads/",
                "title": "Download Python",
                "snippet": "Current stable release",
            },
            {"url": "https://www.google.com/search?q=python", "title": "Search"},
        ]
        metrics = evaluate_engine_record(CASE, candidates, request_success=True)
        self.assertTrue(metrics["candidate_domain_hit_at_10"])
        self.assertTrue(metrics["candidate_target_page_hit_at_20"])
        self.assertEqual(1, metrics["garbage_result_count"])
        self.assertEqual(1.0, metrics["unique_domain_ratio"])

    def test_stability_requires_hit_in_every_repetition(self) -> None:
        records = []
        for repetition, hit in ((1, True), (2, False)):
            candidates = (
                [{"url": "https://python.org/downloads/", "title": "Python"}]
                if hit
                else [{"url": "https://example.com/x", "title": "Other"}]
            )
            records.append(
                {
                    "id": CASE["id"],
                    "engine": "alpha",
                    "repetition": repetition,
                    "language": "en",
                    "category": CASE["category"],
                    "elapsed_ms": 10.0,
                    "metrics": evaluate_engine_record(
                        CASE, candidates, request_success=True
                    ),
                }
            )
        summary = summarize_records(
            records, [{"name": "alpha", "categories": ["general"]}]
        )
        overall = summary["engines"]["alpha"]["overall"]
        self.assertEqual(0.5, overall["domain_recall_at_10"])
        self.assertEqual(0.0, overall["stable_domain_recall_at_10"])
        self.assertEqual(1, overall["intermittent_domain_hit_cases"])

    def test_general_ranking_prefers_stable_recall(self) -> None:
        summary = {
            "engines": {
                "stable": {
                    "spec": {"categories": ["general"]},
                    "languages": {
                        "en": {
                            "stable_domain_recall_at_10": 0.6,
                            "stable_target_page_recall_at_20": 0.1,
                            "stable_nonempty_rate": 1.0,
                            "request_success_rate": 1.0,
                        }
                    },
                },
                "lucky": {
                    "spec": {"categories": ["general"]},
                    "languages": {
                        "en": {
                            "stable_domain_recall_at_10": 0.2,
                            "stable_target_page_recall_at_20": 0.5,
                            "stable_nonempty_rate": 1.0,
                            "request_success_rate": 1.0,
                        }
                    },
                },
                "github": {
                    "spec": {"categories": ["repos"]},
                    "languages": {"en": {}},
                },
            }
        }
        self.assertEqual(["stable", "lucky"], rank_general_engines(summary, "en"))

    def test_public_matrix_does_not_include_urls_or_queries(self) -> None:
        rows = [
            {
                "id": CASE["id"],
                "engine": "alpha",
                "repetition": 1,
                "language": "en",
                "category": CASE["category"],
                "request": {"query": "secret"},
                "candidates": [{"url": "https://example.com"}],
                "elapsed_ms": 1.0,
                "metrics": evaluate_engine_record(CASE, [], request_success=False),
            }
        ]
        matrix = public_case_matrix(rows)
        self.assertNotIn("request", matrix[0])
        self.assertNotIn("candidates", matrix[0])


if __name__ == "__main__":
    unittest.main()
