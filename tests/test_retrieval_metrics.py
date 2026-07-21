from __future__ import annotations

import unittest

from bench.retrieval_metrics import (
    aggregate,
    classify_garbage_types,
    domain_matches,
    evaluate_candidate_stage,
    evaluate_case,
    normalized_domain,
    organization_domain,
    organization_domain_matches,
    target_page_matches,
)


class RetrievalMetricsTest(unittest.TestCase):
    def test_candidate_stage_metrics_keep_initial_and_refined_recall_separate(
        self,
    ) -> None:
        case = {
            "expected_domains_any": ["python.org"],
            "target_url_patterns_any": ["/downloads/"],
        }
        initial = [{"url": "https://python.org/", "title": "Python"}]
        refined = [{"url": "https://python.org/downloads/", "title": "Downloads"}]
        self.assertFalse(
            evaluate_candidate_stage(case, initial)["candidate_target_page_hit_at_10"]
        )
        self.assertTrue(
            evaluate_candidate_stage(case, refined)["candidate_target_page_hit_at_10"]
        )

    def setUp(self) -> None:
        self.case = {
            "expected_domains_any": ["python.org"],
            "target_url_patterns_any": ["/downloads/"],
            "forbidden_result_types": [
                "search_homepage",
                "dictionary",
                "error_page",
                "login_or_captcha",
                "empty_content",
            ],
        }

    def test_domain_normalization_and_subdomain_match(self) -> None:
        self.assertEqual(
            normalized_domain("https://www.PYTHON.org/downloads/"), "python.org"
        )
        self.assertTrue(domain_matches("docs.python.org", "python.org"))
        self.assertFalse(domain_matches("python.org.example.com", "python.org"))

    def test_target_page_requires_expected_domain_and_path(self) -> None:
        self.assertTrue(
            target_page_matches(
                "https://www.python.org/downloads/release/python-3140/",
                ["python.org"],
                ["/downloads/"],
            )
        )
        self.assertFalse(
            target_page_matches(
                "https://example.com/downloads/python/", ["python.org"], ["/downloads/"]
            )
        )
        self.assertFalse(
            target_page_matches(
                "https://python.org/about/", ["python.org"], ["/downloads/"]
            )
        )

    def test_target_page_normalizes_trailing_slash_without_prefix_false_positive(
        self,
    ) -> None:
        self.assertTrue(
            target_page_matches(
                "https://python.org/downloads", ["python.org"], ["/downloads/"]
            )
        )
        self.assertFalse(
            target_page_matches(
                "https://python.org/downloads-malware",
                ["python.org"],
                ["/downloads/"],
            )
        )

    def test_organization_parent_is_partial_not_strict_domain_credit(self) -> None:
        self.assertFalse(domain_matches("usgs.gov", "earthquake.usgs.gov"))
        self.assertEqual(organization_domain("earthquake.usgs.gov"), "usgs.gov")
        self.assertTrue(organization_domain_matches("usgs.gov", "earthquake.usgs.gov"))
        metrics = evaluate_case(
            {
                **self.case,
                "expected_domains_any": ["earthquake.usgs.gov"],
                "target_url_patterns_any": ["/earthquakes/map/"],
            },
            [{"url": "https://usgs.gov/programs/earthquake-hazards"}],
            [],
        )
        self.assertFalse(metrics["candidate_domain_hit_at_10"])
        self.assertTrue(metrics["candidate_organization_domain_hit_at_10"])
        self.assertFalse(metrics["candidate_target_page_hit_at_10"])

    def test_garbage_types_are_detected_independently(self) -> None:
        self.assertIn(
            "search_homepage",
            classify_garbage_types(
                {"url": "https://www.bing.com/search?q=python", "title": "Search"}
            ),
        )
        self.assertIn(
            "dictionary",
            classify_garbage_types(
                {"url": "https://www.iciba.com/word?w=python", "title": "金山词霸"}
            ),
        )
        self.assertIn(
            "error_page",
            classify_garbage_types(
                {"url": "https://example.com/missing", "title": "404 Not Found"}
            ),
        )
        self.assertIn(
            "login_or_captcha",
            classify_garbage_types(
                {"url": "https://example.com/captcha", "title": "Verify you are human"}
            ),
        )
        self.assertIn(
            "empty_content",
            classify_garbage_types(
                {
                    "url": "https://example.com/empty",
                    "title": "Empty",
                    "content_length": 0,
                }
            ),
        )

    def test_case_metrics_separate_discovery_results_and_fetches(self) -> None:
        candidates = [
            {"url": "https://blog.example/python"},
            {"url": "https://www.python.org/downloads/"},
        ]
        results = [
            {
                "url": "https://www.python.org/downloads/",
                "title": "Download Python",
                "snippet": "Current stable release",
                "content_length": 1200,
            },
            {
                "url": "https://www.bing.com/search?q=python",
                "title": "Search",
                "content_length": 300,
            },
        ]
        metrics = evaluate_case(
            self.case,
            candidates,
            results,
            {"attempted": 4, "fetched": 2, "failed": 1, "cancelled": 1},
        )
        self.assertTrue(metrics["candidate_domain_hit_at_5"])
        self.assertTrue(metrics["candidate_target_page_hit_at_10"])
        self.assertTrue(metrics["result_domain_hit_at_5"])
        self.assertEqual(metrics["garbage_result_count"], 1)
        self.assertEqual(metrics["garbage_result_rate"], 0.5)
        self.assertEqual(metrics["fetch_success_rate"], 0.5)

    def test_aggregate_has_p95_fetch_totals_and_all_required_groups(self) -> None:
        records = []
        for index, elapsed in enumerate((10.0, 20.0, 30.0, 40.0), 1):
            metrics = evaluate_case(
                self.case,
                [{"url": "https://python.org/downloads/"}],
                [{"url": "https://python.org/downloads/", "content_length": 10}],
                {"attempted": 2, "fetched": 1, "failed": 1},
            )
            records.append(
                {
                    "id": str(index),
                    "language": "zh" if index < 3 else "en",
                    "category": "software_release",
                    "source_policy": "official_required",
                    "metrics": metrics,
                    "total_elapsed_ms": elapsed,
                }
            )
        summary = aggregate(records)
        overall = summary["overall"]
        self.assertEqual(overall["p95_total_elapsed_ms"], 40.0)
        self.assertEqual(overall["fetch_attempted"], 8)
        self.assertEqual(overall["fetch_success_rate"], 0.5)
        self.assertIn("language:zh", summary["groups"])
        self.assertIn("category:software_release", summary["groups"])
        self.assertIn("source_policy:official_required", summary["groups"])


if __name__ == "__main__":
    unittest.main()
