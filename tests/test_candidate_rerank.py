from __future__ import annotations

import unittest

from bench.candidate_rerank import (
    candidate_document,
    evaluate_ranking,
    public_case_matrix,
    rank_candidates,
    summarize_records,
)


class CandidateRerankTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = {
            "id": "rerank-001",
            "language": "en",
            "category": "software_release",
            "expected_domains_any": ["python.org"],
            "target_url_patterns_any": ["/downloads/"],
            "forbidden_result_types": [
                "search_homepage",
                "dictionary",
                "login_or_captcha",
                "error_page",
                "empty_content",
            ],
        }
        self.candidates = [
            {
                "url": "https://example.com/python-opinion",
                "title": "An opinion about Python",
                "snippet": "A personal blog post.",
                "engine": "bing",
                "rank": 1,
                "engine_score": 1.0,
            },
            {
                "url": "https://www.python.org/downloads/",
                "title": "Download Python",
                "snippet": "Download the latest stable release.",
                "engine": "bing",
                "rank": 2,
                "engine_score": 0.9,
            },
            {
                "url": "https://www.google.com/search?q=python",
                "title": "Google Search",
                "snippet": "Search results.",
                "engine": "bing",
                "rank": 3,
                "engine_score": 0.8,
            },
        ]

    def test_candidate_document_is_bounded_page_metadata(self) -> None:
        value = candidate_document(self.candidates[1], max_chars=128)
        self.assertIn("Title: Download Python", value)
        self.assertIn("Source: www.python.org/downloads/", value)
        self.assertIn("Summary:", value)
        self.assertNotIn("expected_domains_any", value)
        self.assertLessEqual(len(value), 128)

    def test_rank_ablation_keeps_raw_and_uses_semantic_score(self) -> None:
        ranked = rank_candidates(
            "Python latest stable release",
            self.candidates,
            [0.1, 4.2, -1.0],
            hybrid_semantic_weight=0.8,
        )
        self.assertEqual(ranked["raw"]["candidates"][0]["url"], self.candidates[0]["url"])
        self.assertEqual(
            ranked["semantic"]["candidates"][0]["url"], self.candidates[1]["url"]
        )
        self.assertEqual(
            ranked["hybrid"]["candidates"][0]["url"], self.candidates[1]["url"]
        )
        rejected = ranked["hybrid"]["rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertIn("search_homepage", rejected[0]["rejection_reasons"])

    def test_ranking_metrics_include_mrr_garbage_and_false_rejection(self) -> None:
        metrics = evaluate_ranking(self.case, self.candidates)
        self.assertTrue(metrics["candidate_domain_hit_at_5"])
        self.assertEqual(metrics["domain_rank"], 2)
        self.assertEqual(metrics["domain_mrr"], 0.5)
        self.assertEqual(metrics["target_rank"], 2)
        self.assertEqual(metrics["top8_garbage_count"], 1)
        rejected_metrics = evaluate_ranking(
            self.case,
            [self.candidates[0]],
            [self.candidates[1]],
        )
        self.assertEqual(rejected_metrics["rejected_expected_domain_count"], 1)
        self.assertEqual(rejected_metrics["rejected_useful_expected_domain_count"], 1)
        self.assertEqual(rejected_metrics["rejected_target_page_count"], 1)

    def test_latest_query_rejects_encyclopedia_page_shape(self) -> None:
        candidates = [
            {
                "url": "https://baike.baidu.com/item/Python/407313",
                "title": "Python（计算机编程语言）_百度百科",
                "snippet": "Python programming language",
                "engine": "bing",
                "rank": 1,
            },
            self.candidates[1],
        ]
        ranked = rank_candidates(
            "Python 官网 最新稳定版本", candidates, [2.0, 1.0]
        )
        self.assertEqual(len(ranked["hybrid"]["rejected"]), 1)
        self.assertIn(
            "dictionary",
            ranked["hybrid"]["rejected"][0]["rejection_reasons"],
        )

    def test_definition_query_keeps_encyclopedia_page_shape(self) -> None:
        candidate = {
            "url": "https://baike.baidu.com/item/Python/407313",
            "title": "Python（计算机编程语言）_百度百科",
            "snippet": "Python programming language",
            "engine": "bing",
            "rank": 1,
        }
        ranked = rank_candidates("Python 是什么意思", [candidate], [1.0])
        self.assertFalse(ranked["hybrid"]["rejected"])

    def test_rejected_login_on_expected_domain_is_not_counted_as_useful(self) -> None:
        candidate = {
            "url": "https://account.python.org/login",
            "title": "Sign in",
            "snippet": "Authentication required",
            "rejection_reasons": ["login_or_captcha"],
        }
        metrics = evaluate_ranking(self.case, [], [candidate])
        self.assertEqual(metrics["rejected_expected_domain_count"], 1)
        self.assertEqual(metrics["rejected_useful_expected_domain_count"], 0)

    def test_summary_and_public_matrix_do_not_publish_candidates(self) -> None:
        metrics = evaluate_ranking(self.case, self.candidates)
        rows = []
        for strategy in ("raw", "admission", "semantic", "hybrid"):
            rows.append(
                {
                    "id": self.case["id"],
                    "repetition": 1,
                    "language": "en",
                    "category": self.case["category"],
                    "strategy": strategy,
                    "rerank_elapsed_ms": 2.5,
                    "metrics": metrics,
                    "candidates": self.candidates,
                    "query": "private query",
                }
            )
        summary = summarize_records(rows)
        self.assertEqual(summary["strategies"]["raw"]["overall"]["records"], 1)
        self.assertEqual(summary["strategies"]["raw"]["overall"]["domain_mrr"], 0.5)
        public = public_case_matrix(rows)
        self.assertNotIn("candidates", public[0])
        self.assertNotIn("query", public[0])
        self.assertNotIn("url", str(public).casefold())

    def test_score_count_must_match_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "score count"):
            rank_candidates("query", self.candidates, [1.0])


if __name__ == "__main__":
    unittest.main()
