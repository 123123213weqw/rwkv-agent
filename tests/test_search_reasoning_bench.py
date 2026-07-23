from __future__ import annotations

import unittest

from bench.search_reasoning_bench import summarize_records


class SearchReasoningBenchTest(unittest.TestCase):
    def test_summary_compares_every_strategy_to_direct(self) -> None:
        records = []
        for strategy, hit, queries in (
            ("direct", False, 1),
            ("short_cot", True, 1),
            ("feedback", True, 2),
            ("react", True, 2),
        ):
            records.append(
                {
                    "strategy": strategy,
                    "language": "zh",
                    "category": "software_release",
                    "case_success": True,
                    "query_count": queries,
                    "model_call_count": queries,
                    "token_count": 10 * queries,
                    "model_elapsed_ms": 100.0,
                    "search_elapsed_ms": 50.0,
                    "total_elapsed_ms": 150.0,
                    "actions": [
                        {
                            "kind": "search",
                            "stop": "</tool_call>",
                            "reasoning": "brief" if strategy != "direct" else "",
                            "format_evaluation": {"strict_success": True},
                            "validation": {
                                "accepted": True,
                                "entity_retention_rate": 1.0,
                                "reasons": [],
                            },
                        }
                    ],
                    "metrics": {
                        "nonempty_candidates": True,
                        "candidate_count": 5,
                        "domain_hit_at_5": hit,
                        "domain_hit_at_10": hit,
                        "domain_hit_at_20": hit,
                        "target_page_hit_at_10": False,
                        "target_page_hit_at_20": False,
                    },
                }
            )
        summary = summarize_records(records)
        self.assertEqual(summary["overall"]["direct"]["domain_hit_at_10_rate"], 0.0)
        self.assertEqual(
            summary["deltas"]["feedback_vs_direct"]["domain_hit_at_10_rate"],
            1.0,
        )
        self.assertEqual(
            summary["overall"]["feedback"]["average_query_count"], 2.0
        )
        self.assertEqual(
            summary["overall"]["feedback"]["average_entity_retention_rate"],
            1.0,
        )
        self.assertEqual(summary["overall"]["react"]["subject_drift_rate"], 0.0)
        self.assertEqual(summary["overall"]["react"]["max_token_stop_rate"], 0.0)
        self.assertEqual(
            summary["overall"]["react"]["stop_reason_counts"], {"unknown": 1}
        )

    def test_summary_can_select_only_paired_direct_and_feedback(self) -> None:
        records = []
        for strategy, hit, queries in (
            ("direct", False, 1),
            ("feedback", True, 2),
        ):
            records.append(
                {
                    "strategy": strategy,
                    "case_success": True,
                    "query_count": queries,
                    "actions": [],
                    "metrics": {
                        "nonempty_candidates": True,
                        "domain_hit_at_10": hit,
                    },
                }
            )
        summary = summarize_records(
            records, strategies=("direct", "feedback")
        )
        self.assertEqual(set(summary["overall"]), {"direct", "feedback"})
        self.assertEqual(set(summary["deltas"]), {"feedback_vs_direct"})
        self.assertEqual(
            summary["deltas"]["feedback_vs_direct"]["domain_hit_at_10_rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
