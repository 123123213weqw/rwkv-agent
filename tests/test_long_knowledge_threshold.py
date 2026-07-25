from __future__ import annotations

import unittest

from bench.analyze_long_knowledge_threshold import evaluate_thresholds


class LongKnowledgeThresholdTests(unittest.TestCase):
    def test_threshold_reports_missing_rejection_and_positive_loss_together(self) -> None:
        rows = [
            {
                "expectation": "relevant",
                "semantic_scores": [3.0],
                "strategies": {"semantic": {"hit_at_10": 1.0}},
            },
            {
                "expectation": "relevant",
                "semantic_scores": [-1.0],
                "strategies": {"semantic": {"hit_at_10": 1.0}},
            },
            {
                "expectation": "missing",
                "semantic_scores": [-2.0],
                "strategies": {"semantic": {"hit_at_10": None}},
            },
        ]
        result = evaluate_thresholds(rows, [0.0])
        threshold = result["thresholds"][0]
        self.assertEqual(threshold["positive_accept_rate"], 0.5)
        self.assertEqual(threshold["useful_hit_at_10"], 0.5)
        self.assertEqual(threshold["hit_at_10_loss"], 0.5)
        self.assertEqual(threshold["missing_rejection_rate"], 1.0)

    def test_score_field_is_configurable_for_fused_rerank_runs(self) -> None:
        rows = [
            {
                "expectation": "missing",
                "rerank_scores": [-1.0],
                "strategies": {
                    "lexical_dense_rerank": {"hit_at_10": None}
                },
            }
        ]
        result = evaluate_thresholds(
            rows,
            [0.0],
            strategy="lexical_dense_rerank",
            score_field="rerank_scores",
        )
        self.assertEqual(result["score_field"], "rerank_scores")
        self.assertEqual(
            result["thresholds"][0]["missing_rejection_rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
