from __future__ import annotations

import unittest

from bench.long_knowledge_schema import LongKnowledgeCase, RelevantPage
from bench.sweep_long_knowledge_rerank_rrf import sweep_weights


class LongKnowledgeRerankSweepTests(unittest.TestCase):
    def test_sweep_selects_weight_using_retrieval_metrics(self) -> None:
        case = LongKnowledgeCase(
            id="case-1",
            query="target",
            language="en",
            source_dataset="test",
            source_split="dev",
            source_qid="1",
            relevant_pages=(RelevantPage(page_id="target", relevance=1),),
            query_type="descriptive",
        )
        rows = [
            {
                "id": case.id,
                "expectation": case.expectation,
                "index_eligible": True,
                "semantic_scores": [0.0, 10.0],
                "hits": [{"page_id": "other"}, {"page_id": "target"}],
            }
        ]
        result = sweep_weights(rows, {case.id: case}, [0.0, 3.0])
        self.assertEqual(result["selected"]["semantic_weight"], 3.0)
        self.assertEqual(result["selected"]["hit_at_1"], 1.0)
        self.assertEqual(
            result["selection_scope"],
            "development_only_not_held_out",
        )


if __name__ == "__main__":
    unittest.main()
