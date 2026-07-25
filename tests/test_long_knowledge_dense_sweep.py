from __future__ import annotations

import unittest

from bench.long_knowledge_schema import LongKnowledgeCase, RelevantPage
from bench.sweep_long_knowledge_dense_rrf import sweep_dense_weights


class LongKnowledgeDenseSweepTests(unittest.TestCase):
    def test_dense_weight_sweep_can_promote_dense_only_result(self) -> None:
        case = LongKnowledgeCase(
            id="case-1",
            query="described entity",
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
                "index_eligible": True,
                "lexical_hits": [
                    {"page_id": "other"},
                    {"page_id": "target"},
                ],
                "dense_hits": [
                    {"page_id": "target"},
                    {"page_id": "other"},
                ],
            }
        ]
        result = sweep_dense_weights(rows, {case.id: case}, [0.5, 2.0])
        self.assertEqual(result["selected"]["dense_weight"], 2.0)
        self.assertEqual(result["selected"]["hit_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
