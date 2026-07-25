from __future__ import annotations

import unittest

from bench.long_knowledge_hybrid import (
    candidate_document,
    evaluate_order,
    rank_candidates,
    reciprocal_rank_fusion,
    summarize_rows,
)
from bench.long_knowledge_schema import LongKnowledgeCase, RelevantPage


def relevant_case(*, language: str = "en") -> LongKnowledgeCase:
    return LongKnowledgeCase(
        id=f"case-{language}",
        query="language model with recurrent inference",
        language=language,
        source_dataset="test",
        source_split="dev",
        source_qid=f"qid-{language}",
        relevant_pages=(RelevantPage(page_id="target", relevance=1),),
        query_type="descriptive",
    )


class LongKnowledgeHybridTests(unittest.TestCase):
    def test_candidate_document_uses_title_sections_and_passage(self) -> None:
        text = candidate_document(
            {
                "title": " RWKV ",
                "headings": ["Architecture", "Inference"],
                "text": " recurrent   language model ",
            }
        )
        self.assertIn("Title: RWKV", text)
        self.assertIn("Sections: Architecture > Inference", text)
        self.assertIn("Passage: recurrent language model", text)

    def test_evaluate_order_separates_ranking_and_recall_misses(self) -> None:
        case = relevant_case()
        ranking = [{"page_id": str(index)} for index in range(11)]
        ranking.append({"page_id": "target"})
        ranked = evaluate_order(case, ranking, index_eligible=True)
        self.assertEqual(ranked["first_relevant_rank"], 12)
        self.assertEqual(ranked["failure_stage"], "ranking_miss")
        self.assertEqual(ranked["hit_at_10"], 0.0)
        self.assertEqual(ranked["hit_at_50"], 1.0)

        recalled = evaluate_order(
            case,
            [{"page_id": str(index)} for index in range(100)],
            index_eligible=True,
        )
        self.assertEqual(recalled["failure_stage"], "candidate_recall_miss")
        corpus = evaluate_order(case, [], index_eligible=False)
        self.assertEqual(corpus["failure_stage"], "corpus_miss")

    def test_rank_candidates_returns_semantic_and_rrf_orders(self) -> None:
        candidates = [
            {"page_id": "a"},
            {"page_id": "b"},
            {"page_id": "c"},
        ]
        ranked = rank_candidates(candidates, [0.1, 0.9, 0.2], rerank_depth=3)
        self.assertEqual(
            [item["page_id"] for item in ranked["semantic"]],
            ["b", "c", "a"],
        )
        self.assertEqual(
            {item["page_id"] for item in ranked["hybrid"]},
            {"a", "b", "c"},
        )
        self.assertEqual(
            [item["page_id"] for item in ranked["lexical"]],
            ["a", "b", "c"],
        )
        semantic_heavy = rank_candidates(
            candidates,
            [0.1, 0.9, 0.2],
            rerank_depth=3,
            semantic_weight=4.0,
        )
        self.assertEqual(semantic_heavy["hybrid"][0]["page_id"], "b")

    def test_summary_reports_top100_and_failure_stages(self) -> None:
        case = relevant_case()
        zh_case = relevant_case(language="zh")
        rows = []
        for item, candidates in (
            (case, [{"page_id": "x"}, {"page_id": "target"}]),
            (zh_case, [{"page_id": "x"}]),
        ):
            rows.append(
                {
                    "language": item.language,
                    "query_type": item.query_type,
                    "expectation": item.expectation,
                    "latency_ms": {"lexical": 10.0},
                    "strategies": {
                        "lexical": evaluate_order(
                            item,
                            candidates,
                            index_eligible=True,
                        )
                    },
                }
            )
        summary = summarize_rows(rows, strategies=("lexical",))
        overall = summary["strategies"]["lexical"]["overall"]
        self.assertEqual(overall["hit_at_100"], 0.5)
        self.assertEqual(overall["failure_stages"]["top10_hit"], 1)
        self.assertEqual(overall["failure_stages"]["candidate_recall_miss"], 1)
        self.assertEqual(
            set(summary["strategies"]["lexical"]["by_language"]),
            {"en", "zh"},
        )

    def test_reciprocal_rank_fusion_deduplicates_pages(self) -> None:
        result = reciprocal_rank_fusion(
            [
                [{"page_id": "a"}, {"page_id": "b"}],
                [{"page_id": "b"}, {"page_id": "c"}],
            ],
            weights=[1.0, 1.0],
        )
        self.assertEqual([item["page_id"] for item in result], ["b", "a", "c"])
        self.assertIn("fusion_score", result[0])


if __name__ == "__main__":
    unittest.main()
