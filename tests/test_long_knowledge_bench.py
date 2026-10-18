from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from bench.long_knowledge_metrics import aggregate_scores, score_case
from bench.long_knowledge_schema import (
    LongKnowledgeCase,
    RelevantPage,
    load_cases,
    write_cases,
)


def case() -> LongKnowledgeCase:
    return LongKnowledgeCase(
        id="miracl-en-dev-1",
        query="Who designed Python?",
        language="en",
        source_dataset="MIRACL",
        source_split="dev",
        source_qid="1",
        relevant_pages=(
            RelevantPage(page_id="10", relevance=2),
            RelevantPage(page_id="20", relevance=1),
        ),
    )


class LongKnowledgeSchemaTests(unittest.TestCase):
    def test_round_trip_and_duplicate_id_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.jsonl"
            write_cases(path, [case()])
            self.assertEqual(load_cases(path), [case()])
            path.write_text(path.read_text() * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate case id"):
                load_cases(path)

    def test_rejects_non_positive_or_missing_qrels(self) -> None:
        payload = case().to_dict()
        payload["relevant_pages"] = [{"page_id": "10", "relevance": 0}]
        with self.assertRaisesRegex(ValueError, "positive"):
            LongKnowledgeCase.from_dict(payload)
        payload["relevant_pages"] = []
        with self.assertRaisesRegex(ValueError, "at least one"):
            LongKnowledgeCase.from_dict(payload)

    def test_explicit_missing_case_has_no_qrels(self) -> None:
        payload = case().to_dict()
        payload["expectation"] = "missing"
        payload["relevant_pages"] = []
        missing = LongKnowledgeCase.from_dict(payload)
        self.assertEqual(missing.expectation, "missing")
        with self.assertRaisesRegex(ValueError, "must not contain"):
            LongKnowledgeCase.from_dict(
                {**case().to_dict(), "expectation": "missing"}
            )


class LongKnowledgeMetricTests(unittest.TestCase):
    def test_page_level_recall_mrr_and_ndcg(self) -> None:
        row = score_case(case(), ["999", "20", "10", "10"], latency_ms=12.5)
        self.assertEqual(row["retrieved_page_ids"], ["999", "20", "10"])
        self.assertEqual(row["hit_at_1"], 0.0)
        self.assertEqual(row["hit_at_5"], 1.0)
        self.assertEqual(row["recall_at_1"], 0.0)
        self.assertEqual(row["recall_at_5"], 1.0)
        self.assertEqual(row["mrr_at_10"], 0.5)
        dcg = 1 / math.log2(3) + 3 / math.log2(4)
        idcg = 3 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(row["ndcg_at_10"], dcg / idcg)

    def test_aggregate_keeps_language_breakdown_and_latency(self) -> None:
        first = score_case(case(), ["10"], latency_ms=10)
        second_case = LongKnowledgeCase(
            **{**case().__dict__, "id": "miracl-zh-dev-2", "language": "zh", "source_qid": "2"}
        )
        second = score_case(second_case, [], latency_ms=30)
        first["index_eligible"] = True
        second["index_eligible"] = False
        summary = aggregate_scores([first, second])
        self.assertEqual(summary["overall"]["cases"], 2)
        self.assertEqual(summary["overall"]["empty_rate"], 0.5)
        self.assertEqual(summary["overall"]["latency_ms"]["mean"], 20)
        self.assertEqual(set(summary["by_language"]), {"en", "zh"})
        self.assertEqual(set(summary["by_query_type"]), {"unspecified"})
        self.assertEqual(summary["conditional_on_index_coverage"]["cases"], 1)

    def test_expected_missing_is_reported_outside_retrieval_metrics(self) -> None:
        missing_case = LongKnowledgeCase(
            id="compat-en-missing-1",
            query="What happened tomorrow?",
            language="en",
            source_dataset="rwkv-search-compat",
            source_split="v1",
            source_qid="missing-1",
            relevant_pages=(),
            expectation="missing",
        )
        row = score_case(missing_case, [], latency_ms=4)
        self.assertEqual(row["missing_correct"], 1.0)
        self.assertIsNone(row["hit_at_10"])
        summary = aggregate_scores([score_case(case(), ["10"], latency_ms=2), row])
        self.assertEqual(summary["cases_total"], 2)
        self.assertEqual(summary["positive_cases"], 1)
        self.assertEqual(summary["expected_missing_cases"], 1)
        self.assertEqual(summary["expected_missing_accuracy"], 1.0)
        self.assertEqual(summary["overall"]["cases"], 1)


if __name__ == "__main__":
    unittest.main()
