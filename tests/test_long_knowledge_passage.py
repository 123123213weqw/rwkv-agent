from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from bench.long_knowledge_passage import (
    GoldPassage,
    best_gold_overlap,
    build_page_passage_query,
    character_ngram_f1,
    character_ngram_recall,
    combine_passages,
    load_gold_passages,
    parse_positive_passage_qrels,
    reconstruct_final_page_order,
    select_passage_variants,
    summarize_passage_rows,
)
from scripts.prepare_miracl_passage_gold import build_filter_predicate


class FakeScorer:
    model_name = "fake"

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [float("answer" in document) for document in documents]


class LongKnowledgePassageTests(unittest.TestCase):
    def test_parses_only_positive_passage_qrels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qrels.tsv"
            path.write_text(
                "q1 Q0 10#2 1\nq1 Q0 10#3 0\nq2 Q0 11#0 2\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_positive_passage_qrels(str(path)),
                {"q1": {"10#2"}, "q2": {"11#0"}},
            )

    def test_filter_predicate_quotes_columns_and_strings(self) -> None:
        self.assertEqual(
            build_filter_predicate(["10#2", "O'Neil#1"]),
            '"docid"=\'10#2\' OR "docid"=\'O\'\'Neil#1\'',
        )

    def test_loads_gold_by_query_and_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "miracl-passage-gold.v1",
                        "language": "zh",
                        "docid": "10#2",
                        "page_id": "10",
                        "passage_id": "2",
                        "title": "标题",
                        "text": "答案正文",
                        "source_qids": ["q1", "q2"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_gold_passages(str(path))
            self.assertEqual(rows["q1"]["10"][0].docid, "10#2")
            self.assertEqual(rows["q2"]["10"][0].text, "答案正文")

    def test_reconstructs_cross_encoder_page_order(self) -> None:
        ordered = reconstruct_final_page_order(
            {
                "fused_hits": [
                    {"page_id": "a"},
                    {"page_id": "b"},
                    {"page_id": "c"},
                ],
                "rerank_scores": [0.1, 0.9],
            }
        )
        self.assertEqual([item["page_id"] for item in ordered], ["b", "a", "c"])

    def test_page_query_is_restricted_without_gold_information(self) -> None:
        query = build_page_passage_query(
            "白宫哪一年建造？",
            ["10", "20"],
            chunks_per_page=4,
        )
        serialized = json.dumps(query, ensure_ascii=False)
        self.assertIn('"page_id": ["10", "20"]', serialized)
        self.assertIn('"size": 4', serialized)
        self.assertNotIn("relevance", serialized)
        self.assertNotIn("qrel", serialized)

    def test_selects_lead_lexical_and_cross_encoder_independently(self) -> None:
        candidates = [
            {
                "doc_id": "p#0",
                "page_id": "p",
                "chunk_id": 0,
                "char_start": 0,
                "lexical_score": 0.1,
                "title": "Page",
                "text": "lead text",
            },
            {
                "doc_id": "p#1",
                "page_id": "p",
                "chunk_id": 1,
                "char_start": 500,
                "lexical_score": 4.0,
                "title": "Page",
                "text": "lexical passage",
            },
            {
                "doc_id": "p#2",
                "page_id": "p",
                "chunk_id": 2,
                "char_start": 900,
                "lexical_score": 2.0,
                "title": "Page",
                "text": "contains answer",
            },
        ]
        selected = select_passage_variants("question", candidates, FakeScorer())
        self.assertEqual(selected["lead"]["doc_id"], "p#0")
        self.assertEqual(selected["lexical"]["doc_id"], "p#1")
        self.assertEqual(selected["cross_encoder"]["doc_id"], "p#2")
        self.assertEqual(
            selected["lead_plus_cross"]["component_doc_ids"],
            ["p#0", "p#2"],
        )

    def test_combined_passage_is_bounded_and_keeps_both_sides(self) -> None:
        combined = combine_passages(
            {"doc_id": "p#0", "text": "L" * 3000, "char_start": 0},
            {"doc_id": "p#9", "text": "S" * 3000, "char_start": 9000},
            max_chars=1000,
        )
        self.assertEqual(len(combined["text"]), 1000)
        self.assertTrue(combined["text"].startswith("L" * 499))
        self.assertTrue(combined["text"].endswith("S" * 499))

    def test_character_overlap_matches_snapshot_variants(self) -> None:
        exact = character_ngram_f1(
            "德国军队突然入侵波兰，第二次世界大战全面爆发。",
            "德国军队突然入侵波蘭，第二次世界大战全面爆发。",
        )
        unrelated = character_ngram_f1(
            "德国军队突然入侵波兰，第二次世界大战全面爆发。",
            "丘吉尔于1951年再次担任首相。",
        )
        self.assertGreater(exact, 0.75)
        self.assertLess(unrelated, 0.1)
        self.assertGreater(
            character_ngram_recall(
                "prefix The answer is forty two. suffix",
                "The answer is forty two.",
            ),
            0.99,
        )

    def test_best_gold_overlap_uses_same_page_passages(self) -> None:
        gold = [
            GoldPassage(
                language="en",
                docid="1#2",
                page_id="1",
                passage_id="2",
                title="T",
                text="The answer is forty two.",
                source_qids=("q",),
            )
        ]
        self.assertGreater(
            best_gold_overlap({"text": "The answer is forty two."}, gold),
            0.99,
        )

    def test_summary_separates_end_to_end_and_conditional_metrics(self) -> None:
        selected = {
            strategy: [{"text": "evidence"}]
            for strategy in ("lead", "lexical", "cross_encoder", "lead_plus_cross")
        }
        metrics = {
            "lead": {
                "case_best_gold_overlap": 0.1,
                "case_best_gold_recall": 0.2,
                "relevant_selected_passages": [
                    {"page_id": "1", "gold_overlap": 0.1, "gold_recall": 0.2}
                ],
            },
            "lexical": {
                "case_best_gold_overlap": 0.4,
                "case_best_gold_recall": 0.5,
                "relevant_selected_passages": [
                    {"page_id": "1", "gold_overlap": 0.4, "gold_recall": 0.5}
                ],
            },
            "cross_encoder": {
                "case_best_gold_overlap": 0.8,
                "case_best_gold_recall": 0.9,
                "relevant_selected_passages": [
                    {"page_id": "1", "gold_overlap": 0.8, "gold_recall": 0.9}
                ],
            },
            "lead_plus_cross": {
                "case_best_gold_overlap": 0.7,
                "case_best_gold_recall": 1.0,
                "relevant_selected_passages": [
                    {"page_id": "1", "gold_overlap": 0.7, "gold_recall": 1.0}
                ],
            },
        }
        summary = summarize_passage_rows(
            [
                {
                    "page_hit_at_limit": 1.0,
                    "requested_pages": 8,
                    "returned_pages": 8,
                    "hydration_elapsed_ms": 20.0,
                    "selected_passages": selected,
                    "strategy_metrics": metrics,
                }
            ]
        )
        self.assertEqual(summary["page_hit_at_limit"], 1.0)
        self.assertEqual(
            summary["strategies"]["cross_encoder"]["case_hit_at_overlap_0_5"],
            1.0,
        )
        self.assertEqual(
            summary["strategies"]["lead"]["conditional_passage_hit_at_overlap_0_3"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
