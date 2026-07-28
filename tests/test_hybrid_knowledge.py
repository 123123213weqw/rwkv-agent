from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rwkv_search.candidate_index import CandidateHit

from rwkv_agent.tools.hybrid_knowledge import (
    HybridKnowledgeRetriever,
    HybridKnowledgeShadow,
    HybridSearchResult,
    hydrate_pages,
    reciprocal_rank_fusion,
    rerank_candidates,
)


def hit(page_id: str, title: str, text: str) -> CandidateHit:
    return CandidateHit(
        doc_id=f"{page_id}#0",
        page_id=page_id,
        title=title,
        text=text,
        url=f"https://example.invalid/{page_id}",
        page_type="article",
        score=1.0,
        channels=("words",),
        ranks={"words": 1},
        language="en",
        chunk_id=0,
    )


class FakeEncoder:
    model_name = "fake-e5"

    def encode_queries(self, queries):
        return [[1.0, 0.0] for _ in queries]


class KeywordScorer:
    model_name = "fake-cross"

    def score(self, query, documents):
        del query
        return [
            10.0 if "target" in document.casefold() else float(-position)
            for position, document in enumerate(documents)
        ]


class FakeLexical:
    def search(self, query, **kwargs):
        del query, kwargs
        return object(), [
            hit("1", "Other", "other lead"),
            hit("2", "Target", "target lead"),
        ], 1.25


class FakeDense:
    def search(self, index, vector, **kwargs):
        del index, vector, kwargs
        return [
            {
                "doc_id": "2",
                "page_id": "2",
                "title": "Target",
                "text": "target dense",
            },
            {
                "doc_id": "3",
                "page_id": "3",
                "title": "Third",
                "text": "third",
            },
        ], 0.75


class FakePassages:
    def search_pages(self, query, *, page_ids, **kwargs):
        del query, kwargs
        return {
            page_id: [
                {
                    "doc_id": f"{page_id}#0",
                    "page_id": page_id,
                    "title": f"Page {page_id}",
                    "text": f"lead {page_id}",
                    "char_start": 0,
                    "chunk_id": 0,
                },
                {
                    "doc_id": f"{page_id}#2",
                    "page_id": page_id,
                    "title": f"Page {page_id}",
                    "text": f"target detail {page_id}",
                    "char_start": 200,
                    "chunk_id": 2,
                },
            ]
            for page_id in page_ids
        }


class EmptyLexical:
    def search(self, query, **kwargs):
        del query, kwargs
        return object(), [], 0.1


class EmptyDense:
    def search(self, index, vector, **kwargs):
        del index, vector, kwargs
        return [], 0.1


class NeverPassages:
    def search_pages(self, *args, **kwargs):
        raise AssertionError("empty retrieval must not query passages")


class FailingRetriever:
    def search(self, query, *, language):
        del query, language
        raise RuntimeError("boom")


class HybridKnowledgeTests(unittest.TestCase):
    def test_rrf_deduplicates_pages_and_rewards_overlap(self) -> None:
        value = reciprocal_rank_fusion(
            (
                [{"page_id": "1"}, {"page_id": "2"}],
                [{"page_id": "2"}, {"page_id": "3"}],
            )
        )
        self.assertEqual([item["page_id"] for item in value], ["2", "1", "3"])

    def test_rerank_attaches_scores_and_keeps_tail(self) -> None:
        value, _ = rerank_candidates(
            "query",
            [
                {"page_id": "1", "title": "Other"},
                {"page_id": "2", "title": "Target"},
                {"page_id": "3", "title": "Tail"},
            ],
            KeywordScorer(),
            depth=2,
        )
        self.assertEqual([item["page_id"] for item in value], ["2", "1", "3"])
        self.assertEqual(value[0]["rerank_score"], 10.0)
        self.assertNotIn("rerank_score", value[2])

    def test_hydration_keeps_page_order_and_adds_lead_plus_selected(self) -> None:
        pages = [
            {"page_id": "2", "text": "old 2"},
            {"page_id": "1", "text": "old 1"},
        ]
        hydrated, stats = hydrate_pages(
            "query",
            pages,
            FakePassages().search_pages(
                "query",
                page_ids=["2", "1"],
            ),
            KeywordScorer(),
            max_chars=512,
        )
        self.assertEqual([item["page_id"] for item in hydrated], ["2", "1"])
        self.assertIn("lead 2", hydrated[0]["text"])
        self.assertIn("target detail 2", hydrated[0]["text"])
        self.assertEqual(stats["changed_pages"], 2)

    def test_full_retriever_reranks_and_hydrates(self) -> None:
        retriever = HybridKnowledgeRetriever(
            "http://example.invalid",
            encoder=FakeEncoder(),
            scorer=KeywordScorer(),
            lexical_client=FakeLexical(),
            dense_client=FakeDense(),
            passage_client=FakePassages(),
            candidate_limit=5,
            rerank_depth=5,
            result_limit=2,
        )
        result = retriever.search("find target", language="en")
        self.assertEqual(result.hits[0]["page_id"], "2")
        self.assertEqual(len(result.hits), 2)
        self.assertEqual(result.stats["status"], "ok")
        self.assertEqual(
            result.stats["hydration"]["strategy"],
            "lead_plus_cross",
        )

    def test_empty_retrieval_does_not_call_passage_index(self) -> None:
        retriever = HybridKnowledgeRetriever(
            "http://example.invalid",
            encoder=FakeEncoder(),
            scorer=KeywordScorer(),
            lexical_client=EmptyLexical(),
            dense_client=EmptyDense(),
            passage_client=NeverPassages(),
            candidate_limit=5,
        )
        result = retriever.search("missing", language="en")
        self.assertEqual(result.hits, ())
        self.assertEqual(result.stats["hydration"]["status"], "empty")

    def test_shadow_failure_is_logged_as_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.jsonl"
            shadow = HybridKnowledgeShadow(
                FailingRetriever(),
                log_path=str(path),
            )
            value = shadow.compare(
                "query",
                language="en",
                legacy_evidence=[{"page_id": "42"}],
            )
            shadow.close()
            self.assertEqual(value["status"], "fallback_legacy")
            self.assertEqual(value["legacy_page_ids"], ["42"])
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["status"], "fallback_legacy")

    def test_shadow_queue_is_bounded(self) -> None:
        class BlockingRetriever:
            def __init__(self):
                import threading

                self.release = threading.Event()

            def search(self, query, *, language):
                del query, language
                self.release.wait(timeout=2)
                return HybridSearchResult(hits=(), stats={})

        retriever = BlockingRetriever()
        shadow = HybridKnowledgeShadow(retriever, max_pending=1)
        first = shadow.submit("one", language="en", legacy_evidence=[])
        second = shadow.submit("two", language="en", legacy_evidence=[])
        self.assertTrue(first["submitted"])
        self.assertFalse(second["submitted"])
        self.assertEqual(second["reason"], "queue_full")
        retriever.release.set()
        shadow.close()


if __name__ == "__main__":
    unittest.main()
