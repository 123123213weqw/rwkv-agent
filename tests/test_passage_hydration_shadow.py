from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from rwkv_search.candidate_index import CandidateHit
from rwkv_search.config import ShadowSearchConfig
from rwkv_search.passage_hydration import (
    PagePassageClient,
    PassageHydrator,
    build_page_passage_query,
    combine_passages,
)
from rwkv_search.search import SearchResult
from rwkv_search.shadow_search import FineWikiShadowSearch


def candidate_hit(
    *,
    page_id: str = "42",
    doc_id: str = "finewiki:42:7",
    text: str = "旧的词法段落。",
) -> CandidateHit:
    return CandidateHit(
        doc_id=doc_id,
        page_id=page_id,
        title="RWKV",
        text=text,
        url=f"https://zh.wikipedia.org/wiki/{page_id}",
        page_type="article",
        score=0.064,
        channels=("exact", "word"),
        ranks={"exact": 1, "word": 2},
        source="finewiki",
        chunk_id=7,
        char_start=2400,
    )


class FakePassageClient(PagePassageClient):
    def __init__(self) -> None:
        self.calls = []

    def search_pages(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return {
            "42": [
                {
                    "doc_id": "finewiki:42:0",
                    "page_id": "42",
                    "chunk_id": 0,
                    "title": "RWKV",
                    "headings": ["导言"],
                    "text": "RWKV是一种结合RNN推理效率与Transformer训练方式的模型架构。",
                    "url": "https://zh.wikipedia.org/wiki/42",
                    "source": "finewiki",
                    "language": "zh",
                    "modified_at": "2026-06-01T00:00:00Z",
                    "char_start": 0,
                    "lexical_score": 1.0,
                },
                {
                    "doc_id": "finewiki:42:8",
                    "page_id": "42",
                    "chunk_id": 8,
                    "title": "RWKV",
                    "headings": ["架构"],
                    "text": "RWKV使用时间混合和通道混合模块，并以常量空间进行逐token推理。",
                    "url": "https://zh.wikipedia.org/wiki/42",
                    "source": "finewiki",
                    "language": "zh",
                    "modified_at": "2026-06-01T00:00:00Z",
                    "char_start": 2800,
                    "lexical_score": 5.0,
                },
            ]
        }


class FakeScorer:
    model_name = "fake-cross-encoder"

    def score(self, query, documents):
        return [1.0 if "通道混合" in document else 0.1 for document in documents]


class FakeCandidateClient:
    def search(self, query, **kwargs):
        analysis = SimpleNamespace(
            to_dict=lambda: {"original": query, "normalized": query.casefold()}
        )
        return analysis, [candidate_hit()], 12.5


class FailingHydrator:
    def hydrate(self, query, **kwargs):
        raise RuntimeError("synthetic passage failure")


def primary_result() -> SearchResult:
    return SearchResult(
        document_id=1,
        url="https://docs.example/rwkv",
        title="RWKV",
        snippet="primary",
        content="Primary evidence remains visible.",
        published_at=None,
        fetched_at=123.0,
        source_type="official_docs",
        authority=1.0,
        score=0.9,
        score_components={},
    )


class PassageHydrationShadowTests(unittest.TestCase):
    def test_page_query_is_restricted_and_keeps_twelve_candidates(self) -> None:
        payload = build_page_passage_query(
            "RWKV架构是什么",
            ["42", "42", "43"],
            chunks_per_page=12,
        )
        filters = payload["query"]["bool"]["filter"]
        self.assertEqual(filters, [{"terms": {"page_id": ["42", "43"]}}])
        inner = payload["collapse"]["inner_hits"]
        self.assertEqual(inner[0]["name"], "passage_candidates")
        self.assertEqual(inner[0]["size"], 12)
        self.assertEqual(inner[1]["name"], "page_lead")

    def test_combination_keeps_lead_and_selected_inside_budget(self) -> None:
        lead = {
            "doc_id": "42:0",
            "text": "A" * 3000,
            "char_start": 0,
            "headings": ["Lead"],
        }
        selected = {
            "doc_id": "42:8",
            "text": "B" * 3000,
            "char_start": 4000,
            "headings": ["Detail"],
        }
        combined = combine_passages(lead, selected, max_chars=3200)
        self.assertEqual(len(combined["text"]), 3200)
        self.assertEqual(combined["component_doc_ids"], ["42:0", "42:8"])
        self.assertEqual(combined["headings"], ["Lead", "Detail"])

    def test_hydrator_preserves_page_order_and_hydrates_only_head(self) -> None:
        client = FakePassageClient()
        hydrator = PassageHydrator(
            client,
            FakeScorer(),
            max_pages=1,
            chunks_per_page=12,
            max_chars=3200,
        )
        old = candidate_hit()
        tail = candidate_hit(page_id="99", doc_id="finewiki:99:1", text="tail")
        result = hydrator.hydrate(
            "RWKV怎么推理",
            index="finewiki-test",
            hits=[old, tail],
        )
        self.assertEqual([hit.page_id for hit in result.hits], ["42", "99"])
        self.assertIn("RWKV是一种", result.hits[0].text)
        self.assertIn("通道混合", result.hits[0].text)
        self.assertEqual(result.hits[0].hydration_strategy, "lead_plus_cross")
        self.assertEqual(
            result.hits[0].component_doc_ids,
            ("finewiki:42:0", "finewiki:42:8"),
        )
        self.assertEqual(result.hits[1], tail)
        self.assertEqual(result.stats["pages_requested"], 1)
        self.assertEqual(result.stats["pages_returned"], 1)
        self.assertEqual(result.stats["changed_evidence_count"], 1)
        self.assertEqual(client.calls[0][1]["chunks_per_page"], 12)

    def test_shadow_records_pair_but_live_results_remain_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "shadow.jsonl"
            hydrator = PassageHydrator(
                FakePassageClient(),
                FakeScorer(),
                max_pages=8,
                chunks_per_page=12,
            )
            runner = FineWikiShadowSearch(
                ShadowSearchConfig(
                    enabled=True,
                    log_path=str(log_path),
                    passage_hydration_enabled=True,
                ),
                client=FakeCandidateClient(),
                passage_hydrator=hydrator,
            )
            future = runner.start("RWKV怎么推理", {"intent": "search"})
            assert future is not None
            payload = future.result(timeout=2)
            self.assertEqual(payload["evidence_variant"], "lead_plus_cross")
            self.assertEqual(
                payload["legacy_evidence"][0]["text"],
                "旧的词法段落。",
            )
            self.assertIn("通道混合", payload["evidence"][0]["text"])
            self.assertEqual(
                payload["evidence"][0]["metadata"]["hydration_strategy"],
                "lead_plus_cross",
            )

            # Explicit FineWiki remains on the old CandidateHit until a later,
            # separately authorized production switch.
            live, _ = runner.live_results(future)
            self.assertEqual(live[0].content, "旧的词法段落。")

            runner.attach(
                future,
                primary_results=[primary_result()],
                visible_results=[primary_result()],
                primary_latency_ms=3.0,
            )
            deadline = time.time() + 2
            while not log_path.exists() and time.time() < deadline:
                time.sleep(0.01)
            record = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertFalse(record["visible_output_changed"])
            self.assertEqual(
                record["shadow"]["evidence_variant"],
                "lead_plus_cross",
            )
            self.assertEqual(
                record["evidence_pair_comparison"]["same_page_rate"],
                1.0,
            )
            self.assertEqual(
                record["evidence_pair_comparison"]["changed_text_count"],
                1,
            )
            runner.close()

    def test_hydration_failure_falls_back_inside_shadow(self) -> None:
        runner = FineWikiShadowSearch(
            ShadowSearchConfig(
                enabled=True,
                passage_hydration_enabled=True,
            ),
            client=FakeCandidateClient(),
            passage_hydrator=FailingHydrator(),
        )
        future = runner.start("RWKV", {"intent": "search"})
        assert future is not None
        payload = future.result(timeout=2)
        self.assertEqual(payload["evidence_variant"], "legacy")
        self.assertEqual(
            payload["passage_hydration"]["status"],
            "fallback_legacy",
        )
        self.assertEqual(payload["evidence"], payload["legacy_evidence"])
        runner.close()


if __name__ == "__main__":
    unittest.main()
