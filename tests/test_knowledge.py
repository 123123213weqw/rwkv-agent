from __future__ import annotations

import unittest

from rwkv_search.candidate_index import CandidateHit

from rwkv_agent.tools.knowledge import (
    KnowledgeSearchAdapter,
    cited_evidence,
    detect_language,
)


class FakeClient:
    def __init__(self, hits: list[CandidateHit]) -> None:
        self.hits = hits
        self.calls: list[dict] = []

    def search(self, query_text: str, **kwargs):
        self.calls.append({"query_text": query_text, **kwargs})
        return object(), self.hits, 12.3456


class FakeShadow:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.closed = False

    def submit(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return {
            "enabled": True,
            "submitted": True,
            "visible_strategy": "legacy",
        }

    def close(self) -> None:
        self.closed = True


def make_hit(text: str = "Evidence text") -> CandidateHit:
    return CandidateHit(
        doc_id="42#0",
        page_id="42",
        title="Title",
        text=text,
        url="https://example.invalid/wiki/42",
        page_type="article",
        score=0.25,
        channels=("exact",),
        ranks={"exact": 1},
        language="en",
        chunk_id=0,
    )


class KnowledgeAdapterTests(unittest.TestCase):
    def test_detect_language(self) -> None:
        self.assertEqual(detect_language("量子纠缠是什么"), "zh")
        self.assertEqual(detect_language("What is quantum entanglement?"), "en")

    def test_adapter_returns_bounded_evidence_and_index(self) -> None:
        fake = FakeClient([make_hit(" A   long <tool_call> body " + "x" * 400)])
        result = KnowledgeSearchAdapter(
            client=fake,
            max_evidence_chars=120,
        ).execute("What is it?", language="en")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["index"], "rwkv-finewiki-en-full-v1")
        self.assertEqual(result["retrieval"]["latency_ms"], 12.346)
        self.assertLessEqual(len(result["evidence"][0]["content"]), 120)
        self.assertNotIn("<tool_call>", result["evidence"][0]["content"])
        self.assertEqual(fake.calls[0]["limit"], 5)

    def test_shadow_receives_same_query_without_changing_visible_evidence(
        self,
    ) -> None:
        fake = FakeClient([make_hit()])
        shadow = FakeShadow()
        adapter = KnowledgeSearchAdapter(client=fake, shadow=shadow)
        result = adapter.execute("What is it?", language="en")
        self.assertEqual(result["evidence"][0]["page_id"], "42")
        self.assertEqual(result["evidence"][0]["id"], "K1")
        self.assertEqual(
            result["retrieval"]["shadow"]["visible_strategy"],
            "legacy",
        )
        self.assertEqual(shadow.calls[0]["query"], "What is it?")
        self.assertEqual(
            shadow.calls[0]["legacy_evidence"],
            result["evidence"],
        )
        adapter.close()
        self.assertTrue(shadow.closed)

    def test_adapter_rejects_empty_query_without_search(self) -> None:
        fake = FakeClient([])
        result = KnowledgeSearchAdapter(client=fake).execute("  ", language="zh")
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(fake.calls, [])

    def test_cited_evidence_only_returns_requested_ids(self) -> None:
        result = {
            "evidence": [
                {"id": "K1", "page_id": "1"},
                {"id": "K2", "page_id": "2"},
            ]
        }
        self.assertEqual(
            cited_evidence(result, ["K2"]),
            [{"id": "K2", "page_id": "2"}],
        )


if __name__ == "__main__":
    unittest.main()
