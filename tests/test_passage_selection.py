from __future__ import annotations

import unittest

from rwkv_search.config import SearchConfig
from rwkv_search.evidence import EvidenceBuilder
from rwkv_search.passage_selection import (
    query_aspects,
    select_page_passages,
    split_web_passages,
)
from rwkv_search.search import SearchResult


class FakeScorer:
    model_name = "fake-cross-encoder"

    def score(self, query, documents):
        query = query.casefold()
        if "创始" in query or "found" in query:
            target = "bo peng"
        elif "项目" in query or "github" in query or "project" in query:
            target = "rwkv-lm"
        elif "最新" in query or "更新" in query or "latest" in query:
            target = "latest commit"
        else:
            target = "rwkv"
        return [float(target in value.casefold()) for value in documents]


def long_rwkv_page() -> str:
    return "\n\n".join(
        [
            "Navigation Documentation Download Community About Contact " * 12,
            "RWKV is a recurrent neural network language model architecture. " * 12,
            "RWKV was created by Bo Peng, who uses the GitHub account BlinkDL. "
            "The project is developed with the RWKV open-source community. " * 8,
            "BlinkDL maintains GitHub projects including RWKV-LM and ChatRWKV. "
            "The RWKV-LM repository contains training and inference code, while ChatRWKV contains chat examples. " * 7,
            "The latest commit updates RWKV-7 inference examples and model links. "
            "Release and commit timestamps must be read from the repository at request time. " * 8,
            "Unrelated footer privacy policy cookies careers advertising " * 12,
        ]
    )


class PassageSelectionTests(unittest.TestCase):
    def test_compound_query_keeps_entity_anchor_for_each_aspect(self) -> None:
        aspects = query_aspects(
            "RWKV的创始人是谁，他的GitHub项目都有哪些，最新的更新是什么？"
        )
        self.assertEqual(len(aspects), 3)
        self.assertTrue(all("rwkv" in value.casefold() for value in aspects))

    def test_paragraph_first_split_preserves_offsets(self) -> None:
        text = "第一段。" * 80 + "\n\n" + "第二段。" * 80
        passages = split_web_passages(text, target_chars=260, max_chars=400)
        self.assertGreaterEqual(len(passages), 2)
        for passage in passages:
            self.assertLessEqual(len(passage.text), 400)
            self.assertGreaterEqual(passage.char_end, passage.char_start)

    def test_selects_founder_projects_and_latest_update_not_navigation(self) -> None:
        selection = select_page_passages(
            "RWKV的创始人是谁，他的GitHub项目都有哪些，最新的更新是什么？",
            "RWKV official project",
            long_rwkv_page(),
            max_passages=3,
            max_chars=3600,
            target_chars=700,
            hard_max_chars=1000,
            scorer=FakeScorer(),
        )
        self.assertIn("Bo Peng", selection.text)
        self.assertIn("RWKV-LM", selection.text)
        self.assertIn("latest commit", selection.text)
        self.assertNotIn("Navigation Documentation", selection.text)
        self.assertEqual(
            selection.strategy,
            "paragraph_query_view_mmr_v1+cross_encoder",
        )

    def test_cross_encoder_is_optional_and_traced(self) -> None:
        selection = select_page_passages(
            "RWKV latest update",
            "RWKV",
            long_rwkv_page(),
            scorer=FakeScorer(),
        )
        self.assertIn("cross_encoder", selection.strategy)
        self.assertEqual(selection.reranker_model, "fake-cross-encoder")

    def test_evidence_builder_records_selected_spans(self) -> None:
        result = SearchResult(
            document_id=1,
            url="https://github.com/BlinkDL/RWKV-LM",
            title="BlinkDL/RWKV-LM",
            snippet="RWKV",
            content=long_rwkv_page(),
            published_at=None,
            fetched_at=1.0,
            source_type="official_repository",
            authority=0.95,
            score=0.9,
            score_components={"rrf": 0.1},
        )
        evidence = EvidenceBuilder(
            SearchConfig(),
            passage_scorer=FakeScorer(),
        ).build(
            "RWKV的创始人是谁，他的GitHub项目都有哪些，最新的更新是什么？",
            [result],
        )[0]
        passage = evidence.metadata["passage_selection"]
        self.assertGreaterEqual(passage["total_passages"], 3)
        self.assertEqual(len(passage["query_aspects"]), 3)
        self.assertIn("Bo Peng", evidence.text)
        self.assertIn("RWKV-LM", evidence.text)
        self.assertIn("latest commit", evidence.text)


if __name__ == "__main__":
    unittest.main()
