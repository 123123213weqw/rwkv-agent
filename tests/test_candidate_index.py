from __future__ import annotations

import unittest

from rwkv_search.analysis import DocumentAnalyzer, QueryAnalyzer
from rwkv_search.candidate_index import (
    CHANNEL_WEIGHTS,
    CandidateIndexClient,
    build_channel_queries,
    candidate_index_mapping,
    chunk_to_index_document,
)
from rwkv_search.wikipedia import WikipediaChunk


class CandidateIndexTests(unittest.TestCase):
    @staticmethod
    def _source(
        *,
        doc_id: str,
        chunk_id: int,
        char_start: int,
        body: str,
        body_words: str,
        headings=(),
        heading_words: str = "",
    ):
        return {
            "doc_id": doc_id,
            "page_id": "3881",
            "chunk_id": chunk_id,
            "title_original": "Python",
            "heading_original": list(headings),
            "heading_words": heading_words,
            "body_original": body,
            "body_words": body_words,
            "url": "https://zh.wikipedia.org/wiki/Python",
            "page_type": "article",
            "source": "finewiki",
            "wikidata_id": "Q28865",
            "modified_at": "2026-06-01T00:00:00Z",
            "char_start": char_start,
        }

    @classmethod
    def _search_client(cls, *, identity: bool = False):
        version = cls._source(
            doc_id="3881#3",
            chunk_id=3,
            char_start=1842,
            body="Python 3.14 增加了自由线程支持和其他版本变化。",
            body_words="python 3.14 增加 自由 线程 支持 版本 变化",
            headings=("历史",),
            heading_words="历史",
        )
        lead = cls._source(
            doc_id="3881#0",
            chunk_id=0,
            char_start=0,
            body="Python 是一种广泛使用的解释型、高级和通用编程语言。",
            body_words="python 是 一种 广泛 使用 解释型 高级 通用 编程 语言",
            headings=("简介",),
            heading_words="简介",
        )
        outer = {
            "_id": "3881#3",
            "_score": 50.0,
            "_source": version,
            "matched_queries": ["identity"] if identity else [],
            "inner_hits": {
                "top_passages": {
                    "hits": {
                        "hits": [
                            {"_id": "3881#3", "_score": 50.0, "_source": version}
                        ]
                    }
                },
                "lead_passage": {
                    "hits": {
                        "hits": [
                            {"_id": "3881#0", "_score": 40.0, "_source": lead}
                        ]
                    }
                },
            },
        }

        class FakeClient(CandidateIndexClient):
            def _request(self, method, path, body=None, **kwargs):
                request_count = len((body or b"").decode("utf-8").splitlines()) // 2
                return {
                    "responses": [
                        {"hits": {"hits": [outer]}} for _ in range(request_count)
                    ]
                }

        return FakeClient("http://unused")

    def test_mapping_is_strict_and_uses_pre_tokenized_whitespace_fields(self) -> None:
        mapping = candidate_index_mapping()
        self.assertEqual(mapping["mappings"]["dynamic"], "strict")
        properties = mapping["mappings"]["properties"]
        self.assertEqual(properties["body_words"]["analyzer"], "whitespace")
        self.assertEqual(properties["title_normalized"]["type"], "keyword")
        self.assertEqual(properties["wikidata_id"]["type"], "keyword")
        self.assertEqual(properties["metadata_words"]["analyzer"], "whitespace")
        self.assertEqual(properties["alias_normalized"]["type"], "keyword")
        self.assertEqual(properties["alias_words"]["analyzer"], "whitespace")

    def test_chunk_payload_keeps_raw_evidence_and_separate_channels(self) -> None:
        chunk = WikipediaChunk(
            doc_id="1#0",
            page_id="1",
            chunk_id=0,
            title="Python",
            text="Python 是一种编程语言。",
            headings=("简介",),
            url="https://zh.wikipedia.org/wiki/Python",
            snapshot_date="20260301",
            page_type="article",
            char_start=0,
            char_end=15,
            aliases=("Python语言",),
        )
        payload = chunk_to_index_document(chunk, DocumentAnalyzer())
        self.assertEqual(payload["title_normalized"], "python")
        self.assertEqual(payload["body_original"], chunk.text)
        self.assertIn("python", payload["body_words"].split())
        self.assertEqual(payload["source"], "wikipedia")
        self.assertEqual(payload["metadata_words"], "")
        self.assertEqual(payload["alias_original"], ["Python语言"])
        self.assertIn("python", payload["alias_words"].split())

    def test_query_builds_exact_word_and_bigram_channels_without_noise_words(self) -> None:
        analysis = QueryAnalyzer().analyze("什么是胰腺癌的一个疾病")
        channels = dict(build_channel_queries(analysis))
        self.assertEqual(set(channels), {"exact", "alias", "word", "bigram"})
        word_query = channels["word"]["query"]["multi_match"]["query"]
        self.assertNotIn("什么", word_query)
        self.assertNotIn("的", word_query)
        self.assertNotIn("一个", word_query)
        self.assertIn("胰腺癌", word_query)
        self.assertGreater(CHANNEL_WEIGHTS["word"], CHANNEL_WEIGHTS["bigram"])
        identity = channels["exact"]["query"]["bool"]["should"][0]
        self.assertEqual(identity["term"]["title_normalized"]["_name"], "identity")
        self.assertEqual(channels["alias"]["collapse"]["field"], "page_id")
        self.assertTrue(all(body["collapse"]["field"] == "page_id" for body in channels.values()))
        inner_hits = channels["word"]["collapse"]["inner_hits"]
        self.assertEqual(
            [item["name"] for item in inner_hits],
            ["top_passages", "lead_passage"],
        )
        self.assertEqual(inner_hits[0]["size"], 2)

    def test_mixed_script_person_name_does_not_emit_noisy_cjk_bigram_channel(self) -> None:
        analysis = QueryAnalyzer().analyze("Guido van Rossum设计的编程语言")
        channels = dict(build_channel_queries(analysis))
        self.assertIn("word", channels)
        self.assertNotIn("bigram", channels)
        alias_should = channels["alias"]["query"]["bool"]["must"][0]["bool"]["should"]
        self.assertTrue(all("alias_words" not in clause.get("term", {}) for clause in alias_should))

    def test_definition_query_selects_page_lead_instead_of_version_chunk(self) -> None:
        client = self._search_client(identity=True)
        _, hits, _ = client.search("Python是什么", index="test", limit=1)
        self.assertEqual(hits[0].doc_id, "3881#0")
        self.assertEqual(hits[0].chunk_id, 0)
        self.assertEqual(hits[0].char_start, 0)
        self.assertIn("编程语言", hits[0].text)
        self.assertEqual(hits[0].candidate_chunk_count, 2)
        self.assertGreater(hits[0].passage_score, 1.0)

    def test_specific_query_keeps_the_relevant_non_lead_passage(self) -> None:
        client = self._search_client()
        _, hits, _ = client.search("Python 3.14版本变化", index="test", limit=1)
        self.assertEqual(hits[0].doc_id, "3881#3")
        self.assertEqual(hits[0].chunk_id, 3)
        self.assertIn("3.14", hits[0].text)


if __name__ == "__main__":
    unittest.main()
