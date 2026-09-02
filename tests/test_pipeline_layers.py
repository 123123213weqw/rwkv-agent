from __future__ import annotations

import unittest

from rwkv_search.pipeline.answer_policy import AnswerPolicy
from rwkv_search.pipeline.discovery import DiscoveryLayer
from rwkv_search.pipeline.query_compiler import QueryCompiler, QueryHints
from rwkv_search.pipeline.reranker import RetrievalReranker
from rwkv_search.pipeline.source_selector import SourceCapability, SourceSelector


class FakeScorer:
    model_name = "fake-pair-scorer"

    def score(self, query, documents):
        subject = query.casefold().split()[0]
        return [10.0 if subject in value.casefold() else -10.0 for value in documents]


class Candidate:
    def __init__(self, url: str, title: str, rank: int) -> None:
        self.url = url
        self.title = title
        self.snippet = ""
        self.rank = rank
        self.rrf_score = 1.0 / (60 + rank)
        self.engine_score = 0.0
        self.candidate_score = self.rrf_score
        self.engine = "fixture"
        self.source_channels = []
        self.score_components = {}


class PipelineLayerTests(unittest.TestCase):
    def test_query_compiler_preserves_model_query_and_explicit_site_only(self) -> None:
        compiled = QueryCompiler().compile(
            "请查最新版本 site:example.org",
            "NebulaDB release",
            hints=QueryHints(
                freshness="latest",
                source_preference="official",
            ),
        )
        self.assertEqual(
            compiled.execution_queries,
            ("NebulaDB release site:example.org",),
        )
        self.assertEqual(compiled.freshness, "latest")
        self.assertNotIn("official", compiled.execution_queries[0])

    def test_source_selector_uses_declared_capabilities(self) -> None:
        capabilities = {
            "general": SourceCapability("general", "general web", always=True),
            "code": SourceCapability("code", "software repository source code"),
            "papers": SourceCapability("papers", "research paper DOI"),
        }
        selected = SourceSelector(scorer=FakeScorer()).select(
            "software project history",
            tuple(capabilities),
            capabilities,
            max_optional=1,
        )
        self.assertEqual(selected, ("general", "code"))

    def test_discovery_uses_entity_alignment_and_semantic_link_ranking(self) -> None:
        layer = DiscoveryLayer(scorer=FakeScorer(), intermediary_hosts=("medium.com",))
        domains = layer.select_pivot_domains(
            "NebulaDB release",
            (),
            [
                {
                    "url": "https://nebuladb.example/release",
                    "title": "NebulaDB release",
                    "candidate_score": 0.9,
                },
                {
                    "url": "https://medium.com/review",
                    "title": "NebulaDB review",
                    "candidate_score": 1.0,
                },
            ],
            max_domains=2,
        )
        self.assertEqual(domains, ["nebuladb.example"])
        links = layer.select_one_hop_links(
            "NebulaDB release",
            (),
            [
                {
                    "url": "https://nebuladb.example/",
                    "links": [
                        "https://nebuladb.example/releases/v2",
                        "https://nebuladb.example/login",
                        "https://other.example/releases/v2",
                    ],
                }
            ],
            allowed_domains=domains,
            max_links=4,
        )
        self.assertEqual([item["uri"] for item in links], ["https://nebuladb.example/releases/v2"])

    def test_one_reranker_contract_serves_candidates_and_evidence(self) -> None:
        result = RetrievalReranker(scorer=FakeScorer()).rank(
            "NebulaDB facts",
            (),
            [
                {"title": "Noise", "content": "unrelated", "uri": "https://a.example/"},
                {"title": "NebulaDB", "content": "facts", "uri": "https://b.example/"},
            ],
            limit=1,
        )
        self.assertEqual(result.items[0]["uri"], "https://b.example/")
        self.assertIn("cross_encoder", result.metadata["strategy"])

    def test_source_preference_is_metadata_not_a_query_keyword_branch(self) -> None:
        values = [
            Candidate(
                "https://industry.example/quasarkit-release",
                "QuasarKit stable release analysis",
                1,
            ),
            Candidate(
                "https://quasarkit.example/releases/v3",
                "QuasarKit v3",
                3,
            ),
        ]
        ordered, metadata = RetrievalReranker().rank_candidates(
            "QuasarKit stable release",
            (),
            values,
            limit=2,
            source_preference="official",
        )
        self.assertEqual(ordered[0].url, values[1].url)
        self.assertEqual(metadata["source_preference"], "official")
        self.assertEqual(
            ordered[0].score_components["source_preference_alignment"],
            1.0,
        )

    def test_source_preference_does_not_reward_unrelated_site_self_branding(self) -> None:
        values = [
            Candidate(
                "https://blog.example/articles/quasarkit",
                "QuasarKit review - Example Blog",
                1,
            ),
            Candidate(
                "https://example.com/unrelated",
                "Example home page",
                2,
            ),
        ]
        RetrievalReranker().rank_candidates(
            "QuasarKit stable release",
            (),
            values,
            limit=2,
            source_preference="official",
        )
        self.assertEqual(
            values[1].score_components["source_preference_alignment"],
            0.0,
        )

    def test_answer_policy_validates_only_current_evidence_ids(self) -> None:
        policy = AnswerPolicy()
        valid = policy.validate_citations(
            "NebulaDB was released [W1].",
            [{"id": "W1"}],
        )
        self.assertTrue(valid.valid)
        invalid = policy.validate_citations("Claim [W9].", [{"id": "W1"}])
        self.assertFalse(invalid.valid)
        self.assertIn("invalid_citation", invalid.errors)


if __name__ == "__main__":
    unittest.main()
