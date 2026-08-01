from __future__ import annotations

import asyncio
import copy

from rwkv_search.candidate_index import CandidateHit
from rwkv_search.config import RealtimeSearchConfig
from rwkv_search.realtime.local_discovery import (
    LocalIndexDiscovery,
    local_index_for_query,
    stable_local_query,
)
from rwkv_search.realtime.engine import RealtimeSearchEngine, select_fetch_candidates
from rwkv_search.realtime.discovery import URLDiscovery
from rwkv_search.realtime.types import DiscoveredURL


class FakeCandidateClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, int]] = []

    def search(
        self,
        query_text: str,
        *,
        index: str,
        channel_size: int,
        limit: int,
    ) -> tuple[object, list[CandidateHit], float]:
        self.calls.append((query_text, index, channel_size, limit))
        return (
            object(),
            [
                CandidateHit(
                    doc_id="python-0",
                    page_id="python",
                    title="Python",
                    text="Python is a programming language.",
                    url="https://en.wikipedia.org/wiki/Python_(programming_language)",
                    page_type="article",
                    score=0.08,
                    channels=("exact",),
                    ranks={"exact": 1},
                    modified_at="2026-06-01T00:00:00Z",
                    passage_score=1.25,
                )
            ],
            2.0,
        )


def test_local_index_selection_uses_script_and_configured_default() -> None:
    indexes = {"zh": "zh-index", "en": "en-index"}
    assert local_index_for_query("什么是 Python", indexes) == "zh-index"
    assert local_index_for_query("What is Python?", indexes) == "en-index"
    assert local_index_for_query("What is Python?", {"default": "all"}) == "all"


def test_stable_local_query_strips_only_standalone_calendar_years() -> None:
    assert (
        stable_local_query(
            "Central Atlas Tamazight speakers percentage 2024",
            strip_calendar_years=True,
        )
        == "Central Atlas Tamazight speakers percentage"
    )
    assert (
        stable_local_query(
            "2024 年 Central Atlas Tamazight 使用者比例",
            strip_calendar_years=True,
        )
        == "年 Central Atlas Tamazight 使用者比例"
    )
    assert (
        stable_local_query("Python3 CE16808", strip_calendar_years=True)
        == "Python3 CE16808"
    )
    assert stable_local_query("2024 election", strip_calendar_years=True) == "2024 election"
    assert (
        stable_local_query(
            "Central Atlas Tamazight 2024", strip_calendar_years=False
        )
        == "Central Atlas Tamazight 2024"
    )


def test_local_discovery_is_disabled_by_default() -> None:
    client = FakeCandidateClient()
    assert RealtimeSearchConfig().local_discovery_strip_calendar_years is True
    results = asyncio.run(
        LocalIndexDiscovery(
            RealtimeSearchConfig(),
            client=client,
        ).discover("What is Python?")
    )
    assert results == []
    assert client.calls == []


def test_local_discovery_converts_index_hits_to_web_candidates() -> None:
    config = RealtimeSearchConfig(
        local_discovery_enabled=True,
        local_discovery_indexes={"zh": "zh-index", "en": "en-index"},
        local_discovery_channel_size=40,
        local_discovery_limit=12,
    )
    client = FakeCandidateClient()
    results = asyncio.run(
        LocalIndexDiscovery(config, client=client).discover("What is Python?")
    )

    assert client.calls == [("What is Python?", "en-index", 40, 12)]
    assert len(results) == 1
    assert results[0].engine == "local_index"
    assert results[0].title == "Python"
    assert results[0].published_hint == "2026-06-01T00:00:00Z"
    assert results[0].discovery_stage == "local_index"
    assert results[0].rrf_score == 0.0


def test_local_discovery_does_not_weaken_explicit_site_scope() -> None:
    client = FakeCandidateClient()
    config = RealtimeSearchConfig(local_discovery_enabled=True)
    results = asyncio.run(
        LocalIndexDiscovery(config, client=client).discover(
            "Python release site:python.org"
        )
    )
    assert results == []
    assert client.calls == []


def test_local_discovery_skips_changing_information_by_default() -> None:
    client = FakeCandidateClient()
    config = RealtimeSearchConfig(local_discovery_enabled=True)
    results = asyncio.run(
        LocalIndexDiscovery(config, client=client).discover(
            "Python latest release",
            freshness="latest",
        )
    )
    assert results == []
    assert client.calls == []


def test_cached_local_text_bypasses_network_fetch() -> None:
    engine = RealtimeSearchEngine(RealtimeSearchConfig())
    candidate = DiscoveredURL(
        url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        title="Python",
        engine="local_index",
        published_hint="2026-06-01T00:00:00Z",
        cached_text=(
            "Python is a high-level programming language with a large standard "
            "library and a broad software ecosystem. "
        )
        * 3,
    )

    document = asyncio.run(engine._fetch_extract(candidate))

    assert document is not None
    assert document.title == "Python"
    assert document.published_at == "2026-06-01T00:00:00Z"
    assert document.text.startswith("Python is a high-level")
    assert document.simhash


def test_local_results_do_not_gain_false_consensus_across_query_lanes() -> None:
    candidate = DiscoveredURL(
        url="https://en.wikipedia.org/wiki/2024",
        title="2024",
        engine="local_index",
        engines=["local_index"],
        cached_text="A locally indexed article about the year 2024. " * 4,
    )

    class TwoLaneDiscovery(URLDiscovery):
        async def _discover_one(self, *args, **kwargs):
            return [copy.deepcopy(candidate)]

    results = asyncio.run(
        TwoLaneDiscovery(
            RealtimeSearchConfig(local_discovery_enabled=True),
            object(),
        ).discover(
            ["raw query", "rewritten query"],
            freshness="stable",
            max_candidates=10,
        )
    )

    assert len(results) == 1
    first_lane_score = 1.0 / 61.0 + 0.002
    assert abs(results[0].rrf_score - first_lane_score) < 1e-9


def test_strong_cached_evidence_does_not_consume_network_fetch_budget() -> None:
    weak_local = DiscoveredURL(
        url="https://en.wikipedia.org/wiki/2024",
        title="2024",
        engine="local_index",
        cached_text="generic local text " * 10,
        candidate_score=0.3,
        score_components={"entity_coverage": 0.0},
    )
    strong_local = DiscoveredURL(
        url="https://en.wikipedia.org/wiki/Punxsutawney_Phil",
        title="Punxsutawney Phil",
        engine="local_index",
        cached_text="relevant local text " * 10,
        candidate_score=0.27,
        score_components={"entity_coverage": 1.0},
    )
    web = [
        DiscoveredURL(
            url=f"https://example{i}.com/page",
            title=f"Web {i}",
            engine="bing",
        )
        for i in range(5)
    ]

    selected = select_fetch_candidates(
        [weak_local, strong_local, *web],
        max_network_fetches=4,
        local_limit=3,
        local_min_score=0.22,
        local_min_entity_coverage=0.5,
    )

    assert selected[0] is strong_local
    assert weak_local not in selected
    assert selected[1:] == web[:4]


def test_network_fetch_budget_uses_candidate_confidence_not_display_order() -> None:
    candidates = [
        DiscoveredURL(
            url=f"https://example.com/page-{index}",
            title=f"Page {index}",
            engine="dogpile",
            candidate_score=score,
            rrf_score=rrf_score,
        )
        for index, (score, rrf_score) in enumerate(
            [(0.20, 0.04), (0.80, 0.02), (0.60, 0.03), (0.80, 0.05)],
            start=1,
        )
    ]

    selected = select_fetch_candidates(
        candidates,
        max_network_fetches=3,
        local_limit=0,
        local_min_score=0.0,
        local_min_entity_coverage=0.0,
    )

    assert [item.url for item in selected] == [
        "https://example.com/page-4",
        "https://example.com/page-2",
        "https://example.com/page-3",
    ]


def test_official_search_reserves_fetch_slots_for_primary_source_shapes() -> None:
    commentary = [
        DiscoveredURL(
            url=f"https://media.example/article-{index}",
            title=f"Commentary {index}",
            engine="naver",
            candidate_score=1.0 - index * 0.05,
        )
        for index in range(6)
    ]
    primary = [
        DiscoveredURL(
            url="https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng",
            title="Official consolidated legal text",
            engine="dogpile",
            candidate_score=0.4,
        ),
        DiscoveredURL(
            url="https://github.com/huggingface/transformers/releases/tag/v5.14.1",
            title="Official release",
            engine="github",
            candidate_score=0.45,
        ),
    ]

    selected = select_fetch_candidates(
        [*commentary, *primary],
        max_network_fetches=4,
        local_limit=0,
        local_min_score=0.0,
        local_min_entity_coverage=0.0,
        source_preference="official",
    )

    assert {item.url for item in selected[:2]} == {item.url for item in primary}
    assert len(selected) == 4


def test_structured_api_evidence_bypasses_redundant_page_fetch() -> None:
    candidate = DiscoveredURL(
        url="https://github.com/example/project/releases/tag/v1.0.0",
        title="GitHub Release v1.0.0",
        snippet="Official release metadata returned by the GitHub API.",
        engine="github",
        candidate_score=0.37,
        score_components={"entity_coverage": 0.33},
        cached_text=(
            "GitHub Release v1.0.0. Official release metadata returned by the "
            "GitHub API, including the changelog, publication time, fixes, and "
            "compatibility notes required to answer the release question."
        ),
        cached_text_mode="structured_api",
    )

    selected = select_fetch_candidates(
        [candidate],
        max_network_fetches=8,
        local_limit=3,
        local_min_score=0.22,
        local_min_entity_coverage=0.5,
        source_preference="official",
    )

    class UnexpectedFetcher:
        async def fetch(self, _url):
            raise AssertionError("structured API evidence must not trigger a page GET")

    engine = RealtimeSearchEngine(RealtimeSearchConfig(enabled=True))
    engine._fetcher = UnexpectedFetcher()
    outcome = asyncio.run(engine._fetch_extract_outcome(candidate))

    assert selected == [candidate]
    assert outcome.document is not None
    assert outcome.retrieval_mode == "structured_api"
    assert outcome.document.retrieval_mode == "structured_api"
