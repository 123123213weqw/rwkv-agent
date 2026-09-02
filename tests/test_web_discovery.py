from __future__ import annotations

import asyncio
import time

from rwkv_agent.tools.web import PROFILE_FEATURES, _configure, _public_evidence
from rwkv_search.config import RealtimeSearchConfig
from rwkv_search.realtime.discovery import parse_search_html, parse_wikipedia_results
from rwkv_search.realtime.engine import RealtimeSearchEngine
from rwkv_search.realtime.types import DiscoveredURL
from rwkv_search.realtime.source_api import (
    SourceAPIDiscovery,
    parse_crossref_results,
    parse_github_commits,
    parse_github_repositories,
    parse_mediawiki_pages,
    parse_tavily_results,
    select_github_detail_scopes,
    select_source_providers,
)


class FocusedScorer:
    model_name = "fake-focused-reranker"

    def score(self, query, documents):
        query = query.casefold()
        if "创始" in query or "found" in query or "identity" in query:
            targets = ("peng bo", "bo peng")
        elif "项目" in query or "repositories" in query or "projects" in query:
            targets = ("chatrwkv",)
        elif "最新" in query or "newest" in query or "update" in query:
            targets = ("update readme", "latest rwkv-lm commit")
        else:
            targets = ("rwkv",)
        return [
            float(any(target in document.casefold() for target in targets))
            for document in documents
        ]


def test_so360_parser_uses_direct_destination_and_snippet() -> None:
    html = """
    <ul class="result"><li class="res-list">
      <h3 class="res-title"><a
        href="https://www.so.com/link?m=opaque"
        data-mdurl="https://www.poms.org/pomsocietyawards/wickhamskinnerawards">
        Wickham Skinner Awards | POMS
      </a></h3>
      <p class="res-desc">The awards recognize POM scholarship.</p>
    </li></ul>
    """

    results = parse_search_html(html, "so360")

    assert len(results) == 1
    assert results[0].url == (
        "https://www.poms.org/pomsocietyawards/wickhamskinnerawards"
    )
    assert results[0].engine == "so360"
    assert "recognize POM" in results[0].snippet


def test_public_evidence_fills_failed_fetches_from_discovery() -> None:
    evidence, stage = _public_evidence(
        [
            {
                "title": "Fetched page",
                "snippet": "Fetched evidence.",
                "url": "https://fetched.example/page",
            }
        ],
        [
            {
                "title": "Fetched duplicate",
                "snippet": "Duplicate.",
                "url": "https://fetched.example/page",
            },
            {
                "title": "Official search result",
                "snippet": "Useful search snippet.",
                "url": "https://official.example/answer",
                "rrf_score": 0.2,
            },
        ],
    )

    assert stage == "mixed"
    assert [item.uri for item in evidence] == [
        "https://fetched.example/page",
        "https://official.example/answer",
    ]
    assert [item.id for item in evidence] == ["W1", "W2"]


def test_public_evidence_keeps_eight_scoped_candidates() -> None:
    evidence, stage = _public_evidence(
        [],
        [
            {
                "title": f"Page {index}",
                "snippet": f"Snippet {index}",
                "url": f"https://official.example/{index}",
            }
            for index in range(10)
        ],
    )

    assert stage == "discovery"
    assert len(evidence) == 8
    assert evidence[-1].uri == "https://official.example/7"


def test_public_evidence_selects_compound_question_passages_from_full_page() -> None:
    filler = (
        "This navigation paragraph contains generic documentation links and "
        "unrelated introductory material. " * 12
    )
    content = "\n".join(
        [
            filler,
            "RWKV was created by Bo Peng, known as BlinkDL on GitHub.",
            filler,
            "His GitHub projects include BlinkDL/RWKV-LM, ChatRWKV and RWKV-CUDA.",
            filler,
            "The latest RWKV-LM commit updates the CUDA inference example.",
            filler,
        ]
    )

    evidence, stage = _public_evidence(
        [
            {
                "title": "RWKV official project page",
                "snippet": "Generic RWKV landing-page description.",
                "content": content,
                "url": "https://github.com/BlinkDL/RWKV-LM",
            }
        ],
        [],
        query="RWKV的创始人是谁，他的GitHub项目有哪些，最新更新是什么？",
        scorer=FocusedScorer(),
    )

    assert stage == "fetched"
    selected = evidence[0].content
    assert "Bo Peng" in selected
    assert "ChatRWKV" in selected
    assert "latest RWKV-LM commit" in selected
    assert "Generic RWKV landing-page description" not in selected
    assert len(selected) <= 900


def test_fallback_engine_override_is_explicit() -> None:
    config = _configure(
        "missing-config.json",
        profile="legacy",
        fallback_engines=("bing", "so360", "so360"),
    )

    assert config.realtime_search.fallback_engines == ["bing", "so360"]


def test_wikipedia_api_results_use_canonical_article_urls() -> None:
    results = parse_wikipedia_results(
        {
            "query": {
                "search": [
                    {
                        "title": "Punxsutawney Phil",
                        "snippet": "A <span class='searchmatch'>groundhog</span> in Pennsylvania.",
                    }
                ]
            }
        }
    )
    assert len(results) == 1
    assert results[0].url == "https://en.wikipedia.org/wiki/Punxsutawney_Phil"
    assert results[0].snippet == "A groundhog in Pennsylvania."
    assert results[0].engine == "wikipedia"


def test_wikipedia_is_an_explicit_supported_fallback() -> None:
    config = _configure(
        "missing-config.json",
        profile="legacy",
        fallback_engines=("bing", "wikipedia"),
    )
    assert config.realtime_search.fallback_engines == ["bing", "wikipedia"]


def test_public_crawl_root_becomes_site_query_without_polluting_model_text() -> None:
    class CaptureEngine:
        def __init__(self) -> None:
            self.query = ""
            self.seed_urls = ()

        def search_events(self, query, queries, **kwargs):
            self.query = query
            self.seed_urls = kwargs.get("seed_urls", ())
            assert queries == [query]
            return iter(())

        def close(self) -> None:
            pass

    engine = CaptureEngine()
    adapter = __import__("rwkv_agent.tools.web", fromlist=["WebSearchAdapter"]).WebSearchAdapter(
        engine=engine,
        profile="enhanced",
        shadow=False,
    )
    with adapter.scoped("https://www.example.org/"):
        public, trace = adapter.execute_with_trace("Who won the award?")
    assert engine.query == "Who won the award site:example.org"
    assert engine.seed_urls == ("https://www.example.org/",)
    assert public["query"] == "Who won the award?"
    assert trace["scope_root"] == "https://www.example.org/"
    assert adapter._scope_root == ""


def test_scoped_web_search_hard_filters_outside_results_even_if_engine_ignores_site() -> None:
    class IgnoringEngine:
        def search_events(self, query, queries, **_kwargs):
            assert "site:example.org" in query
            yield {
                "type": "discovery_progress",
                "progress": {
                    "candidates": [
                        {
                            "url": "https://example.org/answer",
                            "title": "Official answer",
                            "snippet": "Alice won the award.",
                        },
                        {
                            "url": "https://outside.test/wrong",
                            "title": "Wrong result",
                            "snippet": "An unrelated page.",
                        },
                    ]
                },
            }
            yield {
                "type": "realtime_result",
                "results": [
                    {
                        "url": "https://example.org/answer",
                        "title": "Official answer",
                        "content": "Alice won the award.",
                    },
                    {
                        "url": "https://outside.test/wrong",
                        "title": "Wrong result",
                        "content": "Mallory won the award.",
                    },
                ],
                "stats": {},
            }

        def close(self) -> None:
            pass

    adapter = __import__(
        "rwkv_agent.tools.web", fromlist=["WebSearchAdapter"]
    ).WebSearchAdapter(engine=IgnoringEngine(), profile="legacy", shadow=False)

    with adapter.scoped("https://www.example.org/"):
        public, trace = adapter.execute_with_trace("Who won the award?")

    assert public["scope_mode"] == "strict"
    assert [item["uri"] for item in public["evidence"]] == [
        "https://example.org/answer"
    ]
    assert public["scope_rejected"]["results"] == 1
    assert public["scope_rejected"]["candidates"] == 1
    assert all(
        item["url"].startswith("https://example.org/")
        for item in trace["candidates"]
    )


def test_realtime_engine_contract_accepts_and_normalizes_seed_urls() -> None:
    engine = RealtimeSearchEngine(RealtimeSearchConfig(enabled=False))

    events = list(
        engine.search_events(
            "Who won the award? site:example.org",
            ["Who won the award? site:example.org"],
            freshness="latest",
            depth="single",
            seed_urls=("https://www.example.org/",),
            include_candidates=True,
        )
    )
    seeds = engine._seed_candidates(
        (
            "https://www.example.org/",
            "https://www.example.org/#duplicate",
            "file:///etc/passwd",
        ),
        "Who won the award? site:example.org",
        max_candidates=5,
    )

    assert events == [{"type": "realtime_result", "results": []}]
    assert len(seeds) == 1
    assert seeds[0].url == "https://www.example.org/"
    assert seeds[0].engine == "direct"
    assert seeds[0].discovery_stages == ["seed_url"]


def test_balanced_profile_avoids_expensive_fanout() -> None:
    features = PROFILE_FEATURES["balanced"]

    assert "domain_pivot_enabled" in features
    assert "source_channels_enabled" not in features
    assert "one_hop_link_expansion_enabled" not in features


def test_source_provider_selection_is_source_shaped() -> None:
    configured = ("tavily", "github", "crossref", "mediawiki")

    providers = select_source_providers(
        "RWKV创始人是谁，他的GitHub项目和最新更新是什么？",
        configured,
    )
    assert providers[0] == "tavily"
    assert set(providers[1:]) == {"github", "mediawiki"}
    assert select_source_providers(
        "Find the original RWKV paper and DOI",
        configured,
    ) == ("tavily", "crossref")
    assert select_source_providers(
        "Python current stable release from the official website",
        configured,
    ) == ("tavily",)
    assert select_source_providers(
        "上海唐镇站如何到RWKV公司？",
        configured,
    ) == ("tavily",)
    assert select_source_providers(
        "site:github.com RWKV latest commit",
        configured,
    ) == ()


def test_source_api_keeps_fast_provider_when_another_times_out() -> None:
    class PartialDiscovery(SourceAPIDiscovery):
        async def _one(self, provider, query, diagnostics):
            del query, diagnostics
            if provider == "mediawiki":
                await asyncio.sleep(0.2)
                return []
            return [
                DiscoveredURL(
                    url="https://github.com/example/project",
                    title="example/project",
                    snippet="Official repository result",
                    engine="github",
                )
            ]

    config = RealtimeSearchConfig(
        api_discovery_providers=["github", "mediawiki"],
        api_provider_timeout_seconds=0.05,
        discovery_timeout_seconds=0.1,
    )
    diagnostics = []
    started = time.monotonic()
    results = asyncio.run(
        PartialDiscovery(config, object()).discover(
            "example GitHub项目创始人是谁",
            diagnostics=diagnostics,
        )
    )

    assert time.monotonic() - started < 0.15
    assert [item.engine for item in results] == ["github"]
    assert any(item["engine"] == "mediawiki" for item in diagnostics)


def test_source_api_activation_does_not_fan_out_from_uncalibrated_scorer() -> None:
    class AlwaysHighScorer:
        model_name = "uncalibrated-test-scorer"

        def score(self, query, documents):
            return [100.0] * len(documents)

    class CaptureDiscovery(SourceAPIDiscovery):
        providers: list[str] = []

        async def _one(self, provider, query, diagnostics):
            del query, diagnostics
            self.providers.append(provider)
            return []

    discovery = CaptureDiscovery(
        RealtimeSearchConfig(
            api_discovery_providers=["github", "crossref", "mediawiki"]
        ),
        object(),
        semantic_scorer=AlwaysHighScorer(),
    )
    asyncio.run(
        discovery.discover(
            "Python current stable release from the official website"
        )
    )

    assert discovery.providers == []


def test_tavily_results_keep_rank_score_and_content() -> None:
    results = parse_tavily_results(
        {
            "results": [
                {
                    "url": "https://www.rwkv.com/",
                    "title": "RWKV",
                    "content": "Official RWKV website.",
                    "score": 0.9,
                }
            ]
        }
    )

    assert len(results) == 1
    assert results[0].engine == "tavily"
    assert results[0].engine_score == 0.9
    assert results[0].snippet == "Official RWKV website."


def test_github_repository_and_commit_results_use_html_urls() -> None:
    repositories = parse_github_repositories(
        {
            "items": [
                {
                    "full_name": "BlinkDL/RWKV-LM",
                    "html_url": "https://github.com/BlinkDL/RWKV-LM",
                    "description": "RWKV language model",
                    "language": "Python",
                    "stargazers_count": 15000,
                    "pushed_at": "2026-07-27T00:00:00Z",
                }
            ]
        }
    )
    commits = parse_github_commits(
        [
            {
                "sha": "1234567890abcdef",
                "html_url": "https://github.com/BlinkDL/RWKV-LM/commit/1234",
                "commit": {
                    "message": "Update inference example",
                    "author": {"date": "2026-07-27T01:00:00Z"},
                },
            }
        ]
    )

    assert repositories[0].title == "BlinkDL/RWKV-LM"
    assert repositories[0].published_hint == "2026-07-27T00:00:00Z"
    assert repositories[0].cached_text_mode == "structured_api"
    assert "Stars: 15000" in repositories[0].cached_text
    assert commits[0].url.endswith("/commit/1234")
    assert "Update inference" in commits[0].title
    assert commits[0].cached_text_mode == "structured_api"


def test_github_detail_scopes_are_capability_selected() -> None:
    assert select_github_detail_scopes("vLLM latest release") == ("releases",)
    assert set(
        select_github_detail_scopes(
            "RWKV creator GitHub projects latest update"
        )
    ) == {"profile", "repos", "commits"}
    assert select_github_detail_scopes("Find the official GitHub repository") == ()


def test_github_discovery_uses_entity_name_not_auth_token() -> None:
    class CaptureDiscovery(SourceAPIDiscovery):
        queries: list[str] = []

        async def _request_json(self, method, url, **kwargs):
            if url.endswith("/search/repositories"):
                self.queries.append(kwargs["params"]["q"])
                return {
                    "items": [
                        {
                            "full_name": "BlinkDL/RWKV-LM",
                            "html_url": "https://github.com/BlinkDL/RWKV-LM",
                        }
                    ]
                }
            if url.endswith("/commits") or url.endswith("/repos"):
                return []
            return {}

    discovery = CaptureDiscovery(
        RealtimeSearchConfig(api_discovery_providers=["github"]),
        object(),
    )
    results = asyncio.run(
        discovery._github("rwkv创始人的github项目和最新更新")
    )

    assert discovery.queries == ["rwkv in:name"]
    assert results[0].title == "BlinkDL/RWKV-LM"


def test_github_discovery_treats_creator_as_relation_not_entity() -> None:
    class CaptureDiscovery(SourceAPIDiscovery):
        queries: list[str] = []

        async def _request_json(self, method, url, **kwargs):
            if url.endswith("/search/repositories"):
                self.queries.append(kwargs["params"]["q"])
                return {
                    "items": [
                        {
                            "full_name": "BlinkDL/RWKV-LM",
                            "html_url": "https://github.com/BlinkDL/RWKV-LM",
                        }
                    ]
                }
            if url.endswith("/commits") or url.endswith("/repos"):
                return []
            return {}

    discovery = CaptureDiscovery(
        RealtimeSearchConfig(api_discovery_providers=["github"]),
        object(),
    )
    asyncio.run(
        discovery._github("rwkv creator github projects latest update")
    )

    assert discovery.queries == ["rwkv in:name"]


def test_github_discovery_singleflights_same_entity() -> None:
    class CaptureDiscovery(SourceAPIDiscovery):
        search_calls = 0

        async def _request_json(self, method, url, **kwargs):
            if url.endswith("/search/repositories"):
                self.search_calls += 1
                await asyncio.sleep(0.01)
                return {
                    "items": [
                        {
                            "full_name": "BlinkDL/RWKV-LM",
                            "html_url": "https://github.com/BlinkDL/RWKV-LM",
                        }
                    ]
                }
            if url.endswith("/commits") or url.endswith("/repos"):
                return []
            return {}

    discovery = CaptureDiscovery(
        RealtimeSearchConfig(api_discovery_providers=["github"]),
        object(),
    )

    async def run():
        return await asyncio.gather(
            discovery._github("rwkv creator projects"),
            discovery._github("RWKV founder latest update"),
            discovery._github("rwkv GitHub repository"),
        )

    groups = asyncio.run(run())

    assert discovery.search_calls == 1
    assert all(group[0].title == "BlinkDL/RWKV-LM" for group in groups)
    assert groups[0][0] is not groups[1][0]


def test_github_discovery_emits_owner_profile_repo_index_and_latest_commit() -> None:
    class StructuredDiscovery(SourceAPIDiscovery):
        async def _request_json(self, method, url, **kwargs):
            del method, kwargs
            if url.endswith("/search/repositories"):
                return {
                    "items": [
                        {
                            "full_name": "BlinkDL/RWKV-LM",
                            "html_url": "https://github.com/BlinkDL/RWKV-LM",
                        }
                    ]
                }
            if url.endswith("/users/BlinkDL"):
                return {
                    "login": "BlinkDL",
                    "name": "PENG Bo",
                    "bio": "RWKV is all you need",
                    "html_url": "https://github.com/BlinkDL",
                    "public_repos": 35,
                }
            if url.endswith("/users/BlinkDL/repos"):
                return [
                    {
                        "full_name": "BlinkDL/ChatRWKV",
                        "html_url": "https://github.com/BlinkDL/ChatRWKV",
                        "description": "Chat with RWKV",
                        "pushed_at": "2026-07-27T00:00:00Z",
                    },
                    {
                        "full_name": "BlinkDL/RWKV-CUDA",
                        "html_url": "https://github.com/BlinkDL/RWKV-CUDA",
                        "description": "CUDA kernels",
                    },
                ]
            if url.endswith("/commits"):
                return [
                    {
                        "sha": "1234567890abcdef",
                        "html_url": "https://github.com/BlinkDL/RWKV-LM/commit/1234",
                        "commit": {
                            "message": "Update inference example",
                            "author": {"date": "2026-07-27T01:00:00Z"},
                        },
                    }
                ]
            return {}

    discovery = StructuredDiscovery(
        RealtimeSearchConfig(api_discovery_providers=["github"]),
        object(),
    )
    results = asyncio.run(
        discovery._github("RWKV creator GitHub projects latest update")
    )
    by_stage = {item.discovery_stage: item for item in results}

    assert by_stage["github_owner_profile"].title == "PENG Bo (@BlinkDL)"
    assert "Primary repository owner: BlinkDL" in by_stage[
        "github_owner_profile"
    ].snippet
    assert "BlinkDL/ChatRWKV" in by_stage["github_owner_repository_index"].snippet
    assert "BlinkDL/RWKV-CUDA" in by_stage["github_owner_repository_index"].snippet
    assert by_stage["github_latest_commit"].published_hint == "2026-07-27T01:00:00Z"


def test_github_release_query_uses_one_detail_endpoint() -> None:
    class CaptureDiscovery(SourceAPIDiscovery):
        urls: list[str] = []

        async def _request_json(self, method, url, **kwargs):
            del method, kwargs
            self.urls.append(url)
            if url.endswith("/search/repositories"):
                return {
                    "items": [
                        {
                            "full_name": "vllm-project/vllm",
                            "html_url": "https://github.com/vllm-project/vllm",
                        }
                    ]
                }
            if url.endswith("/releases/latest"):
                return {
                    "tag_name": "v1.0.0",
                    "html_url": "https://github.com/vllm-project/vllm/releases/tag/v1.0.0",
                }
            raise AssertionError(f"unexpected GitHub endpoint: {url}")

    discovery = CaptureDiscovery(
        RealtimeSearchConfig(api_discovery_providers=["github"]),
        object(),
    )
    results = asyncio.run(discovery._github("vLLM latest release"))

    assert len(discovery.urls) == 2
    release = next(
        item for item in results if item.discovery_stage == "github_latest_release"
    )
    assert release.cached_text_mode == "structured_api"


def test_public_evidence_uses_semantic_relevance_not_discovery_stage() -> None:
    results = [
        {
            "title": f"Fetched {index}",
            "content": "Fetched content",
            "url": f"https://fetched.example/{index}",
        }
        for index in range(5)
    ]
    candidates = [
        {
            "title": "Generic high-ranked result",
            "snippet": "Generic result",
            "url": "https://generic.example/",
        },
        {
            "title": "PENG Bo (@BlinkDL)",
            "snippet": "GitHub user: BlinkDL",
            "url": "https://github.com/BlinkDL",
            "engine": "github",
            "discovery_stage": "github_owner_profile",
        },
        {
            "title": "BlinkDL GitHub repositories",
            "snippet": "BlinkDL/RWKV-LM; BlinkDL/ChatRWKV",
            "url": "https://github.com/BlinkDL?tab=repositories",
            "engine": "github",
            "discovery_stage": "github_owner_repository_index",
        },
        {
            "title": "Commit 1234: Update README",
            "snippet": "Update README",
            "url": "https://github.com/BlinkDL/RWKV-LM/commit/1234",
            "engine": "github",
            "published_hint": "2026-07-27T01:00:00Z",
            "discovery_stage": "github_latest_commit",
        },
    ]

    evidence, stage = _public_evidence(
        results,
        candidates,
        query="Find the newest BlinkDL update",
        scorer=FocusedScorer(),
    )

    assert stage == "mixed"
    assert any(item.title == "Commit 1234: Update README" for item in evidence)


def test_public_evidence_reserves_structured_sources_without_scorer() -> None:
    results = [
        {
            "title": f"Generic fetched page {index}",
            "content": "RWKV general discussion",
            "url": f"https://generic.example/{index}",
        }
        for index in range(12)
    ]
    structured = [
        {
            "title": "PENG Bo (@BlinkDL)",
            "snippet": "GitHub user: BlinkDL",
            "url": "https://github.com/BlinkDL",
            "engine": "github",
        },
        {
            "title": "BlinkDL GitHub repositories",
            "snippet": "Public repositories: RWKV-LM, ChatRWKV, RWKV-CUDA.",
            "url": "https://github.com/BlinkDL?tab=repositories",
            "engine": "github",
        },
        {
            "title": "Commit latest: Update README",
            "snippet": "Update README",
            "url": "https://github.com/BlinkDL/RWKV-LM/commit/latest",
            "engine": "github",
        },
    ]

    evidence, stage = _public_evidence(
        results,
        structured,
        query="RWKV founder GitHub projects latest update",
    )

    uris = {item.uri for item in evidence}
    assert stage == "mixed"
    assert len(evidence) == 8
    assert {item["url"] for item in structured} <= uris


def test_public_evidence_keeps_one_latest_record_per_structured_stage() -> None:
    candidates = [
        {
            "title": "BlinkDL repositories",
            "snippet": "RWKV-LM, ChatRWKV",
            "url": "https://github.com/BlinkDL?tab=repositories",
            "engine": "github",
            "discovery_stage": "github_owner_repository_index",
        },
        *[
            {
                "title": f"Commit {day}",
                "snippet": "Update README.md",
                "url": f"https://github.com/BlinkDL/RWKV-LM/commit/{day}",
                "engine": "github",
                "published_hint": f"2026-07-{day}T01:00:00Z",
                "discovery_stage": "github_latest_commit",
            }
            for day in (21, 22, 23)
        ],
    ]

    evidence, _ = _public_evidence(
        [],
        candidates,
        query="BlinkDL projects latest update",
    )

    assert any(item.uri.endswith("?tab=repositories") for item in evidence)
    commits = [
        item for item in evidence if item.discovery_stage == "github_latest_commit"
    ]
    assert len(commits) == 1
    assert commits[0].uri.endswith("/commit/23")


def test_structured_lane_preserves_identity_index_and_latest_event() -> None:
    candidates = [
        {
            "title": "PENG Bo (@BlinkDL)",
            "snippet": "GitHub profile",
            "url": "https://github.com/BlinkDL",
            "engine": "github",
            "discovery_stage": "github_owner_profile",
        },
        {
            "title": "BlinkDL repositories",
            "snippet": "RWKV-LM, ChatRWKV, RWKV-CUDA",
            "url": "https://github.com/BlinkDL?tab=repositories",
            "engine": "github",
            "discovery_stage": "github_owner_repository_index",
        },
        {
            "title": "Commit latest",
            "snippet": "Update README.md",
            "url": "https://github.com/BlinkDL/RWKV-LM/commit/latest",
            "engine": "github",
            "discovery_stage": "github_latest_commit",
        },
        {
            "title": "Release 1",
            "snippet": "Release",
            "url": "https://github.com/BlinkDL/RWKV-LM/releases/tag/1",
            "engine": "github",
            "discovery_stage": "github_latest_release",
        },
        {
            "title": "BlinkDL/ChatRWKV",
            "snippet": "Repository",
            "url": "https://github.com/BlinkDL/ChatRWKV",
            "engine": "github",
            "discovery_stage": "github_owner_repository",
        },
        {
            "title": "RWKV search result",
            "snippet": "Repository search",
            "url": "https://github.com/example/rwkv",
            "engine": "github",
            "discovery_stage": "github_repository_search",
        },
    ]

    evidence, _ = _public_evidence(
        [],
        candidates,
        query="RWKV founder projects latest update",
    )
    stages = {item.discovery_stage for item in evidence}

    assert "github_owner_profile" in stages
    assert "github_owner_repository_index" in stages
    assert "github_latest_commit" in stages


def test_focused_queries_select_different_evidence_without_url_shape_rules() -> None:
    results = [
        {
            "title": f"Generic fetched page {index}",
            "content": "Generic fetched content",
            "url": f"https://generic.example/{index}",
        }
        for index in range(8)
    ]
    candidates = [
        {
            "title": "PENG Bo (@BlinkDL)",
            "snippet": "GitHub user: BlinkDL",
            "url": "https://github.com/BlinkDL",
            "engine": "github",
            "discovery_stage": "github_owner_profile",
        },
        {
            "title": "BlinkDL GitHub repositories",
            "snippet": "Public repositories: RWKV-LM, ChatRWKV, RWKV-CUDA.",
            "url": "https://github.com/BlinkDL?tab=repositories",
            "engine": "github",
            "discovery_stage": "github_owner_repository_index",
        },
        {
            "title": "Commit latest: Update README",
            "snippet": "Update README",
            "url": "https://github.com/BlinkDL/RWKV-LM/commit/latest",
            "engine": "github",
            "published_hint": "2026-07-27T01:00:00Z",
            "discovery_stage": "github_latest_commit",
        },
        {
            "title": "Commit older: Previous update",
            "snippet": "Previous update",
            "url": "https://github.com/BlinkDL/RWKV-LM/commit/older",
            "engine": "github",
            "published_hint": "2026-07-26T01:00:00Z",
            "discovery_stage": "github_latest_commit",
        },
        {
            "title": "RWKV-LM v1.0",
            "snippet": "Latest release",
            "url": "https://github.com/BlinkDL/RWKV-LM/releases/tag/v1.0",
            "engine": "github",
            "published_hint": "2026-07-20T01:00:00Z",
            "discovery_stage": "github_latest_release",
        },
    ]

    cases = (
        ("Find the founder identity", "https://github.com/BlinkDL"),
        (
            "Find the complete repositories and projects list",
            "https://github.com/BlinkDL?tab=repositories",
        ),
        (
            "Find the newest update timestamp",
            "https://github.com/BlinkDL/RWKV-LM/commit/latest",
        ),
    )
    for query, expected_uri in cases:
        evidence, stage = _public_evidence(
            results,
            candidates,
            query=query,
            scorer=FocusedScorer(),
        )
        assert stage == "mixed"
        assert len(evidence) == 8
        assert expected_uri in {item.uri for item in evidence}


def test_structured_repo_index_replaces_thin_fetched_duplicate() -> None:
    uri = "https://github.com/BlinkDL?tab=repositories"
    evidence, _ = _public_evidence(
        [
            {
                "title": "BlinkDL repositories",
                "content": "Uh oh! There was an error while loading.",
                "url": uri,
            }
        ],
        [
            {
                "title": "BlinkDL GitHub repositories",
                "snippet": "Public repositories: RWKV-LM, ChatRWKV, RWKV-CUDA.",
                "url": uri,
                "engine": "github",
                "discovery_stage": "github_owner_repository_index",
            }
        ],
        query="GitHub projects",
    )

    assert evidence[0].content == (
        "Public repositories: RWKV-LM, ChatRWKV, RWKV-CUDA."
    )


def test_crossref_and_mediawiki_results_become_candidates() -> None:
    crossref = parse_crossref_results(
        {
            "message": {
                "items": [
                    {
                        "DOI": "10.1000/rwkv",
                        "URL": "https://doi.org/10.1000/rwkv",
                        "title": ["RWKV paper"],
                        "author": [{"given": "Alice", "family": "Example"}],
                        "published": {"date-parts": [[2026, 7, 1]]},
                        "score": 25.0,
                    }
                ]
            }
        }
    )
    mediawiki = parse_mediawiki_pages(
        {
            "query": {
                "pages": [
                    {
                        "title": "RWKV",
                        "fullurl": "https://zh.wikipedia.org/wiki/RWKV",
                        "extract": "RWKV是一种语言模型架构。",
                        "revisions": [{"timestamp": "2026-07-01T00:00:00Z"}],
                    }
                ]
            }
        },
        language="zh",
    )

    assert crossref[0].url == "https://doi.org/10.1000/rwkv"
    assert crossref[0].published_hint == "2026-07-01"
    assert crossref[0].cached_text_mode == "structured_api"
    assert mediawiki[0].engine == "mediawiki"
    assert "语言模型" in mediawiki[0].snippet
    assert mediawiki[0].cached_text_mode == "structured_api"


def test_api_provider_override_is_explicit() -> None:
    config = _configure(
        "missing-config.json",
        profile="legacy",
        fallback_engines=("bing",),
        api_providers=("tavily", "github", "crossref", "mediawiki"),
    )

    assert config.realtime_search.fallback_engines == ["bing"]
    assert config.realtime_search.api_discovery_providers == [
        "tavily",
        "github",
        "crossref",
        "mediawiki",
    ]
