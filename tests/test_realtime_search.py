from __future__ import annotations

import asyncio
import queue
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rwkv_search.config import RealtimeSearchConfig, SearchConfig
from rwkv_search.db import SearchDatabase
from rwkv_search.g1i_types import G1ICompletion
from rwkv_search.realtime.cache import TTLByteCache
from rwkv_search.realtime.discovery import (
    URLDiscovery,
    bing_search_headers,
    bing_search_params,
    parse_search_html,
    parse_searxng_results,
    searxng_engines_for_query,
    searxng_search_params,
)
from rwkv_search.realtime.engine import RealtimeSearchEngine
from rwkv_search.realtime.extractor import classify_source, extract_page
from rwkv_search.realtime.fetcher import AsyncPageFetcher, FetchError
from rwkv_search.realtime.precision_discovery import compact_general_query
from rwkv_search.realtime.ranker import rank_documents, to_search_results
from rwkv_search.realtime.types import DiscoveredURL, FetchedPage, RealtimeDocument
from rwkv_search.search import SearchResult
from rwkv_search.search_reasoning import CFeedbackPlanner
from rwkv_search.service import SearchService


class RealtimeFixtureHandler(BaseHTTPRequestHandler):
    requests = []

    def do_HEAD(self) -> None:
        type(self).requests.append(("HEAD", self.path))
        self.send_response(405)
        self.end_headers()

    def do_GET(self) -> None:
        type(self).requests.append(("GET", self.path))
        body = (
            "<html><head><title>实时搜索正文</title>"
            "<meta property='article:published_time' content='2026-07-16T08:00:00Z'>"
            "</head><body><nav>导航噪声</nav><main><h1>低资源抓取</h1>"
            "<p>实时搜索只发送一次网页 GET 请求，不发送 HEAD，也不访问 robots.txt。"
            "正文抽取后直接进入证据排序，不写入持久化知识库。</p></main></body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


class FakeRealtimeEngine:
    def __init__(self) -> None:
        self.called = False

    def search_events(self, query, queries, *, freshness, depth, cancel_event=None):
        self.called = True
        yield {
            "type": "discovery_progress",
            "progress": {"candidate_count": 1, "query_count": len(queries)},
        }
        yield {
            "type": "fetch_progress",
            "progress": {"attempted": 1, "succeeded": 1, "failed": 0, "total": 1},
        }
        yield {
            "type": "realtime_result",
            "results": [
                SearchResult(
                    document_id=-9,
                    url="https://docs.example/realtime",
                    title="实时网页证据",
                    snippet="网页正文证据",
                    content="实时网页正文提供了可核查的搜索证据，并且不会写入本地数据库。",
                    published_at="2026-07-16T08:00:00Z",
                    fetched_at=time.time(),
                    source_type="official_docs",
                    authority=0.9,
                    score=1.0,
                    score_components={"realtime": 1.0},
                )
            ],
            "stats": {"fetched": 1},
        }


class FakeDiscovery:
    async def discover(
        self,
        queries,
        *,
        freshness,
        max_candidates,
        diagnostics=None,
        source_channels=(),
    ):
        return [
            DiscoveredURL(
                url="https://docs.example/release",
                title="Official release",
                snippet="Official release notes",
                engine="fixture",
                rank=2,
                rrf_score=0.25,
            )
        ]


class FakePageFetcher:
    def __init__(self) -> None:
        self.calls = []

    async def fetch(self, url):
        self.calls.append(url)
        return FetchedPage(
            requested_url=url,
            final_url=url,
            status=200,
            content_type="text/html; charset=utf-8",
            body=(
                "<html><head><title>Official release</title></head><body><main>"
                "<p>This official release page contains enough useful text for extraction, "
                "ranking, and benchmark observability without a network request.</p>"
                "</main></body></html>"
            ).encode(),
            fetched_at=time.time(),
            elapsed_ms=4.0,
            headers={},
        )


class FeedbackDiscoveryFixture:
    def __init__(self, *, satisfied: bool = False) -> None:
        self.calls = []
        self.satisfied = satisfied

    async def discover(
        self,
        queries,
        *,
        freshness,
        max_candidates,
        diagnostics=None,
        source_channels=(),
    ):
        self.calls.append(list(queries))
        if len(self.calls) == 1 and self.satisfied:
            return [
                DiscoveredURL(
                    url=f"https://www.python.org/{suffix}",
                    title=title,
                    snippet="Official Python stable release information",
                    engine="fixture",
                    rank=index,
                )
                for index, (suffix, title) in enumerate(
                    (
                        ("downloads/", "Download Python"),
                        ("downloads/release/", "Python releases"),
                        ("doc/", "Python documentation"),
                    ),
                    1,
                )
            ]
        if len(self.calls) == 1:
            return [
                DiscoveredURL(
                    url="https://blog.example/python",
                    title="A third-party Python article",
                    snippet="General programming article",
                    engine="fixture",
                    rank=1,
                )
            ]
        return [
            DiscoveredURL(
                url="https://www.python.org/downloads/",
                title="Download Python",
                snippet="Current stable Python release",
                engine="fixture",
                rank=1,
            )
        ]


class PrecisionDiscoveryFixture:
    def __init__(self) -> None:
        self.calls = []

    async def discover(
        self,
        queries,
        *,
        freshness,
        max_candidates,
        diagnostics=None,
        source_channels=(),
    ):
        self.calls.append((list(queries), tuple(source_channels)))
        output = [
            DiscoveredURL(
                url="https://www.python.org/",
                title="Official Python website",
                snippet="Official Python releases and documentation",
                engine="fixture",
                rank=1,
                rrf_score=0.2,
                engines=["fixture"],
            )
        ]
        if any(str(query).startswith("site:") for query in queries):
            output.append(
                DiscoveredURL(
                    url="https://unrelated.example/release",
                    title="Unrelated release",
                    snippet="A search engine ignored the site operator",
                    engine="fixture",
                    rank=2,
                    rrf_score=0.1,
                )
            )
        return output


class PrecisionPageFetcher:
    async def fetch(self, url):
        link = (
            "<a href='/downloads/'>Python releases and downloads</a>"
            if url.rstrip("/") == "https://www.python.org"
            else ""
        )
        return FetchedPage(
            requested_url=url,
            final_url=url,
            status=200,
            content_type="text/html; charset=utf-8",
            body=(
                "<html><head><title>Official Python release</title></head><body><main>"
                + link
                + "<p>This official Python page contains stable release information, "
                "documentation, downloads, support policy, and enough useful text for "
                "the bounded extraction and precision discovery integration test.</p>"
                "</main></body></html>"
            ).encode(),
            fetched_at=time.time(),
            elapsed_ms=4.0,
            headers={},
        )


class FailingDiscoverySession:
    def get(self, *args, **kwargs):
        raise OSError("fixture connection failed")


class CapturingFailingDiscoverySession:
    def __init__(self) -> None:
        self.url = ""

    def get(self, url, *args, **kwargs):
        self.url = url
        raise OSError("fixture connection failed")


class RealtimeSearchTests(unittest.TestCase):
    @staticmethod
    def _completion_from_queries(*queries):
        remaining = iter(queries)

        def complete(prompt, stops, max_tokens):
            query = next(remaining)
            return G1ICompletion(
                '<tool_call>{"name":"web_search","arguments":{"query":"'
                + query
                + '"}}',
                "</tool_call>",
                elapsed_ms=1.0,
            )

        return complete

    def test_searxng_params_use_language_and_bounded_engine_pool(self) -> None:
        self.assertEqual(
            searxng_search_params("新能源汽车政策", "latest")["language"],
            "zh-CN",
        )
        self.assertEqual(
            searxng_search_params("Python latest release", "latest")["language"],
            "en",
        )
        self.assertNotIn(
            "time_range", searxng_search_params("Python latest release", "latest")
        )
        self.assertNotIn("time_range", searxng_search_params("今日台风", "realtime"))
        self.assertNotIn("time_range", searxng_search_params("RWKV 是什么", "none"))
        params = searxng_search_params(
            "RWKV GitHub repository",
            "latest",
            ("general", "repos"),
            ("bing", "duckduckgo", "bing"),
        )
        self.assertEqual(params["engines"], "bing,duckduckgo")
        self.assertNotIn("categories", params)

    def test_english_query_compaction_puts_subject_before_chat_shell(self) -> None:
        self.assertEqual(
            compact_general_query(
                "What is the current stable Python release according to python.org?"
            ),
            "Python python.org current stable release",
        )
        self.assertEqual(
            compact_general_query(
                "Find the Federal Reserve's latest FOMC monetary policy statement."
            ),
            "Federal Reserve's FOMC latest monetary policy statement",
        )
        chinese = "国家统计局最新公布的中国CPI数据是什么？"
        self.assertEqual(compact_general_query(chinese), chinese)

    def test_shared_discovery_cache_reuses_general_channel_response(self) -> None:
        async def run():
            config = RealtimeSearchConfig(
                enabled=True,
                searxng_url="http://127.0.0.1:8888",
                searxng_engines=["bing", "duckduckgo"],
                fallback_engines=[],
            )
            cache = TTLByteCache(1024 * 1024)
            first = URLDiscovery(config, object(), cache=cache)
            second = URLDiscovery(config, object(), cache=cache)
            calls = 0

            async def discover(*args, **kwargs):
                nonlocal calls
                calls += 1
                return [
                    DiscoveredURL(
                        url="https://python.org/downloads/",
                        title="Download Python",
                        engine="bing",
                    )
                ]

            first._searxng = discover
            second._searxng = discover
            left = await first._discover_one(
                "Python latest release",
                "latest",
                source_channels=(),
            )
            right = await second._discover_one(
                "Python latest release",
                "latest",
                source_channels=("general",),
            )
            return calls, left, right

        calls, left, right = asyncio.run(run())
        self.assertEqual(calls, 1)
        self.assertEqual(left[0].url, right[0].url)

    def test_multi_channel_discovery_populates_shared_general_cache(self) -> None:
        async def run():
            config = RealtimeSearchConfig(
                enabled=True,
                searxng_url="http://127.0.0.1:8888",
                searxng_engines=["bing", "duckduckgo"],
                fallback_engines=[],
            )
            cache = TTLByteCache(1024 * 1024)
            enhanced = URLDiscovery(config, object(), cache=cache)
            legacy = URLDiscovery(config, object(), cache=cache)
            calls: list[tuple[str, tuple[str, ...]]] = []

            async def discover(query, freshness, diagnostics, source_channels=()):
                del freshness, diagnostics
                calls.append((query, tuple(source_channels)))
                channel = source_channels[0]
                return [
                    DiscoveredURL(
                        url=f"https://example.org/{channel}",
                        title=f"{channel} result",
                        engine="bing",
                    )
                ]

            enhanced._searxng = discover
            legacy._searxng = discover
            enhanced_results = await enhanced._discover_one(
                "RWKV official GitHub repository",
                "latest",
                source_channels=("general", "repos"),
            )
            legacy_results = await legacy._discover_one(
                "RWKV official GitHub repository",
                "latest",
                source_channels=(),
            )
            return calls, enhanced_results, legacy_results

        calls, enhanced_results, legacy_results = asyncio.run(run())
        self.assertEqual(
            [channel for _, channel in calls],
            [("general",), ("repos",)],
        )
        self.assertEqual(
            legacy_results[0].url,
            "https://example.org/general",
        )
        self.assertIn(
            legacy_results[0].url,
            {item.url for item in enhanced_results},
        )

    def test_bing_params_and_headers_follow_query_locale(self) -> None:
        self.assertEqual(bing_search_params("Python latest release")["mkt"], "en-US")
        self.assertEqual(bing_search_params("Python 最新版本")["mkt"], "zh-CN")
        self.assertEqual(bing_search_params("Python 最新版本")["adlt"], "moderate")
        self.assertEqual(
            bing_search_headers("Python 最新版本")["Accept-Language"],
            "zh-CN,zh;q=0.9",
        )

    def test_bing_base_url_is_configurable_without_changing_the_default(self) -> None:
        self.assertEqual(
            RealtimeSearchConfig().bing_base_url,
            "https://www.bing.com",
        )

        async def run():
            session = CapturingFailingDiscoverySession()
            discovery = URLDiscovery(
                RealtimeSearchConfig(
                    enabled=True,
                    searxng_url="",
                    fallback_engines=["bing"],
                    bing_base_url="https://cn.bing.com/",
                ),
                session,
            )
            await discovery.discover(
                ["Python latest release"],
                freshness="latest",
                max_candidates=5,
            )
            return session.url

        self.assertEqual(asyncio.run(run()), "https://cn.bing.com/search")

    def test_discovery_diagnostics_preserve_searxng_connection_errors(self) -> None:
        async def run():
            config = RealtimeSearchConfig(
                enabled=True,
                searxng_url="http://127.0.0.1:8888",
                searxng_engines=["dogpile"],
                fallback_engines=[],
            )
            diagnostics = []
            results = await URLDiscovery(config, FailingDiscoverySession()).discover(
                ["python release"],
                freshness="latest",
                max_candidates=5,
                diagnostics=diagnostics,
            )
            return results, diagnostics

        results, diagnostics = asyncio.run(run())
        self.assertEqual(results, [])
        self.assertEqual(diagnostics[0]["engine"], "searxng")
        self.assertEqual(diagnostics[0]["error_type"], "OSError")

    def test_multiple_searxng_engines_fan_out_and_rrf_merge(self) -> None:
        async def run():
            config = RealtimeSearchConfig(
                enabled=True,
                searxng_url="http://127.0.0.1:8888",
                searxng_engines=["dogpile", "naver"],
                fallback_engines=[],
            )
            discovery = URLDiscovery(config, object())
            calls = []

            async def request(
                query,
                freshness,
                diagnostics,
                source_channels,
                *,
                engines,
                diagnostic_engine,
            ):
                calls.append((tuple(engines), diagnostic_engine))
                engine = engines[0]
                if engine == "dogpile":
                    return [
                        DiscoveredURL(
                            url="https://python.org/downloads/",
                            title="Python downloads",
                            engine=engine,
                        ),
                        DiscoveredURL(
                            url="https://example.com/python",
                            title="Python article",
                            engine=engine,
                        ),
                    ]
                return [
                    DiscoveredURL(
                        url="https://python.org/downloads",
                        title="Download the latest Python release",
                        snippet="Official stable releases",
                        engine=engine,
                    )
                ]

            discovery._searxng_request = request
            results = await discovery._searxng(
                "Python latest release", "latest"
            )
            return calls, results

        calls, results = asyncio.run(run())
        self.assertEqual(
            calls,
            [
                (("dogpile",), "searxng:dogpile"),
                (("naver",), "searxng:naver"),
            ],
        )
        self.assertEqual(results[0].url, "https://python.org/downloads")
        self.assertEqual(results[0].engines, ["dogpile", "naver"])
        self.assertGreater(results[0].rrf_score, results[1].rrf_score)

    def test_searxng_language_lanes_are_additive_and_deduplicated(self) -> None:
        config = RealtimeSearchConfig(
            searxng_engines=["dogpile", "naver"],
            searxng_language_engines={
                "zh": ["baidu", "dogpile"],
                "default": ["yandex"],
            },
        )

        self.assertEqual(
            searxng_engines_for_query(config, "Python latest release"),
            ("dogpile", "naver", "yandex"),
        )
        self.assertEqual(
            searxng_engines_for_query(config, "Python 最新版本"),
            ("dogpile", "naver", "baidu"),
        )
        self.assertEqual(
            searxng_engines_for_query(config, "Python 最新リリース"),
            ("dogpile", "naver", "yandex"),
        )

    def test_searxng_language_lane_participates_in_fanout(self) -> None:
        async def run():
            config = RealtimeSearchConfig(
                enabled=True,
                searxng_url="http://127.0.0.1:8888",
                searxng_engines=["dogpile", "naver"],
                searxng_language_engines={"zh": ["baidu"]},
                fallback_engines=[],
            )
            discovery = URLDiscovery(config, object())
            calls = []

            async def request(
                query,
                freshness,
                diagnostics,
                source_channels,
                *,
                engines,
                diagnostic_engine,
            ):
                calls.append((tuple(engines), diagnostic_engine))
                return []

            discovery._searxng_request = request
            await discovery._searxng("Python 最新版本", "latest")
            return calls

        self.assertEqual(
            asyncio.run(run()),
            [
                (("dogpile",), "searxng:dogpile"),
                (("naver",), "searxng:naver"),
                (("baidu",), "searxng:baidu"),
            ],
        )

    def test_searxng_fanout_keeps_healthy_engine_when_peer_fails(self) -> None:
        async def run():
            config = RealtimeSearchConfig(
                enabled=True,
                searxng_url="http://127.0.0.1:8888",
                searxng_engines=["dogpile", "naver"],
                fallback_engines=[],
            )
            discovery = URLDiscovery(config, object())

            async def request(
                query,
                freshness,
                diagnostics,
                source_channels,
                *,
                engines,
                diagnostic_engine,
            ):
                if engines == ("naver",):
                    raise TimeoutError("fixture timeout")
                return [
                    DiscoveredURL(
                        url="https://python.org/downloads/",
                        title="Python downloads",
                        engine=engines[0],
                    )
                ]

            discovery._searxng_request = request
            return await discovery._searxng("Python latest release", "latest")

        results = asyncio.run(run())
        self.assertEqual([item.url for item in results], ["https://python.org/downloads"])
        self.assertEqual(results[0].engines, ["dogpile"])

    def test_searxng_engine_lane_cache_is_reused_across_pool_configs(self) -> None:
        class Stream:
            async def iter_chunked(self, _size):
                yield (
                    b'{"results":[{"url":"https://python.org/downloads/",'
                    b'"title":"Python downloads","engine":"dogpile"}]}'
                )

        class Response:
            status = 200
            headers = {}
            content = Stream()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        class Session:
            def __init__(self):
                self.calls = 0

            async def get(self, *_args, **_kwargs):
                self.calls += 1
                return Response()

        async def run():
            cache = TTLByteCache(1024 * 1024)
            session = Session()
            first = URLDiscovery(
                RealtimeSearchConfig(
                    enabled=True,
                    searxng_engines=["dogpile"],
                ),
                session,
                cache=cache,
            )
            second = URLDiscovery(
                RealtimeSearchConfig(
                    enabled=True,
                    searxng_engines=["dogpile", "naver"],
                ),
                session,
                cache=cache,
            )
            left = await first._searxng_request(
                "Python latest release",
                "latest",
                [],
                (),
                engines=("dogpile",),
                diagnostic_engine="searxng",
            )
            right = await second._searxng_request(
                "Python latest release",
                "latest",
                [],
                (),
                engines=("dogpile",),
                diagnostic_engine="searxng:dogpile",
            )
            return session.calls, left, right

        calls, left, right = asyncio.run(run())
        self.assertEqual(calls, 1)
        self.assertEqual(left[0].url, right[0].url)

    def test_searxng_lane_retries_one_transient_read_failure(self) -> None:
        class Stream:
            async def iter_chunked(self, _size):
                yield (
                    b'{"results":[{"url":"https://python.org/downloads/",'
                    b'"title":"Python downloads","engine":"dogpile"}]}'
                )

        class Response:
            status = 200
            headers = {}
            content = Stream()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        class FlakySession:
            def __init__(self):
                self.calls = 0

            async def get(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("transient local SearXNG timeout")
                return Response()

        async def run():
            session = FlakySession()
            discovery = URLDiscovery(
                RealtimeSearchConfig(enabled=True, searxng_engines=["dogpile"]),
                session,
            )
            diagnostics = []
            results = await discovery._searxng_request(
                "Python latest release",
                "latest",
                diagnostics,
                (),
                engines=("dogpile",),
                diagnostic_engine="searxng",
            )
            return session.calls, diagnostics, results

        calls, diagnostics, results = asyncio.run(run())
        self.assertEqual(calls, 2)
        self.assertEqual(diagnostics, [])
        self.assertEqual(results[0].url, "https://python.org/downloads")

    def test_force_ipv4_configures_the_shared_aiohttp_connector(self) -> None:
        async def run() -> None:
            engine = RealtimeSearchEngine(
                RealtimeSearchConfig(enabled=True, force_ipv4=True)
            )
            await engine._initialize()
            try:
                self.assertEqual(engine._session.connector._family, socket.AF_INET)
                self.assertEqual(engine.status()["network_family"], "ipv4")
            finally:
                await engine._session.close()

        asyncio.run(run())

    def test_discovery_diagnostics_preserve_fallback_connection_errors(self) -> None:
        async def run():
            config = RealtimeSearchConfig(
                enabled=True,
                searxng_url="",
                fallback_engines=["bing"],
            )
            diagnostics = []
            results = await URLDiscovery(config, FailingDiscoverySession()).discover(
                ["python release"],
                freshness="latest",
                max_candidates=5,
                diagnostics=diagnostics,
            )
            return results, diagnostics

        results, diagnostics = asyncio.run(run())
        self.assertEqual(results, [])
        self.assertEqual(diagnostics[0]["engine"], "bing")
        self.assertEqual(diagnostics[0]["error_type"], "OSError")
        self.assertIn("fixture connection failed", diagnostics[0]["message"])

    def test_benchmark_visibility_includes_candidates_fetch_outcomes_and_stats(
        self,
    ) -> None:
        config = RealtimeSearchConfig(
            enabled=True,
            fast_max_queries=1,
            fast_max_candidates=5,
            fast_max_fetch_pages=2,
            fast_deadline_seconds=2.0,
        )
        engine = RealtimeSearchEngine(config)
        engine._discovery = FakeDiscovery()
        engine._fetcher = FakePageFetcher()
        events = queue.Queue()

        asyncio.run(
            engine._search(
                "official release",
                ["official release"],
                "latest",
                "single",
                None,
                events,
                True,
            )
        )
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        discovery = next(
            item for item in emitted if item["type"] == "discovery_progress"
        )
        fetch = next(item for item in emitted if item["type"] == "fetch_progress")
        result = next(item for item in emitted if item["type"] == "realtime_result")
        self.assertEqual(discovery["progress"]["candidates"][0]["position"], 1)
        self.assertEqual(discovery["progress"]["errors"], [])
        self.assertEqual(discovery["progress"]["candidates"][0]["rank"], 2)
        self.assertEqual(fetch["progress"]["fetch"]["status"], "succeeded")
        self.assertEqual(result["stats"]["attempted"], 1)
        self.assertEqual(result["stats"]["fetched"], 1)
        self.assertEqual(result["stats"]["failed"], 0)
        self.assertGreaterEqual(result["stats"]["fetch_success_rate"], 1.0)

    def test_seed_urls_are_bounded_direct_candidates_and_reach_fetch(self) -> None:
        config = RealtimeSearchConfig(
            enabled=True,
            candidate_admission_enabled=True,
            fast_max_queries=1,
            fast_max_candidates=5,
            fast_max_fetch_pages=2,
            fast_deadline_seconds=2.0,
        )
        engine = RealtimeSearchEngine(config)
        engine._discovery = FakeDiscovery()
        fetcher = FakePageFetcher()
        engine._fetcher = fetcher
        events = queue.Queue()

        asyncio.run(
            engine._search(
                "official release",
                ["official release site:example.org"],
                "latest",
                "single",
                None,
                events,
                True,
                seed_urls=(
                    "https://www.example.org/",
                    "https://www.example.org/#duplicate",
                    "file:///etc/passwd",
                ),
            )
        )
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        discovery = next(
            item["progress"] for item in emitted if item["type"] == "discovery_progress"
        )
        result = next(item for item in emitted if item["type"] == "realtime_result")

        self.assertEqual(discovery["seed_candidate_count"], 1)
        direct = next(
            item for item in discovery["candidates"] if item["engine"] == "direct"
        )
        self.assertEqual(direct["url"], "https://www.example.org/")
        self.assertIn("seed_url", direct["discovery_stages"])
        self.assertEqual(fetcher.calls[0], "https://www.example.org/")
        self.assertEqual(result["stats"]["seed_candidates"], 1)

    def test_c_feedback_runs_two_discoveries_then_one_shared_fetch_stage(self) -> None:
        config = RealtimeSearchConfig(
            enabled=True,
            fast_max_queries=2,
            fast_max_candidates=5,
            fast_max_fetch_pages=2,
            fast_deadline_seconds=2.0,
        )
        planner = CFeedbackPlanner(
            self._completion_from_queries(
                "Python latest stable release official",
                "Python official downloads release page",
            )
        )
        engine = RealtimeSearchEngine(config, feedback_planner=planner)
        discovery_fixture = FeedbackDiscoveryFixture()
        fetcher = FakePageFetcher()
        engine._discovery = discovery_fixture
        engine._fetcher = fetcher
        events = queue.Queue()
        asyncio.run(
            engine._search(
                "Python当前最新稳定版本是什么？请以官网为准。",
                ["Python 当前版本"],
                "latest",
                "single",
                None,
                events,
                True,
            )
        )
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        discovery = next(
            item["progress"] for item in emitted if item["type"] == "discovery_progress"
        )
        result = next(item for item in emitted if item["type"] == "realtime_result")
        self.assertEqual(len(discovery_fixture.calls), 2)
        self.assertEqual(discovery_fixture.calls[0], ["Python latest stable release official"])
        self.assertTrue(discovery["feedback_query_executed"])
        self.assertEqual(discovery["discovery_request_count"], 2)
        self.assertEqual(len(discovery["model_search_plans"]), 2)
        self.assertNotIn("reasoning", discovery["model_search_plans"][1])
        self.assertTrue(
            any(
                "model_feedback" in item["discovery_stages"]
                for item in discovery["candidates"]
            )
        )
        self.assertEqual(len(fetcher.calls), result["stats"]["attempted"])
        self.assertLessEqual(len(fetcher.calls), config.fast_max_fetch_pages)

    def test_c_feedback_gate_can_skip_the_second_discovery(self) -> None:
        config = RealtimeSearchConfig(
            enabled=True,
            fast_max_queries=2,
            fast_max_candidates=5,
            fast_max_fetch_pages=2,
            fast_deadline_seconds=2.0,
        )
        planner = CFeedbackPlanner(
            self._completion_from_queries("Python latest stable release official")
        )
        engine = RealtimeSearchEngine(config, feedback_planner=planner)
        fixture = FeedbackDiscoveryFixture(satisfied=True)
        engine._discovery = fixture
        engine._fetcher = FakePageFetcher()
        events = queue.Queue()
        asyncio.run(
            engine._search(
                "Python最新稳定版本，以官网为准",
                ["Python 当前版本"],
                "latest",
                "single",
                None,
                events,
                True,
            )
        )
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        discovery = next(
            item["progress"] for item in emitted if item["type"] == "discovery_progress"
        )
        self.assertEqual(len(fixture.calls), 1)
        self.assertFalse(discovery["feedback_query_executed"])
        self.assertEqual(
            discovery["model_search_plans"][1]["stop_reason"],
            "gate_not_triggered",
        )

    def test_c_feedback_invalid_duplicate_does_not_run_q2(self) -> None:
        config = RealtimeSearchConfig(
            enabled=True,
            fast_max_queries=2,
            fast_max_candidates=5,
            fast_max_fetch_pages=1,
            fast_deadline_seconds=2.0,
        )
        planner = CFeedbackPlanner(
            self._completion_from_queries(
                "Python latest stable release official",
                "Python latest stable release official",
            )
        )
        engine = RealtimeSearchEngine(config, feedback_planner=planner)
        fixture = FeedbackDiscoveryFixture()
        engine._discovery = fixture
        engine._fetcher = FakePageFetcher()
        events = queue.Queue()
        asyncio.run(
            engine._search(
                "Python最新稳定版本，以官网为准",
                ["Python 当前版本"],
                "latest",
                "single",
                None,
                events,
                True,
            )
        )
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        discovery = next(
            item["progress"] for item in emitted if item["type"] == "discovery_progress"
        )
        self.assertEqual(len(fixture.calls), 1)
        self.assertFalse(discovery["feedback_query_executed"])
        self.assertIn(
            "duplicate_query",
            discovery["model_search_plans"][1]["validation"]["reasons"],
        )

    def test_c_feedback_q1_failure_falls_back_without_q2(self) -> None:
        def invalid_complete(prompt, stops, max_tokens):
            return G1ICompletion("not a tool call", "</s>", elapsed_ms=1.0)

        config = RealtimeSearchConfig(
            enabled=True,
            fast_max_queries=1,
            fast_max_candidates=5,
            fast_max_fetch_pages=1,
            fast_deadline_seconds=2.0,
        )
        engine = RealtimeSearchEngine(
            config,
            feedback_planner=CFeedbackPlanner(invalid_complete),
        )
        fixture = FeedbackDiscoveryFixture()
        engine._discovery = fixture
        engine._fetcher = FakePageFetcher()
        events = queue.Queue()
        asyncio.run(
            engine._search(
                "Python最新稳定版本",
                ["fallback route query"],
                "latest",
                "single",
                None,
                events,
                True,
            )
        )
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        discovery = next(
            item["progress"] for item in emitted if item["type"] == "discovery_progress"
        )
        self.assertEqual(fixture.calls, [["fallback route query"]])
        self.assertFalse(discovery["feedback_query_executed"])
        self.assertEqual(len(discovery["model_search_plans"]), 1)

    def test_c_feedback_planning_timeout_fails_open_to_route_query(self) -> None:
        def slow_complete(prompt, stops, max_tokens):
            time.sleep(0.2)
            return G1ICompletion(
                '<tool_call>{"name":"web_search","arguments":'
                '{"query":"Python latest release official"}}',
                "</tool_call>",
            )

        planner = CFeedbackPlanner(slow_complete, timeout_seconds=0.1)
        config = RealtimeSearchConfig(
            enabled=True,
            fast_max_queries=1,
            fast_max_candidates=5,
            fast_max_fetch_pages=1,
            fast_deadline_seconds=1.0,
        )
        engine = RealtimeSearchEngine(config, feedback_planner=planner)
        fixture = FeedbackDiscoveryFixture()
        engine._discovery = fixture
        engine._fetcher = FakePageFetcher()
        events = queue.Queue()
        asyncio.run(
            engine._search(
                "Python最新稳定版本",
                ["fallback route query"],
                "latest",
                "single",
                None,
                events,
                True,
            )
        )
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        discovery = next(
            item["progress"] for item in emitted if item["type"] == "discovery_progress"
        )
        self.assertEqual(fixture.calls[0], ["fallback route query"])
        self.assertEqual(
            discovery["model_planning_errors"][0]["error_type"], "TimeoutError"
        )
        self.assertFalse(discovery["feedback_query_executed"])

    def test_c_feedback_deadline_prevents_discovery_after_slow_planning(self) -> None:
        def slow_complete(prompt, stops, max_tokens):
            time.sleep(0.2)
            return G1ICompletion(
                '<tool_call>{"name":"web_search","arguments":'
                '{"query":"Python latest release official"}}',
                "</tool_call>",
            )

        planner = CFeedbackPlanner(slow_complete, timeout_seconds=0.3)
        config = RealtimeSearchConfig(
            enabled=True,
            fast_max_queries=1,
            fast_max_candidates=5,
            fast_max_fetch_pages=1,
            fast_deadline_seconds=0.1,
        )
        engine = RealtimeSearchEngine(config, feedback_planner=planner)
        fixture = FeedbackDiscoveryFixture()
        engine._discovery = fixture
        engine._fetcher = FakePageFetcher()
        events = queue.Queue()
        asyncio.run(
            engine._search(
                "Python最新稳定版本",
                ["fallback route query"],
                "latest",
                "single",
                None,
                events,
                True,
            )
        )
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        discovery = next(
            item["progress"] for item in emitted if item["type"] == "discovery_progress"
        )
        self.assertEqual(fixture.calls, [])
        self.assertEqual(discovery["discovery_request_count"], 0)
        self.assertIn(
            discovery["errors"][0]["error_type"],
            {"DeadlineExceeded", "TimeoutError"},
        )

    def test_c_feedback_disables_extra_discovery_extensions(self) -> None:
        config = RealtimeSearchConfig(
            enabled=True,
            source_channels_enabled=True,
            domain_pivot_enabled=True,
            fast_max_queries=2,
            fast_max_candidates=5,
            fast_max_fetch_pages=1,
            fast_deadline_seconds=2.0,
        )
        planner = CFeedbackPlanner(
            self._completion_from_queries(
                "Python latest stable release official",
                "Python official downloads release page",
            )
        )
        engine = RealtimeSearchEngine(config, feedback_planner=planner)
        fixture = FeedbackDiscoveryFixture()
        engine._discovery = fixture
        engine._fetcher = FakePageFetcher()
        events = queue.Queue()
        asyncio.run(
            engine._search(
                "Python最新稳定版本，以官网为准",
                ["fallback route query"],
                "latest",
                "single",
                None,
                events,
                True,
            )
        )
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        discovery = next(
            item["progress"] for item in emitted if item["type"] == "discovery_progress"
        )
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual(discovery["source_channels"], [])
        self.assertEqual(discovery["pivot_queries"], [])
        self.assertLessEqual(discovery["discovery_request_count"], 2)

    def test_c_feedback_cancellation_prevents_model_and_feedback_fetch(self) -> None:
        calls = []

        def complete(prompt, stops, max_tokens):
            calls.append(prompt)
            raise AssertionError("cancelled search must not call the model")

        config = RealtimeSearchConfig(
            enabled=True,
            fast_max_queries=1,
            fast_max_candidates=5,
            fast_max_fetch_pages=1,
            fast_deadline_seconds=1.0,
        )
        engine = RealtimeSearchEngine(
            config,
            feedback_planner=CFeedbackPlanner(complete),
        )
        fixture = FeedbackDiscoveryFixture()
        fetcher = FakePageFetcher()
        engine._discovery = fixture
        engine._fetcher = fetcher
        events = queue.Queue()
        cancelled = threading.Event()
        cancelled.set()
        asyncio.run(
            engine._search(
                "Python最新稳定版本",
                ["fallback route query"],
                "latest",
                "single",
                cancelled,
                events,
                True,
            )
        )
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        result = next(item for item in emitted if item["type"] == "realtime_result")
        self.assertEqual(calls, [])
        self.assertEqual(len(fixture.calls), 1)
        self.assertEqual(fetcher.calls, [])
        self.assertFalse(result["stats"]["feedback_query_executed"])

    def test_candidate_admission_is_visible_without_changing_the_default(self) -> None:
        config = RealtimeSearchConfig(
            enabled=True,
            candidate_admission_enabled=True,
            fast_max_queries=1,
            fast_max_candidates=5,
            fast_max_fetch_pages=2,
            fast_deadline_seconds=2.0,
        )
        engine = RealtimeSearchEngine(config)
        engine._discovery = FakeDiscovery()
        engine._fetcher = FakePageFetcher()
        events = queue.Queue()
        asyncio.run(
            engine._search(
                "official release",
                ["official release"],
                "latest",
                "single",
                None,
                events,
                True,
            )
        )
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        progress = next(
            item["progress"] for item in emitted if item["type"] == "discovery_progress"
        )
        self.assertTrue(progress["candidate_admission_enabled"])
        self.assertEqual(progress["raw_candidate_count"], 1)
        self.assertEqual(progress["rejected_candidate_count"], 0)
        self.assertGreater(progress["candidates"][0]["candidate_score"], 0)

    def test_precision_discovery_runs_one_bounded_pivot_and_one_hop_stage(self) -> None:
        config = RealtimeSearchConfig(
            enabled=True,
            candidate_admission_enabled=True,
            source_channels_enabled=True,
            domain_pivot_enabled=True,
            domain_pivot_max_domains=2,
            domain_pivot_max_candidates=5,
            one_hop_link_expansion_enabled=True,
            one_hop_max_links=8,
            fast_max_queries=1,
            fast_max_candidates=10,
            fast_max_fetch_pages=3,
            fast_deadline_seconds=3.0,
        )
        engine = RealtimeSearchEngine(config)
        discovery_fixture = PrecisionDiscoveryFixture()
        engine._discovery = discovery_fixture
        engine._fetcher = PrecisionPageFetcher()
        events = queue.Queue()
        asyncio.run(
            engine._search(
                "What is the latest official Python release?",
                ["Python latest stable release official"],
                "latest",
                "single",
                None,
                events,
                True,
            )
        )
        emitted = []
        while not events.empty():
            emitted.append(events.get_nowait())
        discovery = next(
            item for item in emitted if item["type"] == "discovery_progress"
        )
        enrichment = next(
            item for item in emitted if item["type"] == "discovery_enrichment"
        )
        result = next(item for item in emitted if item["type"] == "realtime_result")
        self.assertEqual(len(discovery_fixture.calls), 2)
        self.assertEqual(discovery_fixture.calls[0][1], ("general",))
        self.assertTrue(discovery_fixture.calls[1][0][0].startswith("site:python.org "))
        self.assertEqual(discovery["progress"]["pivot_domains"], ["python.org"])
        self.assertNotIn(
            "unrelated.example",
            " ".join(item["url"] for item in discovery["progress"]["candidates"]),
        )
        self.assertEqual(discovery["progress"]["discovery_request_count"], 2)
        self.assertEqual(enrichment["progress"]["new_candidate_count"], 1)
        self.assertEqual(
            enrichment["progress"]["candidates"][0]["url"],
            discovery["progress"]["candidates"][0]["url"],
        )
        self.assertEqual(
            enrichment["progress"]["new_candidates"][0]["discovery_stage"],
            "one_hop_link",
        )
        self.assertEqual(result["stats"]["one_hop_candidates"], 1)
        self.assertLessEqual(result["stats"]["discovery_request_count"], 3)

    def test_financial_primary_sources_are_classified_as_official(self) -> None:
        for url in (
            "https://www.cninfo.com.cn/new/disclosure",
            "https://www.sse.com.cn/disclosure/listedinfo/announcement/",
            "https://www1.hkexnews.hk/listedco/listconews/",
            "https://www.sec.gov/edgar/search/",
        ):
            source_type, authority = classify_source(url)
            self.assertEqual(source_type, "company_filing", url)
            self.assertGreaterEqual(authority, 0.98, url)

    def test_bing_html_result_parser(self) -> None:
        results = parse_search_html(
            """<ol><li class='b_algo'><h2><a href='https://example.com/a?utm_source=x'>
            Example result</a></h2><p>Useful result summary.</p></li></ol>""",
            "bing",
        )
        self.assertEqual(results[0].url, "https://example.com/a")
        self.assertEqual(results[0].title, "Example result")
        self.assertEqual(results[0].snippet, "Useful result summary.")

    def test_discovery_records_query_and_engine_provenance(self) -> None:
        class ProvenanceDiscovery(URLDiscovery):
            async def _discover_one(
                self, query, freshness, diagnostics=None, source_channels=()
            ):
                rank = 2 if query == "secondary" else 1
                return [
                    DiscoveredURL(
                        url="https://example.com/release",
                        title="Release",
                        engine="bing" if rank == 1 else "mwmbl",
                        engines=["bing"] if rank == 1 else ["mwmbl"],
                        positions=[rank],
                    )
                ]

        config = RealtimeSearchConfig(enabled=True)
        results = asyncio.run(
            ProvenanceDiscovery(config, object()).discover(
                ["primary", "secondary"], freshness="latest", max_candidates=5
            )
        )
        self.assertEqual(results[0].matched_queries, ["primary", "secondary"])
        self.assertEqual(results[0].query_positions, {"primary": 1, "secondary": 1})
        self.assertEqual(results[0].engines, ["bing", "mwmbl"])

    def test_specialist_source_channel_uses_a_separate_source_native_query(
        self,
    ) -> None:
        class ChannelDiscovery(URLDiscovery):
            def __init__(self, config, session):
                super().__init__(config, session)
                self.calls = []

            async def _searxng(
                self,
                query,
                freshness,
                diagnostics=None,
                source_channels=(),
            ):
                self.calls.append((query, tuple(source_channels)))
                channel = source_channels[0]
                return [
                    DiscoveredURL(
                        url=f"https://example.com/{channel}",
                        title=channel,
                        engine=channel,
                    )
                ]

        discovery = ChannelDiscovery(
            RealtimeSearchConfig(enabled=True, searxng_url="http://127.0.0.1:8888"),
            object(),
        )
        results = asyncio.run(
            discovery.discover(
                ["llama.cpp latest release GitHub"],
                freshness="latest",
                max_candidates=5,
                source_channels=("general", "repos"),
            )
        )
        self.assertEqual(
            discovery.calls,
            [
                ("llama.cpp latest release GitHub", ("general",)),
                ("llama.cpp latest release GitHub", ("repos",)),
            ],
        )
        self.assertEqual({item.discovery_stage for item in results}, {"initial"})
        by_url = {item.url: item for item in results}
        self.assertEqual(
            by_url["https://example.com/general"].source_channels, ["general"]
        )
        self.assertEqual(by_url["https://example.com/repos"].source_channels, ["repos"])

    def test_searxng_result_parser_preserves_native_merge_signals(self) -> None:
        results = parse_searxng_results(
            {
                "results": [
                    {
                        "url": "https://python.org/downloads/?utm_source=test",
                        "title": "Download Python",
                        "content": "Official releases",
                        "engine": "bing",
                        "engines": ["bing", "mwmbl"],
                        "positions": [1, 3],
                        "score": 2.5,
                    }
                ]
            }
        )
        self.assertEqual(results[0].url, "https://python.org/downloads")
        self.assertEqual(results[0].engine_score, 2.5)
        self.assertEqual(results[0].engines, ["bing", "mwmbl"])
        self.assertEqual(results[0].positions, [1, 3])

    def test_fetcher_uses_one_get_without_head_or_robots(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), RealtimeFixtureHandler)
        RealtimeFixtureHandler.requests = []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        async def run() -> FetchedPage:
            import aiohttp

            config = RealtimeSearchConfig(
                enabled=True,
                allow_private_networks=True,
                page_timeout_seconds=2.0,
            )
            async with aiohttp.ClientSession(auto_decompress=False) as session:
                return await AsyncPageFetcher(config, session).fetch(
                    f"http://127.0.0.1:{server.server_port}/article"
                )

        try:
            page = asyncio.run(run())
            self.assertIn("实时搜索", page.body.decode("utf-8"))
            self.assertEqual(RealtimeFixtureHandler.requests, [("GET", "/article")])
        finally:
            server.shutdown()
            server.server_close()

    def test_private_target_is_blocked_by_default(self) -> None:
        async def run() -> None:
            import aiohttp

            config = RealtimeSearchConfig(enabled=True, allow_private_networks=False)
            async with aiohttp.ClientSession(auto_decompress=False) as session:
                fetcher = AsyncPageFetcher(config, session)
                with self.assertRaises(FetchError):
                    await fetcher.fetch("http://127.0.0.1/")

        asyncio.run(run())

    def test_fetcher_cancellation_retrieves_the_inner_task(self) -> None:
        async def run() -> None:
            started = asyncio.Event()
            finished = asyncio.Event()

            class BlockingFetcher(AsyncPageFetcher):
                async def _fetch_redirects(self, requested_url: str) -> FetchedPage:
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        finished.set()
                    raise AssertionError("unreachable")

            fetcher = BlockingFetcher(
                RealtimeSearchConfig(page_timeout_seconds=5.0),
                object(),
            )
            task = asyncio.create_task(fetcher.fetch("https://example.com/page"))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(finished.is_set())

        asyncio.run(run())

    def test_fetcher_timeout_retrieves_the_inner_task(self) -> None:
        async def run() -> None:
            finished = asyncio.Event()

            class BlockingFetcher(AsyncPageFetcher):
                async def _fetch_redirects(self, requested_url: str) -> FetchedPage:
                    try:
                        await asyncio.Event().wait()
                    finally:
                        finished.set()
                    raise AssertionError("unreachable")

            fetcher = BlockingFetcher(
                RealtimeSearchConfig(page_timeout_seconds=0.01),
                object(),
            )
            with self.assertRaises(asyncio.TimeoutError):
                await fetcher.fetch("https://example.com/page")
            self.assertTrue(finished.is_set())

        asyncio.run(run())

    def test_extraction_ranking_and_service_event_bridge(self) -> None:
        page = FetchedPage(
            requested_url="https://docs.example/article",
            final_url="https://docs.example/article",
            status=200,
            content_type="text/html; charset=utf-8",
            body=(
                "<html><head><title>实时搜索优化</title></head><body><main>"
                "<p>实时搜索通过并发抓取、正文抽取、来源排序和证据去重提升回答质量。"
                "它不依赖模型中的过期知识，并为回答保留可核查来源。"
                "系统还会限制每页字节数、总并发、单域名并发和请求超时，"
                "在证据充分时提前停止抓取，从而降低本地服务器的资源消耗。</p>"
                "</main></body></html>"
            ).encode(),
            fetched_at=time.time(),
            elapsed_ms=8.0,
            headers={},
        )
        document = extract_page(page)
        self.assertIsNotNone(document)
        ranked = rank_documents(
            "实时搜索如何优化",
            [document],
            freshness_mode="latest",
            limit=3,
        )
        self.assertEqual(to_search_results("实时搜索", ranked)[0].url, page.final_url)

        with tempfile.TemporaryDirectory() as tmp:
            engine = FakeRealtimeEngine()
            service = SearchService(
                SearchDatabase(Path(tmp) / "search.db"), realtime_engine=engine
            )
            events = list(
                service.ask_events(
                    "搜索一下实时抓取方案",
                    search_mode="always",
                    source_scope="web",
                )
            )
        kinds = [event["type"] for event in events]
        self.assertTrue(engine.called)
        self.assertIn("discovery_progress", kinds)
        self.assertIn("fetch_progress", kinds)
        evidence = next(event for event in events if event["type"] == "evidence")
        self.assertEqual(
            evidence["evidence"][0]["url"], "https://docs.example/realtime"
        )

    def test_semantic_reranker_handles_named_entity_relevance(self) -> None:
        now = time.time()
        wrong = RealtimeDocument(
            url="https://dictionary.example/what",
            title="什么的意思",
            text="什么是一个用于提问的中文词语。",
            published_at=None,
            fetched_at=now,
            source_type="web",
            authority=0.7,
            extraction_quality=1.0,
            rrf_score=0.2,
            simhash="0000000000000001",
        )
        right = RealtimeDocument(
            url="https://www.python.org/",
            title="Welcome to Python.org",
            text="Python is a programming language with official documentation.",
            published_at=None,
            fetched_at=now,
            source_type="official_docs",
            authority=1.0,
            extraction_quality=1.0,
            rrf_score=0.05,
            simhash="1000000000000000",
        )
        ranked = rank_documents(
            "什么是 Python？请给出来源",
            [wrong, right],
            freshness_mode="stable",
            limit=5,
            scorer=type(
                "EntityScorer",
                (),
                {
                    "model_name": "test",
                    "score": lambda self, query, documents: [
                        10.0 if "python" in value.casefold() else -10.0
                        for value in documents
                    ],
                },
            )(),
        )
        self.assertEqual([item.url for item in ranked], ["https://www.python.org/"])

    def test_semantic_reranker_prefers_primary_evidence_without_vertical_rules(self) -> None:
        now = time.time()
        portal = RealtimeDocument(
            url="https://portal.example/",
            title="综合新闻首页",
            text="教育 体育 国际 新闻 财经 股票 娱乐等频道入口。",
            published_at=None,
            fetched_at=now,
            source_type="web",
            authority=0.8,
            extraction_quality=1.0,
            rrf_score=0.5,
            simhash="0000000000000001",
        )
        filing = RealtimeDocument(
            url="https://www.sse.com.cn/disclosure/announcement/",
            title="上市公司最新公告",
            text="上海证券交易所上市公司公告与定期报告。",
            published_at=None,
            fetched_at=now,
            source_type="company_filing",
            authority=0.98,
            extraction_quality=0.8,
            rrf_score=0.05,
            simhash="1000000000000000",
        )
        unrelated_government = RealtimeDocument(
            url="https://www.gov.cn/holiday/",
            title="国务院办公厅关于节假日安排的通知",
            text="公布全年法定节假日放假调休日期。",
            published_at=None,
            fetched_at=now,
            source_type="regulator",
            authority=0.98,
            extraction_quality=1.0,
            rrf_score=0.8,
            simhash="0100000000000000",
        )
        ranked = rank_documents(
            "最新的股票",
            [portal, unrelated_government, filing],
            freshness_mode="realtime",
            limit=5,
            scorer=type(
                "PrimaryEvidenceScorer",
                (),
                {
                    "model_name": "test",
                    "score": lambda self, query, documents: [
                        10.0
                        if "交易所" in value or "上市公司最新公告" in value
                        else -10.0
                        for value in documents
                    ],
                },
            )(),
        )
        self.assertEqual([item.url for item in ranked], [filing.url])

    def test_snippet_fallback_keeps_blocked_candidate_as_labeled_evidence(self) -> None:
        class BlockedFetcher:
            async def fetch(self, url):
                raise FetchError("HTTP 403")

        engine = RealtimeSearchEngine(
            RealtimeSearchConfig(
                snippet_fallback_enabled=True,
                snippet_fallback_min_chars=40,
            )
        )
        engine._fetcher = BlockedFetcher()
        candidate = DiscoveredURL(
            url="https://docs.example/releases/current",
            title="Official current release notes",
            snippet=(
                "The current stable release includes documented compatibility, "
                "security, and migration changes from the official project."
            ),
            engine="dogpile",
            rank=1,
            rrf_score=0.2,
            candidate_score=0.7,
            score_components={"entity_coverage": 1.0},
        )

        outcome = asyncio.run(engine._fetch_extract_outcome(candidate))

        self.assertIsNotNone(outcome.document)
        self.assertEqual(outcome.retrieval_mode, "search_snippet_fallback")
        self.assertEqual(outcome.to_debug_dict()["status"], "fallback")
        assert outcome.document is not None
        self.assertEqual(outcome.document.retrieval_mode, "search_snippet_fallback")
        result = to_search_results(
            "current release", [outcome.document]
        )[0]
        self.assertEqual(result.score_components["snippet_fallback"], 1.0)

    def test_snippet_fallback_does_not_preserve_not_found_url(self) -> None:
        class MissingFetcher:
            async def fetch(self, url):
                raise FetchError("HTTP 404")

        engine = RealtimeSearchEngine(
            RealtimeSearchConfig(snippet_fallback_enabled=True)
        )
        engine._fetcher = MissingFetcher()
        candidate = DiscoveredURL(
            url="https://docs.example/releases/missing",
            title="Missing release notes",
            snippet="A sufficiently long search result snippet that should not rescue a missing URL. "
            * 2,
            candidate_score=0.8,
            score_components={"entity_coverage": 1.0},
        )

        outcome = asyncio.run(engine._fetch_extract_outcome(candidate))

        self.assertIsNone(outcome.document)
        self.assertEqual(outcome.to_debug_dict()["status"], "failed")

    def test_snippet_fallback_can_recover_extraction_failure_without_new_get(
        self,
    ) -> None:
        class EmptyDocumentEngine(RealtimeSearchEngine):
            async def _fetch_extract(self, candidate):
                return None

        candidate = DiscoveredURL(
            url="https://docs.example/releases/current",
            title="Official current release notes",
            snippet=(
                "The current stable release includes documented compatibility, "
                "security, and migration changes from the official project."
            ),
            engine="dogpile",
            rank=1,
            rrf_score=0.2,
            candidate_score=0.7,
            score_components={"entity_coverage": 1.0},
        )
        control = EmptyDocumentEngine(
            RealtimeSearchConfig(
                snippet_fallback_enabled=True,
                snippet_fallback_on_extraction_error=False,
                snippet_fallback_min_chars=40,
            )
        )
        experiment = EmptyDocumentEngine(
            RealtimeSearchConfig(
                snippet_fallback_enabled=True,
                snippet_fallback_on_extraction_error=True,
                snippet_fallback_min_chars=40,
            )
        )

        control_outcome = asyncio.run(control._fetch_extract_outcome(candidate))
        experiment_outcome = asyncio.run(
            experiment._fetch_extract_outcome(candidate)
        )

        self.assertIsNone(control_outcome.document)
        self.assertEqual(control_outcome.error_type, "ExtractionError")
        self.assertIsNotNone(experiment_outcome.document)
        self.assertEqual(experiment_outcome.error_type, "ExtractionError")
        self.assertEqual(
            experiment_outcome.retrieval_mode, "search_snippet_fallback"
        )
        self.assertEqual(experiment_outcome.to_debug_dict()["status"], "fallback")

    def test_snippet_fallback_composite_confidence_can_replace_lexical_overlap(
        self,
    ) -> None:
        class BlockedFetcher:
            async def fetch(self, url):
                raise FetchError("HTTP 403")

        candidate = DiscoveredURL(
            url="https://docs.example/releases/current",
            title="Official current release notes",
            snippet=(
                "The current stable release includes documented compatibility, "
                "security, and migration changes from the official project."
            ),
            engine="dogpile",
            rank=1,
            rrf_score=0.2,
            candidate_score=0.45,
            score_components={"entity_coverage": 0.333333},
        )
        control = RealtimeSearchEngine(
            RealtimeSearchConfig(
                snippet_fallback_enabled=True,
                snippet_fallback_min_chars=40,
                snippet_fallback_entity_bypass_score=1.1,
            )
        )
        experiment = RealtimeSearchEngine(
            RealtimeSearchConfig(
                snippet_fallback_enabled=True,
                snippet_fallback_min_chars=40,
                snippet_fallback_entity_bypass_score=0.42,
            )
        )
        control._fetcher = BlockedFetcher()
        experiment._fetcher = BlockedFetcher()

        control_outcome = asyncio.run(control._fetch_extract_outcome(candidate))
        experiment_outcome = asyncio.run(
            experiment._fetch_extract_outcome(candidate)
        )

        self.assertIsNone(control_outcome.document)
        self.assertIsNotNone(experiment_outcome.document)
        self.assertEqual(
            experiment_outcome.retrieval_mode, "search_snippet_fallback"
        )

    def test_snippet_fallback_uses_cjk_density_threshold(self) -> None:
        class BlockedFetcher:
            async def fetch(self, url):
                raise FetchError("HTTP 403")

        engine = RealtimeSearchEngine(
            RealtimeSearchConfig(
                snippet_fallback_enabled=True,
                snippet_fallback_min_chars=96,
                snippet_fallback_min_cjk_chars=32,
            )
        )
        engine._fetcher = BlockedFetcher()
        candidate = DiscoveredURL(
            url="https://docs.example/zh/release",
            title="官方版本说明",
            snippet="这是官方发布页面的搜索结果摘要，包含当前稳定版本、发布日期、兼容性变化以及升级说明。",
            candidate_score=0.8,
            score_components={"entity_coverage": 1.0},
        )

        outcome = asyncio.run(engine._fetch_extract_outcome(candidate))

        self.assertIsNotNone(outcome.document)
        self.assertEqual(outcome.retrieval_mode, "search_snippet_fallback")

    def test_document_ranking_keeps_precise_candidate_signal(self) -> None:
        now = time.time()
        generic = RealtimeDocument(
            url="https://example.com/investor/",
            title="Investor relations",
            text="Quarterly investor information and general company materials. " * 4,
            published_at=None,
            fetched_at=now,
            source_type="web",
            authority=0.7,
            extraction_quality=0.8,
            rrf_score=0.03,
            candidate_score=0.2,
            simhash="0000000000000001",
        )
        precise = RealtimeDocument(
            url="https://example.com/investor/earnings/q3/",
            title="Q3 quarterly earnings press release",
            text="The latest Q3 quarterly earnings press release and webcast materials. "
            * 3,
            published_at=None,
            fetched_at=now,
            source_type="web",
            authority=0.7,
            extraction_quality=0.8,
            rrf_score=0.03,
            candidate_score=0.8,
            simhash="1000000000000000",
        )

        ranked = rank_documents(
            "latest quarterly earnings materials",
            [generic, precise],
            freshness_mode="latest",
            limit=2,
            per_domain_limit=1,
        )

        self.assertEqual([item.url for item in ranked], [precise.url])

    def test_default_final_domain_limit_matches_accepted_benchmark(self) -> None:
        self.assertEqual(SearchConfig().per_domain_limit, 4)

    def test_default_searxng_pool_matches_accepted_benchmark(self) -> None:
        self.assertEqual(
            RealtimeSearchConfig().searxng_engines,
            ["dogpile", "naver"],
        )

    def test_fetcher_accepts_markdown_response_types(self) -> None:
        self.assertIn("text/markdown", AsyncPageFetcher.ALLOWED_TYPES)
        self.assertIn("text/x-markdown", AsyncPageFetcher.ALLOWED_TYPES)


if __name__ == "__main__":
    unittest.main()
