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

from rwkv_search.config import RealtimeSearchConfig
from rwkv_search.db import SearchDatabase
from rwkv_search.realtime.discovery import (
    URLDiscovery,
    bing_search_headers,
    bing_search_params,
    parse_search_html,
    parse_searxng_results,
    searxng_search_params,
)
from rwkv_search.realtime.engine import RealtimeSearchEngine
from rwkv_search.realtime.extractor import classify_source, extract_page
from rwkv_search.realtime.fetcher import AsyncPageFetcher, FetchError
from rwkv_search.realtime.ranker import rank_documents, to_search_results
from rwkv_search.realtime.types import DiscoveredURL, FetchedPage, RealtimeDocument
from rwkv_search.search import SearchResult
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
    async def fetch(self, url):
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


class RealtimeSearchTests(unittest.TestCase):
    def test_searxng_params_use_query_language_without_disabling_engines(self) -> None:
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
        self.assertEqual(
            searxng_search_params(
                "RWKV GitHub repository", "latest", ("general", "repos")
            )["categories"],
            "general,repos",
        )

    def test_bing_params_and_headers_follow_query_locale(self) -> None:
        self.assertEqual(bing_search_params("Python latest release")["mkt"], "en-US")
        self.assertEqual(bing_search_params("Python 最新版本")["mkt"], "zh-CN")
        self.assertEqual(bing_search_params("Python 最新版本")["adlt"], "moderate")
        self.assertEqual(
            bing_search_headers("Python 最新版本")["Accept-Language"],
            "zh-CN,zh;q=0.9",
        )

    def test_discovery_diagnostics_preserve_searxng_connection_errors(self) -> None:
        async def run():
            config = RealtimeSearchConfig(
                enabled=True,
                searxng_url="http://127.0.0.1:8888",
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
                ("llama.cpp", ("repos",)),
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
            events = list(service.ask_events("搜索一下实时抓取方案"))
        kinds = [event["type"] for event in events]
        self.assertTrue(engine.called)
        self.assertIn("discovery_progress", kinds)
        self.assertIn("fetch_progress", kinds)
        evidence = next(event for event in events if event["type"] == "evidence")
        self.assertEqual(
            evidence["evidence"][0]["url"], "https://docs.example/realtime"
        )

    def test_named_entity_is_a_hard_relevance_constraint(self) -> None:
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
        )
        self.assertEqual([item.url for item in ranked], ["https://www.python.org/"])

    def test_finance_ranking_rejects_generic_portal_homepages(self) -> None:
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
        )
        self.assertEqual([item.url for item in ranked], [filing.url])


if __name__ == "__main__":
    unittest.main()
