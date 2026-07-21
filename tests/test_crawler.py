from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rwkv_search.config import CrawlConfig
from rwkv_search.crawler import FocusedCrawler
from rwkv_search.db import SearchDatabase
from rwkv_search.search import HybridSearcher


class FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/robots.txt":
            body = b"User-agent: *\nDisallow: /private\n"
            self._send(200, "text/plain", body)
        elif self.path == "/":
            body = """<html><head><title>RWKV 搜索首页</title></head><body><main>
            <h1>本地搜索</h1><p>这是 RWKV 本地搜索爬虫的测试网页，正文能够被抽取并建立索引。</p>
            <a href='/page2'>索引优化</a><a href='/private'>禁止页面</a></main></body></html>""".encode()
            self._send(200, "text/html; charset=utf-8", body)
        elif self.path == "/page2":
            body = """<html><head><title>混合检索优化</title></head><body><article>
            <p>搜索质量使用 BM25、向量检索、RRF 融合和 cross encoder 重排提升。</p>
            </article></body></html>""".encode()
            self._send(200, "text/html; charset=utf-8", body)
        elif self.path == "/private":
            self._send(200, "text/html", b"<html><body>secret content that must never be indexed</body></html>")
        else:
            self._send(404, "text/plain", b"missing")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


class CrawlerTests(unittest.TestCase):
    def test_focused_crawl_respects_robots_and_indexes_links(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                db = SearchDatabase(Path(tmp) / "search.db")
                config = CrawlConfig(
                    global_concurrency=2,
                    per_host_concurrency=1,
                    per_host_delay_seconds=0,
                    max_depth=1,
                    allow_private_networks=True,
                )
                crawler = FocusedCrawler(db, config)
                crawler.seed([f"http://127.0.0.1:{server.server_port}/"])
                counters = asyncio.run(crawler.run(10))
                self.assertEqual(counters["indexed"], 2)
                self.assertGreaterEqual(counters["skipped"], 1)
                results = HybridSearcher(db).search("向量检索 RRF")
                self.assertTrue(results)
                self.assertIn("page2", results[0].url)
                secret = HybridSearcher(db).search("secret content")
                self.assertFalse(secret)
        finally:
            server.shutdown()
            server.server_close()
