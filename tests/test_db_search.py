from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from rwkv_search.db import SearchDatabase
from rwkv_search.search import HybridSearcher


class DatabaseSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db = SearchDatabase(Path(self.temp.name) / "search.db")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_upsert_dedup_and_chinese_search(self) -> None:
        kwargs = dict(
            url="https://docs.example/rwkv-search",
            canonical_url="https://docs.example/rwkv-search",
            title="RWKV 本地搜索设计",
            content="RWKV 搜索系统使用本地爬虫、BM25 关键词检索和可核查引用。",
            published_at="2026-07-15T00:00:00Z",
            fetched_at=time.time(),
            etag='"v1"',
            last_modified=None,
            content_type="text/html",
            language="zh-CN",
            source_type="official_docs",
            authority=0.9,
        )
        first_id, changed = self.db.upsert_document(**kwargs)
        second_id, changed_again = self.db.upsert_document(**kwargs)
        self.assertEqual(first_id, second_id)
        self.assertTrue(changed)
        self.assertFalse(changed_again)
        results = HybridSearcher(self.db).search("本地关键词检索")
        self.assertEqual(results[0].document_id, first_id)
        self.assertIn("BM25", results[0].snippet)
