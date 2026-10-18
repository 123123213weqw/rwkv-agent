from __future__ import annotations

import unittest

from rwkv_search.text import canonicalize_url, extract_document, search_tokens


class TextTests(unittest.TestCase):
    def test_canonicalize_removes_tracking_and_fragment(self) -> None:
        self.assertEqual(
            canonicalize_url("HTTPS://Example.COM:443/docs/?utm_source=x&b=2&a=1#part"),
            "https://example.com/docs?a=1&b=2",
        )

    def test_chinese_bigram_and_latin_tokens(self) -> None:
        tokens = search_tokens("RWKV 搜索服务 SLA")
        self.assertIn("rwkv", tokens)
        self.assertIn("搜索", tokens)
        self.assertIn("索服", tokens)
        self.assertIn("sla", tokens)

    def test_fallback_extraction_ignores_navigation_and_finds_links(self) -> None:
        doc = extract_document(
            """<html lang='zh-CN'><head><title>测试文档</title></head><body>
            <nav>不要索引导航</nav><main><h1>RWKV 搜索</h1><p>这是用于本地检索的有效正文，包含足够多的说明文字。</p>
            <a href='/next?utm_source=x'>下一页</a></main><script>bad()</script></body></html>""",
            "https://example.com/start",
        )
        self.assertEqual(doc.title, "测试文档")
        self.assertIn("有效正文", doc.text)
        self.assertNotIn("不要索引导航", doc.text)
        self.assertIn("https://example.com/next", doc.links)
