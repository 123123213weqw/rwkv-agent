from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rwkv_search.commoncrawl import filter_records, iter_cdxj, parse_warc_http


class CommonCrawlTests(unittest.TestCase):
    def test_parse_and_filter_local_cdxj(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part.cdxj"
            data = {
                "url": "https://docs.example/page",
                "filename": "crawl-data/test.warc.gz",
                "offset": "100",
                "length": "900",
                "mime": "text/html",
                "status": "200",
                "languages": "zho",
            }
            path.write_text("com,example,docs)/page 20260715000000 " + json.dumps(data) + "\n", encoding="utf-8")
            records = list(filter_records(iter_cdxj(path), domains=["example"], languages=["zho"]))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].offset, 100)

    def test_parse_warc_http_payload(self) -> None:
        payload = (
            b"WARC/1.0\r\nContent-Length: 100\r\n\r\n"
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
            b"<html><body>hello</body></html>"
        )
        parsed = parse_warc_http(payload)
        self.assertEqual(parsed.status, 200)
        self.assertIn(b"hello", parsed.body)
        self.assertEqual(parsed.headers["content-type"], "text/html; charset=utf-8")
