from __future__ import annotations

import asyncio
import importlib.util
import json
import socket
import ssl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.web_extraction import (
    ExtractionOutput,
    aggregate_results,
    classify_fetch_exception,
    classify_http_status,
    content_kind,
    evaluate_output,
    looks_like_js_shell,
    parse_metadata_and_text,
    run_extractor,
    run_snapshot_benchmark,
    sha256_bytes,
)
from rwkv_search.realtime.extractor import extract_page
from rwkv_search.realtime.hybrid_extractor import (
    extract_hybrid_html,
    hybrid_fallback_reason,
    prefer_fallback,
)
from rwkv_search.realtime.types import FetchedPage


HTML = b"""<!doctype html>
<html><head><title>Example Release Notes</title>
<meta name="author" content="Example Team">
<meta property="article:published_time" content="2026-07-22">
</head><body><nav>Privacy Policy Navigation</nav><main><article>
<h1>Example Release Notes</h1>
<p>This release contains a stable implementation and enough explanatory prose
to be treated as a real document by the production extraction threshold.</p>
<table><tr><th>Version</th><th>Status</th></tr><tr><td>3.14</td><td>stable</td></tr></table>
<pre><code>for i in range(5): print(i)</code></pre>
</article></main><footer>Cookie Settings</footer></body></html>"""


def _case(**updates: object) -> dict:
    value = {
        "id": "extract-en-001",
        "language": "en",
        "page_type": "release",
        "content_kind": "html",
        "expected_static_outcome": "usable",
        "require_title": True,
        "title_contains_any": ["Example Release Notes"],
        "content_contains_any": ["stable implementation"],
        "forbidden_content_any": ["Cookie Settings"],
        "table_text_any": ["Version"],
        "code_text_any": ["range(5)"],
        "require_author": True,
        "require_published_at": True,
        "require_table": True,
        "require_code": True,
        "min_text_chars": 80,
    }
    value.update(updates)
    return value


def _snapshot(**updates: object) -> dict:
    value = {
        "fetch_outcome": "ok",
        "status": 200,
        "content_type": "text/html; charset=utf-8",
        "body_path": "extract-en-001.html",
        "body_sha256": sha256_bytes(HTML),
        "body_bytes": len(HTML),
        "final_url": "https://example.com/release",
    }
    value.update(updates)
    return value


class WebExtractionTest(unittest.TestCase):
    def test_fetch_exception_taxonomy(self) -> None:
        values = [
            (socket.gaierror("lookup failed"), "dns_error"),
            (ssl.SSLError("certificate verify failed"), "tls_error"),
            (asyncio.TimeoutError(), "request_timeout"),
            (ConnectionError("connection refused"), "connect_error"),
            (asyncio.CancelledError(), "deadline_cancelled"),
        ]
        for exception, expected in values:
            with self.subTest(expected=expected):
                self.assertEqual(classify_fetch_exception(exception), expected)

    def test_http_taxonomy(self) -> None:
        values = [(200, "ok"), (403, "http_403"), (429, "http_429"), (404, "http_4xx"), (503, "http_5xx")]
        for status, expected in values:
            with self.subTest(status=status):
                self.assertEqual(classify_http_status(status), expected)

    def test_content_kind_classification(self) -> None:
        self.assertEqual(content_kind("application/pdf", "https://x/a"), "pdf")
        self.assertEqual(content_kind("application/json", "https://x/a"), "json")
        self.assertEqual(content_kind("text/markdown", "https://x/a"), "markdown")
        self.assertEqual(content_kind("text/plain", "https://x/a"), "plain_text")
        self.assertEqual(content_kind("text/html; charset=utf-8", "https://x/a"), "html")
        self.assertEqual(content_kind("image/png", "https://x/a"), "unsupported")

    def test_metadata_and_js_shell_detection(self) -> None:
        title, text, author, published = parse_metadata_and_text(HTML.decode())
        self.assertEqual(title, "Example Release Notes")
        self.assertIn("stable implementation", text)
        self.assertEqual(author, "Example Team")
        self.assertEqual(published, "2026-07-22")
        shell = "<html><body><div id='root'></div><script>a()</script><script>b()</script></body></html>"
        self.assertTrue(looks_like_js_shell(shell, ""))
        generic_shell = (
            "<html><body>Loading application"
            + "<script></script>" * 12
            + " " * 15_000
            + "</body></html>"
        )
        self.assertTrue(looks_like_js_shell(generic_shell, "Loading application"))

    def test_current_extractor_and_quality_metrics(self) -> None:
        output = run_extractor(
            "current", HTML, "text/html; charset=utf-8", "https://x/release"
        )
        self.assertEqual(output.failure, "ok")
        metrics = evaluate_output(_case(), _snapshot(), output)
        self.assertTrue(metrics["title_hit"])
        self.assertTrue(metrics["content_hit"])
        self.assertTrue(metrics["author_hit"])
        self.assertTrue(metrics["published_at_hit"])
        self.assertTrue(metrics["table_hit"])
        self.assertTrue(metrics["code_hit"])
        self.assertEqual(metrics["forbidden_hits"], [])
        self.assertTrue(metrics["passed"])

    def test_json_output_does_not_require_title(self) -> None:
        body = json.dumps(
            {
                "tag_name": "v1.0",
                "html_url": "https://example.com",
                "description": "official release metadata " * 8,
            }
        ).encode()
        output = run_extractor(
            "current", body, "application/json", "https://example.com/a"
        )
        metrics = evaluate_output(
            _case(
                content_kind="json",
                require_title=False,
                title_contains_any=[],
                content_contains_any=["tag_name"],
                forbidden_content_any=[],
                table_text_any=[],
                code_text_any=[],
                require_author=False,
                require_published_at=False,
                require_table=False,
                require_code=False,
            ),
            _snapshot(content_type="application/json"),
            output,
        )
        self.assertIsNone(metrics["title_hit"])
        self.assertTrue(metrics["passed"])

    def test_snapshot_hash_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "extract-en-001.html").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "snapshot hash mismatch"):
                run_snapshot_benchmark(
                    [_case()], {"extract-en-001": _snapshot()}, root, ["current"]
                )

    def test_aggregate_reports_metadata_structure_and_failures(self) -> None:
        output = ExtractionOutput(
            "current",
            True,
            text="stable implementation " * 10,
            title="Example Release Notes",
            author="Example Team",
            published_at="2026-07-22",
            failure="ok",
            elapsed_ms=2.0,
        )
        metrics = evaluate_output(
            _case(require_table=False, require_code=False), _snapshot(), output
        )
        records = [
            {
                "extractor": "current",
                "available": True,
                "failure": "ok",
                "expected_static_outcome": "usable",
                "language": "en",
                "page_type": "release",
                "elapsed_ms": 2.0,
                "metrics": metrics,
            }
        ]
        summary = aggregate_results(
            [_case()], {"extract-en-001": _snapshot()}, records
        )
        current = summary["extractors"]["current"]
        self.assertEqual(current["pass_rate"], 1.0)
        self.assertEqual(current["author_hit_rate"], 1.0)
        self.assertEqual(current["published_at_hit_rate"], 1.0)
        self.assertEqual(current["by_language"]["en"]["pass_rate"], 1.0)

    def test_hybrid_generic_fallback_signals_do_not_use_domain_rules(self) -> None:
        primary = "short but otherwise valid article text " * 4
        self.assertEqual(
            hybrid_fallback_reason(primary, "visible page text " * 100),
            "low_main_content_ratio",
        )
        primary = "article " * 100 + " Privacy Policy Contact Us"
        self.assertEqual(
            hybrid_fallback_reason(primary, primary),
            "generic_boilerplate",
        )
        primary = "article " * 100
        self.assertIsNone(hybrid_fallback_reason(primary, primary))
        primary = "article " * 100 + " Recommended Service Privacy Policy"
        self.assertEqual(
            hybrid_fallback_reason(primary, primary),
            "generic_boilerplate_tail",
        )

    def test_hybrid_fallback_only_replaces_with_better_usable_output(self) -> None:
        primary = "article " * 100 + " Privacy Policy Contact Us"
        cleaner = "clean article " * 30
        self.assertTrue(prefer_fallback(primary, cleaner))
        self.assertFalse(prefer_fallback(primary, ""))

    @unittest.skipUnless(
        importlib.util.find_spec("resiliparse")
        and importlib.util.find_spec("trafilatura"),
        "optional extraction benchmark dependencies are not installed",
    )
    def test_hybrid_fast_combines_fast_body_and_metadata(self) -> None:
        output = run_extractor(
            "hybrid_fast",
            HTML,
            "text/html; charset=utf-8",
            "https://example.com/release",
        )
        self.assertEqual(output.failure, "ok")
        self.assertIn("stable implementation", output.text)
        self.assertEqual(output.author, "Example Team")
        self.assertEqual(output.published_at, "2026-07-22")
        self.assertEqual(output.details["primary"], "resiliparse")
        self.assertEqual(output.details["metadata"], "trafilatura_metadata")
        self.assertFalse(output.details["fallback_triggered"])
        self.assertFalse(output.details["fallback_used"])

    @unittest.skipUnless(
        importlib.util.find_spec("resiliparse")
        and importlib.util.find_spec("trafilatura"),
        "optional extraction dependencies are not installed",
    )
    def test_benchmark_hybrid_uses_production_implementation(self) -> None:
        raw_html = HTML.decode()
        production = extract_hybrid_html(
            HTML, raw_html, "https://example.com/release"
        )
        benchmark = run_extractor(
            "hybrid_fast",
            HTML,
            "text/html; charset=utf-8",
            "https://example.com/release",
        )
        self.assertEqual(benchmark.text, production.document.text)
        self.assertEqual(benchmark.title, production.document.title)
        self.assertEqual(benchmark.author, production.author)
        self.assertEqual(
            benchmark.published_at, production.document.published_at
        )
        for key in (
            "primary",
            "metadata",
            "fallback_triggered",
            "fallback_used",
            "fallback_reason",
            "primary_failure",
            "primary_text_length",
            "fallback_failure",
        ):
            self.assertEqual(benchmark.details[key], production.details[key])

    @unittest.skipUnless(
        importlib.util.find_spec("resiliparse")
        and importlib.util.find_spec("trafilatura"),
        "optional extraction dependencies are not installed",
    )
    def test_realtime_extract_page_uses_hybrid_text_and_metadata(self) -> None:
        page = FetchedPage(
            requested_url="https://example.com/release",
            final_url="https://example.com/release",
            status=200,
            content_type="text/html; charset=utf-8",
            body=HTML,
            fetched_at=1.0,
            elapsed_ms=2.0,
        )
        document = extract_page(page)
        self.assertIsNotNone(document)
        assert document is not None
        self.assertIn("stable implementation", document.text)
        self.assertNotIn("Cookie Settings", document.text)
        self.assertEqual(document.title, "Example Release Notes")
        self.assertEqual(document.published_at, "2026-07-22")

    def test_hybrid_safely_degrades_when_optional_extractor_is_missing(self) -> None:
        original_import = __import__

        def fail_resiliparse(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("resiliparse"):
                raise ImportError("resiliparse intentionally unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_resiliparse):
            extracted = extract_hybrid_html(
                HTML, HTML.decode(), "https://example.com/release"
            )
        self.assertIn("stable implementation", extracted.document.text)
        self.assertEqual(extracted.details["primary"], "legacy_fallback")
        self.assertEqual(
            extracted.details["fallback_reason"], "extractor_unavailable"
        )

    def test_aggregate_reports_hybrid_fallback_rates(self) -> None:
        output = ExtractionOutput(
            "hybrid_fast",
            True,
            text="stable implementation " * 10,
            title="Example Release Notes",
            author="Example Team",
            published_at="2026-07-22",
            failure="ok",
            elapsed_ms=2.0,
            details={
                "fallback_triggered": True,
                "fallback_used": True,
                "fallback_reason": "generic_boilerplate",
            },
        )
        metrics = evaluate_output(
            _case(require_table=False, require_code=False), _snapshot(), output
        )
        records = [
            {
                "extractor": "hybrid_fast",
                "available": True,
                "failure": "ok",
                "expected_static_outcome": "usable",
                "language": "en",
                "page_type": "release",
                "elapsed_ms": 2.0,
                "details": output.details,
                "metrics": metrics,
            }
        ]
        summary = aggregate_results(
            [_case()], {"extract-en-001": _snapshot()}, records
        )
        hybrid = summary["extractors"]["hybrid_fast"]
        self.assertEqual(hybrid["fallback_trigger_rate"], 1.0)
        self.assertEqual(hybrid["fallback_use_rate"], 1.0)
        self.assertEqual(
            hybrid["fallback_reasons"], {"generic_boilerplate": 1}
        )


if __name__ == "__main__":
    unittest.main()
