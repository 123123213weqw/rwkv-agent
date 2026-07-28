from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rwkv_agent.tools.web import (
    EnhancedWebShadow,
    WebSearchAdapter,
    _SHARED_DISCOVERY_CACHE,
    _configure,
)


class FakeEngine:
    def __init__(self) -> None:
        self.closed = False

    def search_events(self, *args, **kwargs):
        del args, kwargs
        yield {
            "type": "discovery_progress",
            "progress": {
                "initial_candidates": [
                    {
                        "url": "https://old.invalid/page",
                        "title": "Initial",
                    }
                ],
                "candidates": [
                    {
                        "url": "https://example.invalid/page",
                        "title": "Final",
                        "snippet": "Evidence",
                        "discovery_stage": "domain_pivot",
                    }
                ],
            },
        }
        yield {
            "type": "fetch_progress",
            "progress": {
                "fetch": {
                    "requested_url": "https://example.invalid/page",
                    "status": "succeeded",
                }
            },
        }
        yield {
            "type": "realtime_result",
            "results": [
                {
                    "url": "https://example.invalid/page",
                    "title": "Fetched",
                    "content": "Fetched evidence",
                    "source_type": "web",
                    "score": 1.0,
                }
            ],
            "stats": {"attempted": 1, "fetched": 1},
        }

    def close(self) -> None:
        self.closed = True


class FakeEnhancedAdapter:
    def __init__(
        self,
        *,
        fail: bool = False,
        empty: bool = False,
    ) -> None:
        self.fail = fail
        self.empty = empty
        self.closed = False

    def execute_with_trace(self, query: str):
        if self.fail:
            raise RuntimeError("boom")
        return (
            {
                "status": "ok",
                "evidence": (
                    []
                    if self.empty
                    else [
                        {
                            "id": "W1",
                            "uri": "https://enhanced.invalid/page",
                        }
                    ]
                ),
            },
            {
                "schema_version": "agent-web-trace.v1",
                "query": query,
                "candidates": [
                    {"url": "https://enhanced.invalid/page"}
                ],
            },
        )

    def close(self) -> None:
        self.closed = True


class WebShadowTests(unittest.TestCase):
    def test_profiles_freeze_expected_feature_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "realtime_search": {
                            "enabled": True,
                            "searxng_url": "",
                        }
                    }
                ),
                encoding="utf-8",
            )
            legacy = _configure(str(path), profile="legacy")
            enhanced = _configure(str(path), profile="enhanced")
            self.assertFalse(legacy.realtime_search.candidate_admission_enabled)
            self.assertFalse(legacy.realtime_search.query_compaction_enabled)
            self.assertFalse(legacy.realtime_search.domain_pivot_enabled)
            self.assertTrue(enhanced.realtime_search.candidate_admission_enabled)
            self.assertTrue(enhanced.realtime_search.query_compaction_enabled)
            self.assertTrue(enhanced.realtime_search.source_channels_enabled)
            self.assertTrue(enhanced.realtime_search.domain_pivot_enabled)
            self.assertTrue(
                enhanced.realtime_search.one_hop_link_expansion_enabled
            )

    def test_adapter_trace_records_candidates_fetches_and_results(self) -> None:
        engine = FakeEngine()
        adapter = WebSearchAdapter(engine=engine, shadow=False)
        public, trace = adapter.execute_with_trace("query")
        self.assertEqual(public["evidence"][0]["id"], "W1")
        self.assertEqual(trace["initial_candidates"][0]["title"], "Initial")
        self.assertEqual(trace["candidates"][0]["title"], "Final")
        self.assertEqual(trace["fetches"][0]["status"], "succeeded")
        self.assertEqual(trace["results"][0]["content_length"], 16)
        adapter.close()
        self.assertTrue(engine.closed)

    def test_legacy_and_enhanced_adapters_share_discovery_cache(self) -> None:
        with patch("rwkv_agent.tools.web.RealtimeSearchEngine") as engine_type:
            legacy = WebSearchAdapter(profile="legacy", shadow=False)
            enhanced = WebSearchAdapter(profile="enhanced", shadow=False)
        self.assertEqual(engine_type.call_count, 2)
        for call in engine_type.call_args_list:
            self.assertIs(
                call.kwargs["discovery_cache"],
                _SHARED_DISCOVERY_CACHE,
            )
        legacy.close()
        enhanced.close()

    def test_shadow_failure_is_logged_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.jsonl"
            shadow = EnhancedWebShadow(
                FakeEnhancedAdapter(fail=True),
                log_path=str(path),
            )
            value = shadow.compare(
                "query",
                legacy_trace={"candidates": []},
                legacy_evidence=[{"uri": "https://legacy.invalid"}],
            )
            shadow.close()
            self.assertEqual(value["status"], "fallback_legacy")
            self.assertEqual(
                json.loads(path.read_text())["status"],
                "fallback_legacy",
            )

    def test_empty_enhanced_evidence_uses_auditable_legacy_fallback(self) -> None:
        shadow = EnhancedWebShadow(FakeEnhancedAdapter(empty=True))
        row = shadow.compare(
            "query",
            legacy_trace={"candidates": []},
            legacy_evidence=[{"uri": "https://legacy.invalid/page"}],
        )
        shadow.close()
        self.assertEqual(row["status"], "fallback_legacy_evidence")
        self.assertTrue(row["fallback_used"])
        self.assertEqual(
            row["effective_urls"],
            ["https://legacy.invalid/page"],
        )
        self.assertEqual(row["enhanced_urls"], [])

    def test_shadow_queue_is_bounded(self) -> None:
        import threading

        class BlockingAdapter(FakeEnhancedAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.release = threading.Event()

            def execute_with_trace(self, query: str):
                self.release.wait(timeout=2)
                return super().execute_with_trace(query)

        adapter = BlockingAdapter()
        shadow = EnhancedWebShadow(adapter, max_pending=1)
        first = shadow.submit(
            "one",
            legacy_trace={},
            legacy_evidence=[],
        )
        second = shadow.submit(
            "two",
            legacy_trace={},
            legacy_evidence=[],
        )
        self.assertTrue(first["submitted"])
        self.assertFalse(second["submitted"])
        self.assertEqual(second["reason"], "queue_full")
        adapter.release.set()
        shadow.close()
        self.assertTrue(adapter.closed)


if __name__ == "__main__":
    unittest.main()
