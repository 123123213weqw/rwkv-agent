from __future__ import annotations

import json
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from rwkv_agent.tools.web import (
    EnhancedWebShadow,
    WebSearchAdapter,
    _SHARED_DISCOVERY_CACHE,
    _configure,
    build_web_shadow_from_env,
)
from rwkv_search.config import RealtimeSearchConfig


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
    def test_scoped_original_query_lane_runs_only_once(self) -> None:
        class CapturingEngine:
            def __init__(self) -> None:
                self.config = RealtimeSearchConfig(
                    original_query_lane_enabled=True
                )
                self.queries = []

            def search_events(self, _query, queries, **_kwargs):
                self.queries.append(list(queries))
                yield {"type": "realtime_result", "results": [], "stats": {}}

            def close(self) -> None:
                pass

        engine = CapturingEngine()
        adapter = WebSearchAdapter(engine=engine, shadow=False)
        original = (
            "Which game by Duoyi Network won the Outstanding Mobile Game "
            "award at the 2024 Golden Finger Awards, and when did the test "
            "for Wan Xian Zhu Lu start?"
        )
        with adapter.scoped(
            "https://www.duoyi.com/",
            original_query=original,
        ):
            first_public, first_trace = adapter.execute_with_trace(
                "Duoyi Golden Finger award",
            )
            second_public, second_trace = adapter.execute_with_trace(
                "Wan Xian Zhu Lu test date",
            )

        self.assertEqual(len(engine.queries[0]), 2)
        self.assertEqual(len(engine.queries[1]), 1)
        self.assertTrue(first_trace["original_query_lane"])
        self.assertFalse(second_trace["original_query_lane"])
        self.assertIn(
            "golden finger",
            first_trace["original_query_lane"].casefold(),
        )
        self.assertEqual(first_public["execution_queries"], engine.queries[0])
        self.assertEqual(second_public["execution_queries"], engine.queries[1])

    def test_scoped_original_query_lane_is_atomic_across_parallel_branches(self) -> None:
        class CapturingEngine:
            def __init__(self) -> None:
                self.config = RealtimeSearchConfig(
                    original_query_lane_enabled=True
                )
                self.queries = []

            def search_events(self, _query, queries, **_kwargs):
                self.queries.append(list(queries))
                yield {"type": "realtime_result", "results": [], "stats": {}}

            def close(self) -> None:
                pass

        engine = CapturingEngine()
        adapter = WebSearchAdapter(engine=engine, shadow=False)
        original = "Which official pages identify the two award recipients?"
        with adapter.scoped(
            "https://example.org/",
            original_query=original,
        ):
            with ThreadPoolExecutor(max_workers=4) as executor:
                traces = list(
                    executor.map(
                        lambda index: adapter.execute_with_trace(
                            f"award recipient branch {index}"
                        )[1],
                        range(4),
                    )
                )

        self.assertEqual(
            sum(bool(trace["original_query_lane"]) for trace in traces),
            1,
        )
        self.assertEqual(sum(len(queries) == 2 for queries in engine.queries), 1)

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

    def test_shadow_sampling_skips_without_executing_candidate(self) -> None:
        class CountingAdapter(FakeEnhancedAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def execute_with_trace(self, query: str):
                self.calls += 1
                return super().execute_with_trace(query)

        adapter = CountingAdapter()
        shadow = EnhancedWebShadow(
            adapter,
            sample_rate=0.1,
            sampler=lambda: 0.9,
        )

        value = shadow.submit(
            "private query",
            legacy_trace={},
            legacy_evidence=[],
        )
        shadow.close()

        self.assertFalse(value["submitted"])
        self.assertEqual(value["reason"], "sample_not_selected")
        self.assertEqual(value["sample_rate"], 0.1)
        self.assertEqual(adapter.calls, 0)

    def test_metrics_log_omits_queries_urls_content_and_full_traces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow-metrics.jsonl"
            shadow = EnhancedWebShadow(
                FakeEnhancedAdapter(),
                log_path=str(path),
                log_mode="metrics",
            )
            shadow.compare(
                "private user query",
                legacy_trace={
                    "status": "ok",
                    "candidates": [
                        {
                            "url": "https://private.invalid/page",
                            "content": "private page content",
                        }
                    ],
                    "results": [{"url": "https://private.invalid/page"}],
                    "fetches": [{"status": "succeeded"}],
                    "warnings": [],
                    "latency_ms": 12.5,
                    "evidence_stage": "fetched",
                },
                legacy_evidence=[
                    {"uri": "https://private.invalid/page"}
                ],
            )
            shadow.close()

            raw = path.read_text(encoding="utf-8")
            row = json.loads(raw)
            self.assertEqual(
                row["schema_version"],
                "agent-web-shadow-metrics.v1",
            )
            self.assertEqual(row["query_chars"], 18)
            self.assertEqual(row["legacy"]["fetch_success_count"], 1)
            self.assertNotIn("private user query", raw)
            self.assertNotIn("private.invalid", raw)
            self.assertNotIn("private page content", raw)
            self.assertNotIn("legacy_trace", row)
            self.assertNotIn("enhanced_trace", row)

    def test_environment_shadow_defaults_to_ten_percent_metrics_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                "os.environ",
                {
                    "RWKV_AGENT_WEB_SHADOW": "1",
                    "RWKV_AGENT_WEB_SHADOW_LOG": str(
                        Path(directory) / "shadow.jsonl"
                    ),
                },
                clear=True,
            ), patch(
                "rwkv_agent.tools.web.WebSearchAdapter",
                return_value=FakeEnhancedAdapter(),
            ):
                shadow = build_web_shadow_from_env("configs/default.json")

        self.assertIsNotNone(shadow)
        assert shadow is not None
        self.assertEqual(shadow.sample_rate, 0.1)
        self.assertEqual(shadow.log_mode, "metrics")
        shadow.close()

    def test_invalid_environment_sampling_fails_closed(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "RWKV_AGENT_WEB_SHADOW": "1",
                "RWKV_AGENT_WEB_SHADOW_SAMPLE_RATE": "all",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "sampling configuration"):
                build_web_shadow_from_env("configs/default.json")

        with patch.dict(
            "os.environ",
            {
                "RWKV_AGENT_WEB_SHADOW": "1",
                "RWKV_AGENT_WEB_SHADOW_SAMPLE_RATE": "1.1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                build_web_shadow_from_env("configs/default.json")


if __name__ == "__main__":
    unittest.main()
