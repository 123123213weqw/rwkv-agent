from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from bench.run_realtime_retrieval_bench import (
    apply_model_queries,
    load_cases,
    run_case,
    serialize_result,
)
from rwkv_search.search_request import SearchRequestBuilder


class FakeBenchEngine:
    def search_events(
        self,
        query,
        queries,
        *,
        freshness,
        depth,
        include_candidates=False,
    ):
        self.call = {
            "query": query,
            "queries": list(queries),
            "freshness": freshness,
            "depth": depth,
            "include_candidates": include_candidates,
        }
        yield {
            "type": "discovery_progress",
            "progress": {
                "candidate_count": 1,
                "elapsed_ms": 5.0,
                "candidates": [
                    {
                        "url": "https://www.python.org/downloads/",
                        "title": "Download Python",
                        "snippet": "Current stable release",
                        "engine": "bing",
                        "rank": 1,
                    }
                ],
            },
        }
        yield {
            "type": "fetch_progress",
            "progress": {
                "attempted": 1,
                "succeeded": 1,
                "failed": 0,
                "fetch": {
                    "requested_url": "https://www.python.org/downloads/",
                    "final_url": "https://www.python.org/downloads/",
                    "status": "succeeded",
                    "elapsed_ms": 8.0,
                },
            },
        }
        yield {
            "type": "realtime_result",
            "results": [
                {
                    "url": "https://www.python.org/downloads/",
                    "title": "Download Python",
                    "snippet": "Current stable release",
                    "content": "Official Python release information.",
                }
            ],
            "stats": {
                "attempted": 1,
                "fetched": 1,
                "failed": 0,
                "cancelled": 0,
                "elapsed_ms": 15.0,
            },
        }


class FailingBenchEngine:
    def search_events(self, *args, **kwargs):
        raise RuntimeError("discovery unavailable")


class RealtimeRetrievalRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = {
            "id": "retrieval-en-001",
            "query": "What is the current stable Python release according to python.org?",
            "language": "en",
            "category": "software_release",
            "freshness": "latest",
            "source_policy": "official_required",
            "expected_domains_any": ["python.org"],
            "target_url_patterns_any": ["/downloads/"],
            "forbidden_result_types": [
                "search_homepage",
                "dictionary",
                "error_page",
                "login_or_captcha",
                "empty_content",
            ],
        }

    def test_serialize_result_bounds_content_but_keeps_length_and_position(
        self,
    ) -> None:
        value = serialize_result(
            {"url": "https://example.com", "content": "abcdef", "title": "Example"}, 3
        )
        self.assertNotIn("content", value)
        self.assertEqual(value["content_length"], 6)
        self.assertEqual(value["position"], 3)

    def test_run_case_records_request_candidates_fetches_results_and_events(
        self,
    ) -> None:
        engine = FakeBenchEngine()
        record = run_case(self.case, engine, SearchRequestBuilder())
        self.assertTrue(engine.call["include_candidates"])
        self.assertEqual(record["execution"]["freshness"], "latest")
        self.assertEqual(record["candidates"][0]["position"], 1)
        self.assertEqual(record["fetches"][0]["status"], "succeeded")
        self.assertEqual(record["results"][0]["content_length"], 36)
        self.assertTrue(record["metrics"]["candidate_domain_hit_at_5"])
        self.assertTrue(record["metrics"]["result_target_page_hit_at_10"])
        self.assertEqual(record["metrics"]["fetch_success_rate"], 1.0)
        self.assertNotIn("candidates", record["events"][0]["progress"])
        self.assertNotIn("fetch", record["events"][1]["progress"])
        self.assertEqual(record["initial_candidates"], record["post_pivot_candidates"])
        self.assertEqual(
            record["candidate_stage_metrics"]["initial"]["candidate_count"], 1
        )

    def test_runner_failure_is_preserved_as_a_record(self) -> None:
        record = run_case(self.case, FailingBenchEngine(), SearchRequestBuilder())
        self.assertEqual(record["events"][-1]["type"], "runner_error")
        self.assertEqual(record["metrics"]["candidate_count"], 0)
        self.assertEqual(record["metrics"]["result_count"], 0)

    def test_case_filter_rejects_unknown_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown case ids"):
            load_cases(
                Path("bench/realtime_web_retrieval.jsonl"),
                ["retrieval-en-999"],
                0,
            )

    def test_frozen_model_queries_are_attached_by_case_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plans.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "retrieval-en-001",
                        "model_query": "python.org stable release",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            cases = apply_model_queries([self.case], path)
        self.assertEqual(cases[0]["model_query"], "python.org stable release")

    def test_frozen_model_queries_must_cover_every_selected_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plans.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing model queries"):
                apply_model_queries([self.case], path)


if __name__ == "__main__":
    unittest.main()
