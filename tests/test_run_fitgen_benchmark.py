from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from benchmarks import run_fitgen_benchmark as module


class CoreScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (module.CORE_DIR / "bfcl.jsonl").is_file():
            with (module.CORE_DIR / "bfcl.jsonl").open() as handle:
                cls.bfcl = [json.loads(line) for line in handle]
        else:
            cls.bfcl = []

    def test_parse_single_and_parallel_calls(self):
        calls, raw = module.parse_bfcl_calls('{"name":"a.b","arguments":{"x":1}}</tool_call>')
        self.assertEqual(calls, [{"name": "a.b", "arguments": {"x": 1}}])
        self.assertTrue(raw.startswith("<tool_call>"))
        calls, _ = module.parse_bfcl_calls('[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]</tool_call>')
        self.assertEqual([item["name"] for item in calls], ["a", "b"])
        calls, _ = module.parse_bfcl_calls(
            "[{'name':'a','arguments':{'x':1}},"
            "{'name':'b','arguments':{'y':2}}]</tool_call>"
        )
        self.assertEqual([item["name"] for item in calls], ["a", "b"])

    def test_parse_rejects_extra_keys(self):
        with self.assertRaises(ValueError):
            module.parse_bfcl_calls('{"name":"a","arguments":{},"extra":1}</tool_call>')

    def test_parse_repairs_arguments_first_missing_braces(self):
        raw = (
            '[{"arguments":{"location":"Los Angeles","population":'
            '{"adults":2,"children":2,"singles":0},"name":"waste.calculate"},'
            '{"arguments":{"location":"New York","population":'
            '{"adults":1,"children":0,"singles":1},"name":"waste.calculate"}]}'
            '</tool_call>'
        )
        calls, _ = module.parse_bfcl_calls(raw)
        self.assertEqual([call["name"] for call in calls], ["waste.calculate"] * 2)
        self.assertEqual(calls[0]["arguments"]["population"]["adults"], 2)

    def test_valid_argument_named_name_is_not_rewritten(self):
        calls, _ = module.parse_bfcl_calls(
            '[{"name":"people.lookup","arguments":{"name":"Ada"}}]</tool_call>'
        )
        self.assertEqual(calls[0]["arguments"], {"name": "Ada"})

    def test_bfcl_schema_normalizer_is_gold_blind_and_lossless(self):
        case = {
            "available_tools": [
                {
                    "name": "demo.run",
                    "parameters": {
                        "type": "dict",
                        "required": ["location", "floor", "score"],
                        "properties": {
                            "location": {"type": "string"},
                            "floor": {"type": "integer"},
                            "score": {"type": "float"},
                        },
                    },
                }
            ]
        }
        calls = module.normalize_bfcl_calls(
            case,
            [
                {
                    "name": "demo.run",
                    "arguments": {
                        "city": "Paris",
                        "floor": ["2", "3"],
                        "score": 4,
                        "invented": {"drop": True},
                    },
                }
            ],
        )
        self.assertEqual(
            calls,
            [
                {
                    "name": "demo.run",
                    "arguments": {"location": "Paris", "floor": 2, "score": 4.0},
                },
                {
                    "name": "demo.run",
                    "arguments": {"location": "Paris", "floor": 3, "score": 4.0},
                },
            ],
        )

    def test_bfcl_schema_normalizer_uses_explicit_pm_grounding(self):
        case = {
            "prompt": "Find a restaurant open until at least 11 PM.",
            "available_tools": [
                {
                    "name": "restaurant.find",
                    "parameters": {
                        "type": "object",
                        "properties": {"operating_hours": {"type": "integer"}},
                    },
                }
            ],
        }
        calls = module.normalize_bfcl_calls(
            case,
            [{"name": "restaurant.find", "arguments": {"operating_hours": 11}}],
        )
        self.assertEqual(calls[0]["arguments"]["operating_hours"], 23)

    def test_bfcl_schema_normalizer_expands_user_named_enum_array(self):
        case = {
            "prompt": "Get strength and weakness traits for ENFJ.",
            "available_tools": [
                {
                    "name": "traits.get",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "traits": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["strengths", "weaknesses"],
                                },
                            }
                        },
                    },
                }
            ],
        }
        calls = module.normalize_bfcl_calls(
            case,
            [{"name": "traits.get", "arguments": {"traits": ["strengths"]}}],
        )
        self.assertEqual(
            calls[0]["arguments"]["traits"], ["strengths", "weaknesses"]
        )

    def test_web_score_binds_public_root_only_at_tool_adapter(self):
        class FakeWeb:
            def __init__(self) -> None:
                self.root = ""

            @contextmanager
            def scoped(self, root: str, *, original_query: str = ""):
                self.root = root
                self.original_query = original_query
                yield

        class FakeController:
            def __init__(self) -> None:
                self.web = FakeWeb()
                self.message = ""

            def run_stateful_search(self, message: str, **_kwargs):
                self.message = message
                return {
                    "status": "ok",
                    "answer": "",
                    "tool_result": {"evidence": []},
                    "trace": {
                        "rounds": [],
                        "state_runtime": {"release": {"released": 1}},
                    },
                }

        case = {
            "schema_version": "agent-benchmark-case.v1",
            "id": "web-x",
            "dataset": "webwalkerqa",
            "split": "dev",
            "track": "web",
            "language": "en",
            "prompt": "Who won the award?",
            "metadata": {"root_url": "https://www.example.org/"},
            "gold": {
                "answers": ["Alice"],
                "source_uris": ["https://www.example.org/private/gold"],
            },
            "limits": {"max_requests": 8, "max_rounds": 2, "timeout_seconds": 60},
        }
        controller = FakeController()
        row = module.score_web(case, controller)
        self.assertEqual(controller.web.root, "https://www.example.org/")
        self.assertEqual(controller.web.original_query, "Who won the award?")
        self.assertEqual(controller.message, case["prompt"])
        self.assertNotIn("private/gold", controller.message)
        self.assertEqual(row["benchmark"]["search_scope"], "https://www.example.org/")
        self.assertTrue(row["abstained"])
        self.assertEqual(row["claims"], [])

    def test_web_score_treats_insufficient_evidence_as_safe_abstention(self):
        class FakeWeb:
            @contextmanager
            def scoped(self, _root: str, *, original_query: str = ""):
                self.original_query = original_query
                yield

        class FakeController:
            def __init__(self) -> None:
                self.web = FakeWeb()

            def run_stateful_search(self, _message: str, **_kwargs):
                return {
                    "status": "insufficient_evidence",
                    "answer": "No sufficient verifiable evidence was found.",
                    "tool_result": {"evidence": []},
                    "trace": {
                        "rounds": [
                            {
                                "round": 1,
                                "branches": [
                                    {
                                        "route": {
                                            "strict": True,
                                            "arguments": {"query": "award winner"},
                                        },
                                        "effective_query": (
                                            "award winner site:example.org"
                                        ),
                                    }
                                ],
                            }
                        ],
                        "state_runtime": {
                            "forked_states": 0,
                            "release": {"released": 1},
                        },
                    },
                }

        case = {
            "schema_version": "agent-benchmark-case.v1",
            "id": "web-insufficient",
            "dataset": "webwalkerqa",
            "split": "dev",
            "track": "web_research",
            "language": "en",
            "prompt": "Who won the award?",
            "metadata": {"root_url": "https://example.org/"},
            "gold": {
                "answerable": True,
                "answers": ["Alice"],
                "requires_citations": True,
                "source_uris": ["https://example.org/answer"],
            },
            "limits": {"max_requests": 8, "max_rounds": 2},
        }

        row = module.score_web(case, FakeController())

        self.assertEqual(row["status"], "ok")
        self.assertTrue(row["abstained"])
        self.assertEqual(row["claims"], [])
        self.assertEqual(
            row["benchmark"]["executed_search_queries"],
            ["award winner site:example.org"],
        )
        self.assertEqual(
            row["benchmark"]["primary_retrieval_metric"],
            "exact_page_recall",
        )

    def test_web_score_excludes_controller_partial_support_notice_from_claims(self):
        class FakeWeb:
            @contextmanager
            def scoped(self, _root: str, *, original_query: str = ""):
                yield

        class FakeController:
            web = FakeWeb()

            def run_stateful_search(self, _message: str, **_kwargs):
                return {
                    "status": "ok",
                    "answer": (
                        "The verified fact is supported [W1].\n"
                        "Other requested details were omitted because support "
                        "was insufficient."
                    ),
                    "tool_result": {
                        "evidence": [
                            {
                                "id": "W1",
                                "title": "Verified fact",
                                "content": "The verified fact is supported.",
                                "uri": "https://example.org/fact",
                            }
                        ]
                    },
                    "trace": {
                        "rounds": [],
                        "state_runtime": {"release": {"released": 1}},
                        "answer_protocol": {"policy_notice": "partial_support"},
                    },
                }

        case = {
            "id": "web-partial",
            "prompt": "What facts are verified?",
            "metadata": {"root_url": "https://example.org/"},
            "limits": {},
        }

        row = module.score_web(case, FakeController())

        self.assertEqual(len(row["claims"]), 1)
        self.assertTrue(row["claims"][0]["supported"])

    def test_alce_only_treats_unsupported_evidence_as_safe_abstention(self):
        self.assertTrue(
            module._is_insufficient_evidence_validation(
                ["missing_citation", "unsupported_claim"]
            )
        )
        self.assertTrue(
            module._is_insufficient_evidence_validation(["unsupported_claim"])
        )
        self.assertFalse(
            module._is_insufficient_evidence_validation(["missing_citation"])
        )
        self.assertFalse(
            module._is_insufficient_evidence_validation(
                ["unsupported_claim", "protocol_tag"]
            )
        )

    def test_official_ast_accepts_frozen_gold_for_every_category(self):
        if not self.bfcl:
            self.skipTest("frozen BFCL data is only available on the server")
        for category in ("simple", "multiple", "parallel", "parallel_multiple"):
            case = next(row for row in self.bfcl if row["metadata"]["category"] == category)
            calls = json.loads(json.dumps(case["gold"]["tool_calls"]))
            for call in calls:
                call["arguments"] = {key: value for key, value in call["arguments"].items() if value != ""}
            result = module.bfcl_official_ast(case, calls)
            self.assertTrue(result["valid"], (category, result))

    def test_official_ast_rejects_wrong_value(self):
        if not self.bfcl:
            self.skipTest("frozen BFCL data is only available on the server")
        case = self.bfcl[0]
        calls = json.loads(json.dumps(case["gold"]["tool_calls"]))
        first = next(iter(calls[0]["arguments"]))
        calls[0]["arguments"][first] = "definitely wrong"
        self.assertFalse(module.bfcl_official_ast(case, calls)["valid"])

    def test_smoke_selection_is_deterministic_and_stratified(self):
        if not self.bfcl:
            self.skipTest("frozen BFCL data is only available on the server")
        one = module.select_smoke(self.bfcl, "bfcl", 20)
        two = module.select_smoke(self.bfcl, "bfcl", 20)
        self.assertEqual([row["id"] for row in one], [row["id"] for row in two])
        counts = {}
        for row in one:
            key = row["metadata"]["category"]
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(counts, {"multiple": 5, "parallel": 5, "parallel_multiple": 5, "simple": 5})

    def test_selected_cases_can_read_an_explicit_locked_split(self):
        if not self.bfcl:
            self.skipTest("frozen BFCL data is only available on the server")
        with tempfile.TemporaryDirectory() as temporary:
            cases_dir = Path(temporary)
            expected = self.bfcl[:3]
            (cases_dir / "bfcl.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in expected),
                encoding="utf-8",
            )
            actual = module.selected_cases("bfcl", None, cases_dir=cases_dir)
        self.assertEqual([row["id"] for row in actual], [row["id"] for row in expected])

    def test_deferred_scoring_requires_gold_free_fresh_cases(self):
        public = {
            "split": "fresh_web_once",
            "gold": {
                "answerable": True,
                "requires_citations": True,
                "should_call_tools": True,
            },
        }
        module.validate_deferred_scoring_cases(
            "webwalkerqa", [public], smoke=None
        )
        exposed = {
            **public,
            "gold": {**public["gold"], "answers": ["private answer"]},
        }
        with self.assertRaisesRegex(ValueError, "must not expose private Gold"):
            module.validate_deferred_scoring_cases(
                "webwalkerqa", [exposed], smoke=None
            )
        with self.assertRaisesRegex(ValueError, "full webwalkerqa"):
            module.validate_deferred_scoring_cases(
                "frames", [public], smoke=None
            )

    def test_runtime_receives_explicit_api_provider_policy(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            module, "WebSearchAdapter"
        ) as web_adapter, patch.object(
            module, "AgentController"
        ) as controller, patch.object(module, "ModelClient"), patch.object(
            module, "MODEL_URLS", ("http://model",)
        ):
            runtime = module.make_runtime(
                "webwalkerqa",
                Path(temporary),
                web_profile="balanced",
                web_fallback_engines=("bing",),
                web_api_providers=("github", "crossref", "mediawiki"),
                longbench_mode="lexical",
                alce_max_tokens=32,
                alce_prompt_profile="full",
            )
            web_adapter.assert_called_once_with(
                str(module.DEFAULT_CONFIG),
                profile="balanced",
                shadow=False,
                fallback_engines=("bing",),
                api_providers=("github", "crossref", "mediawiki"),
            )
            self.assertEqual(runtime.controller_pool.controllers, [controller.return_value])

    def test_base_and_error_results_validate(self):
        if not self.bfcl:
            self.skipTest("frozen BFCL data is only available on the server")
        case = self.bfcl[0]
        module.validate_result(module.base_result(case, status="ok"))
        module.validate_result(module.error_result(case, RuntimeError("x"), 1.5))

    def test_controller_pool_never_leases_one_sidecar_twice(self):
        class FakeController:
            def __init__(self, name: str) -> None:
                self.name = name
                self.active = 0
                self.max_active = 0
                self.used = 0
                self.closed = False
                self.lock = threading.Lock()

            def work(self) -> str:
                with self.lock:
                    self.active += 1
                    self.used += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.01)
                with self.lock:
                    self.active -= 1
                return self.name

            def close(self) -> None:
                self.closed = True

        controllers = [FakeController("gpu0"), FakeController("gpu1")]
        pool = module.ControllerLeasePool(controllers)

        def run_one() -> str:
            with pool.lease() as controller:
                return controller.work()

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _value: run_one(), range(20)))
        pool.close()
        self.assertEqual(set(results), {"gpu0", "gpu1"})
        self.assertTrue(all(controller.max_active == 1 for controller in controllers))
        self.assertTrue(all(controller.used for controller in controllers))
        self.assertTrue(all(controller.closed for controller in controllers))

    def test_summary_freezes_reliability_counters(self):
        report = {"summary": {"overall": {"metrics": {}}}}
        results = [
            {
                "status": "error",
                "trace": {"states_leaked": 2},
                "benchmark": {
                    "native_status": "route_error",
                    "error": "HTTP 409 state lease conflict",
                },
            }
        ]
        evaluations = [
            {
                "metrics": {
                    "protocol_leak": True,
                    "within_latency_budget": True,
                    "within_request_budget": False,
                    "within_round_budget": True,
                }
            }
        ]
        summary = module.summarize(
            "webwalkerqa", results, report, evaluations
        )
        self.assertEqual(
            summary["reliability"],
            {
                "http_409_count": 1,
                "route_error_count": 1,
                "state_leak_count": 2,
                "protocol_leak_count": 1,
                "budget_overrun_count": 1,
            },
        )

    def test_summary_does_not_count_409_inside_completion_hash(self):
        report = {"summary": {"overall": {"metrics": {}}}}
        results = [
            {
                "status": "ok",
                "trace": {},
                "benchmark": {
                    "completion": {"output_sha256": "409a47b47371537f"}
                },
            }
        ]
        summary = module.summarize("webwalkerqa", results, report, [])
        self.assertEqual(summary["reliability"]["http_409_count"], 0)


if __name__ == "__main__":
    unittest.main()
