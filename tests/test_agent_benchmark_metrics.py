from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.agent_benchmark_metrics import (
    aggregate_evaluations,
    answer_tokens,
    compare_evaluations,
    evaluate_agent_case,
)
from benchmarks.agent_benchmark_schema import (
    CASE_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
)
from benchmarks.run_agent_benchmark_metrics import build_report, main


def base_case(*, case_id: str = "case-1", track: str = "web_research") -> dict:
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "id": case_id,
        "dataset": "fixture",
        "split": "test",
        "track": track,
        "language": "zh",
        "prompt": "谁维护青鸟项目？",
        "gold": {
            "answers": ["星河实验室"],
            "answerable": True,
            "requires_citations": True,
            "source_uris": ["https://example.com/project"],
            "evidence_ids": ["gold-project"],
        },
        "limits": {
            "max_rounds": 2,
            "max_requests": 8,
            "max_latency_ms": 20000,
        },
        "metadata": {"context_bucket": "8k-32k"},
    }


def base_result(*, case_id: str = "case-1") -> dict:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "case_id": case_id,
        "status": "ok",
        "answer": "星河实验室 [W1]",
        "abstained": False,
        "tool_calls": [],
        "evidence": [
            {
                "id": "W1",
                "gold_id": "gold-project",
                "uri": "https://example.com/project/",
            }
        ],
        "claims": [
            {
                "text": "星河实验室维护青鸟项目",
                "citations": ["W1"],
                "requires_citation": True,
                "supported": True,
            }
        ],
        "trace": {
            "requests": 4,
            "rounds": 2,
            "states_created": 5,
            "states_released": 5,
            "states_leaked": 0,
            "states_reused": 4,
        },
        "resources": {
            "latency_ms": 14000,
            "ttft_ms": 800,
            "gpu_peak_mib": 16000,
            "cpu_state_peak_mib": 120,
            "input_tokens": 4000,
            "output_tokens": 30,
        },
    }


class AgentBenchmarkMetricTests(unittest.TestCase):
    def test_chinese_answer_tokenization_and_grounding(self) -> None:
        self.assertEqual(answer_tokens("星河实验室"), list("星河实验室"))
        row = evaluate_agent_case(base_case(), base_result())
        metrics = row["metrics"]
        self.assertTrue(metrics["answer_exact_match"])
        self.assertEqual(metrics["source_recall"], 1.0)
        self.assertEqual(metrics["citation_source_recall"], 1.0)
        self.assertEqual(metrics["claim_citation_coverage"], 1.0)
        self.assertTrue(metrics["state_cleanup_success"])
        self.assertTrue(metrics["state_reuse_success"])
        self.assertTrue(metrics["within_latency_budget"])
        self.assertFalse(metrics["protocol_leak"])

    def test_tool_protocol_scores_calls_arguments_order_and_groups(self) -> None:
        case = base_case(track="tool_protocol")
        case["gold"] = {
            "should_call_tools": True,
            "tool_calls": [
                {
                    "name": "web_search",
                    "arguments": {"query": "RWKV author"},
                    "parallel_group": 1,
                },
                {
                    "name": "web_search",
                    "arguments": {"query": "RWKV organization"},
                    "parallel_group": 1,
                },
            ],
        }
        result = base_result()
        result["answer"] = ""
        result["evidence"] = []
        result["claims"] = []
        result["tool_calls"] = list(reversed(case["gold"]["tool_calls"]))
        result["protocol"] = {"tool_call_valid": True}
        metrics = evaluate_agent_case(case, result)["metrics"]
        self.assertTrue(metrics["tool_needed_accuracy"])
        self.assertTrue(metrics["tool_call_exact_match"])
        self.assertFalse(metrics["tool_sequence_exact_match"])
        self.assertTrue(metrics["tool_group_exact_match"])
        self.assertTrue(metrics["tool_protocol_valid"])

    def test_bfcl_strict_metric_uses_executable_gold_not_empty_sentinels(self) -> None:
        case = base_case(track="tool_protocol")
        case["dataset"] = "bfcl"
        case["available_tools"] = [
            {
                "name": "cellbio.get_proteins",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_compartment": {"type": "string"},
                        "include_description": {"type": "boolean"},
                    },
                    "required": ["cell_compartment"],
                },
            }
        ]
        case["gold"] = {
            "should_call_tools": True,
            "tool_calls": [
                {
                    "name": "cellbio.get_proteins",
                    "arguments": {
                        "cell_compartment": "plasma membrane",
                        "include_description": "",
                        "annotation_only": "drop",
                    },
                }
            ],
        }
        result = base_result()
        result.update(
            answer="",
            evidence=[],
            claims=[],
            tool_calls=[
                {
                    "name": "cellbio.get_proteins",
                    "arguments": {"cell_compartment": "plasma membrane"},
                }
            ],
            protocol={"tool_call_valid": True},
        )
        metrics = evaluate_agent_case(case, result)["metrics"]
        self.assertTrue(metrics["tool_call_exact_match"])

    def test_no_tool_false_positive_and_protocol_leak_are_visible(self) -> None:
        case = base_case(track="tool_protocol")
        case["gold"] = {"should_call_tools": False, "tool_calls": []}
        result = base_result()
        result["answer"] = '<tool_result>{"status":"ok"}</tool_result>'
        result["evidence"] = []
        result["claims"] = []
        result["tool_calls"] = [
            {"name": "web_search", "arguments": {"query": "你好"}}
        ]
        row = evaluate_agent_case(case, result)
        self.assertTrue(row["metrics"]["tool_false_positive"])
        self.assertFalse(row["metrics"]["tool_needed_accuracy"])
        self.assertTrue(row["metrics"]["protocol_leak"])
        self.assertIn("protocol_tag", row["diagnostics"]["leak_kinds"])

    def test_invalid_citation_and_unsupported_claim_are_counted(self) -> None:
        result = base_result()
        result["answer"] = "星河实验室 [W9]"
        result["claims"][0]["citations"] = []
        result["claims"][0]["supported"] = False
        metrics = evaluate_agent_case(base_case(), result)["metrics"]
        self.assertEqual(metrics["citation_validity_precision"], 0.0)
        self.assertEqual(metrics["citation_invalid_rate"], 1.0)
        self.assertEqual(metrics["claim_citation_coverage"], 0.0)
        self.assertEqual(metrics["unsupported_claim_rate"], 1.0)

    def test_grouped_citations_are_parsed_and_removed_from_answer(self) -> None:
        case = base_case()
        case["gold"]["answers"] = ["星河实验室维护青鸟项目"]
        case["gold"]["source_uris"] = [
            "https://example.com/project",
            "https://example.com/design",
        ]
        result = base_result()
        result["answer"] = "星河实验室维护青鸟项目 [W1, W2]"
        result["evidence"].append(
            {
                "id": "W2",
                "uri": "https://example.com/design",
            }
        )
        row = evaluate_agent_case(case, result)
        self.assertTrue(row["metrics"]["answer_exact_match"])
        self.assertEqual(row["diagnostics"]["citations"], ["W1", "W2"])
        self.assertEqual(row["metrics"]["citation_validity_precision"], 1.0)
        self.assertEqual(row["metrics"]["citation_source_recall"], 1.0)

    def test_source_domain_credit_does_not_fake_exact_url_credit(self) -> None:
        result = base_result()
        result["evidence"][0]["uri"] = "https://docs.example.com/other"
        metrics = evaluate_agent_case(base_case(), result)["metrics"]
        self.assertEqual(metrics["source_recall"], 0.0)
        self.assertEqual(metrics["source_domain_recall"], 1.0)
        self.assertEqual(metrics["citation_source_recall"], 0.0)
        self.assertEqual(metrics["citation_source_domain_recall"], 1.0)

    def test_aggregate_is_null_aware_and_never_builds_grand_score(self) -> None:
        good = evaluate_agent_case(base_case(), base_result())
        second_case = base_case(case_id="case-2")
        second_case["gold"].pop("source_uris")
        second_result = base_result(case_id="case-2")
        second_result["resources"]["latency_ms"] = 22000
        second = evaluate_agent_case(second_case, second_result)
        summary = aggregate_evaluations([good, second])
        self.assertIsNone(summary["grand_score"])
        self.assertEqual(
            summary["overall"]["metrics"]["source_recall"]["applicable"],
            1,
        )
        self.assertEqual(
            summary["overall"]["metrics"]["latency_ms"]["p95"],
            22000,
        )
        self.assertEqual(summary["by_context_bucket"]["8k-32k"]["cases"], 2)

    def test_paired_comparison_requires_same_cases_and_uses_direction(self) -> None:
        baseline = evaluate_agent_case(base_case(), base_result())
        baseline["metrics"]["latency_ms"] = 18000
        candidate = copy.deepcopy(baseline)
        candidate["metrics"]["latency_ms"] = 14000
        candidate["metrics"]["protocol_leak"] = False
        comparison = compare_evaluations([baseline], [candidate])
        latency = comparison["metrics"]["latency_ms"]
        self.assertEqual(latency["direction"], "lower")
        self.assertEqual(latency["wins"], 1)
        with self.assertRaisesRegex(ValueError, "identical case IDs"):
            compare_evaluations([baseline], [])

    def test_report_hashes_inputs_and_cli_writes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases.jsonl"
            results = root / "results.jsonl"
            report_path = root / "report.json"
            rows_path = root / "rows.jsonl"
            cases.write_text(
                json.dumps(base_case(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            results.write_text(
                json.dumps(base_result(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            report, rows = build_report(
                cases_path=cases,
                results_path=results,
            )
            self.assertEqual(report["coverage"]["rate"], 1.0)
            self.assertEqual(len(report["inputs"]["cases"]["sha256"]), 64)
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                main(
                    [
                        "--cases",
                        str(cases),
                        "--results",
                        str(results),
                        "--output",
                        str(report_path),
                        "--rows-output",
                        str(rows_path),
                    ]
                ),
                0,
            )
            self.assertTrue(report_path.is_file())
            self.assertEqual(len(rows_path.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
