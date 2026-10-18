from __future__ import annotations

import json
import unittest

from benchmarks.evaluate_s_level_bench import (
    DEFAULT_TARGETS,
    MEASUREMENT_SCHEMA,
    evaluate,
    measurements_from_fitgen,
)


def targets() -> dict[str, object]:
    return json.loads(DEFAULT_TARGETS.read_text(encoding="utf-8"))


def passing_measurements(profile: str = "s_level") -> dict[str, object]:
    definitions = targets()["metrics"]
    assert isinstance(definitions, dict)
    values = {
        name: definition[profile]
        for name, definition in definitions.items()
    }
    subgroup_values = {
        name: definition[profile]
        for name, definition in definitions.items()
        if definition.get("subgroup_gate")
    }
    return {
        "schema_version": MEASUREMENT_SCHEMA,
        "cases": 500,
        "live_repetitions": 3,
        "metrics": values,
        "language_groups": {"zh": dict(subgroup_values), "en": dict(subgroup_values)},
    }


class SLevelBenchTests(unittest.TestCase):
    def test_exact_s_level_targets_pass_without_grand_score(self) -> None:
        report = evaluate(passing_measurements(), targets(), profile="s_level")
        self.assertTrue(report["release_passed"], report["failed_gates"])
        self.assertIsNone(report["grand_score"])
        self.assertEqual(report["gate_counts"]["missing"], 0)

    def test_missing_metric_fails_closed(self) -> None:
        measurements = passing_measurements()
        del measurements["metrics"]["exact_page_recall_at_20"]
        report = evaluate(measurements, targets(), profile="s_level")
        self.assertFalse(report["release_passed"])
        self.assertIn(
            "metric.exact_page_recall_at_20",
            report["missing_gate_ids"],
        )

    def test_upper_bound_and_language_floor_are_hard_gates(self) -> None:
        measurements = passing_measurements()
        measurements["metrics"]["unsupported_claim_rate"] = 0.02
        measurements["language_groups"]["zh"]["exact_page_recall_at_20"] = 0.85
        report = evaluate(measurements, targets(), profile="s_level")
        failed = {item["gate_id"] for item in report["failed_gates"]}
        self.assertIn("metric.unsupported_claim_rate", failed)
        self.assertIn("language.zh.exact_page_recall_at_20", failed)

    def test_fitgen_adapter_maps_only_semantically_supported_metrics(self) -> None:
        summary = {
            "cases": 80,
            "reliability": {"state_leak_count": 0},
            "unified_metrics": {
                "citation_validity_precision": {"kind": "macro_rate", "mean": 0.65},
                "citation_exact_page_recall": {"kind": "macro_rate", "mean": 0.23125},
                "claim_citation_coverage": {"kind": "macro_rate", "mean": 0.873214},
                "unsupported_claim_rate": {"kind": "macro_rate", "mean": 0.126786},
                "result_status_ok": {"kind": "rate", "rate": 1.0},
                "latency_ms": {"kind": "distribution", "p50": 28248.949, "p95": 38223.079},
                "answer_token_f1": {"kind": "macro_rate", "mean": 0.221927},
            },
        }
        funnel = {
            "stage_hit_rates": {
                "domain_candidate_hit": 0.95,
                "exact_raw_candidate_hit": 0.5,
                "exact_final_evidence_hit": 0.3875,
                "search_invoked": 0.95,
            },
            "stage_macro_recalls": {
                "raw_candidate_recall": 0.39375,
                "final_evidence_recall": 0.30625,
            },
            "by_language": {
                "zh": {"stage_hit_rates": {"domain_candidate_hit": 0.977778}},
                "en": {"stage_hit_rates": {"domain_candidate_hit": 0.962963}},
            },
        }
        converted = measurements_from_fitgen(summary, funnel)
        metrics = converted["metrics"]
        self.assertEqual(metrics["exact_page_recall_at_20"], 0.39375)
        self.assertEqual(metrics["final_evidence_exact_recall"], 0.30625)
        self.assertEqual(metrics["discovered_to_evidence_retention_rate"], 0.775)
        self.assertEqual(metrics["search_false_negative_rate"], 0.05)
        self.assertNotIn("answer_factual_accuracy", metrics)
        self.assertNotIn("answer_token_f1", metrics)


if __name__ == "__main__":
    unittest.main()
