from __future__ import annotations

import json
import unittest

from benchmarks.evaluate_s_level_bench import (
    DEFAULT_TARGETS,
    MEASUREMENT_SCHEMA,
    evaluate,
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

if __name__ == "__main__":
    unittest.main()
