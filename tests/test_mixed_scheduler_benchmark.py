from __future__ import annotations

import unittest

from benchmarks.run_mixed_scheduler_benchmark import (
    WORKLOAD_BLOCK,
    build_work_items,
    compare_reference,
    counter_delta,
    percentile,
    required_state_capacity,
    summarize_rows,
)


class MixedSchedulerBenchmarkTests(unittest.TestCase):
    def test_workload_is_deterministic_and_contains_all_runtime_classes(self) -> None:
        first = build_work_items(16)
        second = build_work_items(16)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(
            {item.kind for item in first},
            {"completion", "gate", "state_chat", "state_branch"},
        )
        self.assertEqual(
            [item.kind for item in first[: len(WORKLOAD_BLOCK)]],
            list(WORKLOAD_BLOCK),
        )
        self.assertEqual(required_state_capacity(first), 10)

    def test_concurrency_must_preserve_frozen_workload_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive multiple"):
            build_work_items(7)

    def test_percentile_is_nearest_rank(self) -> None:
        self.assertEqual(percentile([4, 1, 3, 2], 0.5), 2)
        self.assertEqual(percentile([4, 1, 3, 2], 0.95), 4)
        self.assertEqual(percentile([], 0.95), 0)

    def test_summary_and_reference_comparison_are_kind_aware(self) -> None:
        rows = [
            {
                "item_id": "a",
                "kind": "completion",
                "state_rows": 0,
                "latency_ms": 10,
                "fingerprint": {"token_ids": [1]},
            },
            {
                "item_id": "b",
                "kind": "gate",
                "state_rows": 0,
                "latency_ms": 20,
                "fingerprint": {
                    "label": "x",
                    "scores": {"x": 1.0, "y": 0.0},
                },
            },
        ]
        reference = summarize_rows(rows, wall_seconds=0.1, released_states=0)
        mixed_rows = [dict(row, latency_ms=row["latency_ms"] * 2) for row in rows]
        mixed = summarize_rows(
            mixed_rows,
            wall_seconds=0.1,
            released_states=0,
        )
        comparison = compare_reference(reference, mixed)
        self.assertTrue(comparison["all_exact"])
        self.assertEqual(comparison["exact"], 2)
        self.assertEqual(
            comparison["mixed_p95_over_isolated"],
            {"completion": 2.0, "gate": 2.0},
        )
        self.assertEqual(comparison["gate_label_mismatches"], [])
        self.assertEqual(comparison["gate_score_max_abs_delta"], 0)

    def test_gate_comparison_requires_same_label_and_bounded_score_drift(self) -> None:
        reference = summarize_rows(
            [
                {
                    "item_id": "gate",
                    "kind": "gate",
                    "state_rows": 0,
                    "latency_ms": 1,
                    "fingerprint": {
                        "label": "tool",
                        "scores": {"tool": 2.0, "chat": 1.0},
                    },
                }
            ],
            wall_seconds=0.1,
            released_states=0,
        )
        within = summarize_rows(
            [
                {
                    "item_id": "gate",
                    "kind": "gate",
                    "state_rows": 0,
                    "latency_ms": 1,
                    "fingerprint": {
                        "label": "tool",
                        "scores": {"tool": 2.05, "chat": 0.95},
                    },
                }
            ],
            wall_seconds=0.1,
            released_states=0,
        )
        self.assertTrue(compare_reference(reference, within)["all_exact"])
        changed = summarize_rows(
            [
                {
                    "item_id": "gate",
                    "kind": "gate",
                    "state_rows": 0,
                    "latency_ms": 1,
                    "fingerprint": {
                        "label": "chat",
                        "scores": {"tool": 0.9, "chat": 1.1},
                    },
                }
            ],
            wall_seconds=0.1,
            released_states=0,
        )
        self.assertFalse(compare_reference(reference, changed)["all_exact"])

    def test_counter_delta_ignores_unchanged_metrics(self) -> None:
        before = {"inference": {"metrics": {"done": 2, "failed": 0}}}
        after = {"inference": {"metrics": {"done": 5, "failed": 0}}}
        self.assertEqual(
            counter_delta(before, after, ("inference", "metrics")),
            {"done": 3},
        )


if __name__ == "__main__":
    unittest.main()
