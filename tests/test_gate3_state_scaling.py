from __future__ import annotations

from benchmarks.run_gate3_state_scaling import (
    _physical_batch,
    build_tasks,
    compare_arms,
    percentile,
    telemetry_summary,
    validate_rows,
)


def test_tasks_are_deterministic_unique_and_independent() -> None:
    first = build_tasks(100)
    assert first == build_tasks(100)
    assert len({task.owner_id for task in first}) == 100
    assert len({task.marker for task in first}) == 100
    assert all(task.marker in task.prompt for task in first)


def test_percentile_and_physical_batch_metrics() -> None:
    assert percentile([4, 1, 3, 2], 0.5) == 2
    assert percentile([4, 1, 3, 2], 0.95) == 4
    assert _physical_batch(
        {"shape_counts": {"B1T1": 2, "B32T1": 3, "B8T31": 1}}
    ) == 32


def test_validation_detects_foreign_session_marker() -> None:
    tasks = build_tasks(2)
    rows = [
        {
            "task": 1,
            "state_id": "s1",
            "marker": tasks[0].marker,
            "status": "ok",
            "text": tasks[1].marker,
        },
        {
            "task": 2,
            "state_id": "s2",
            "marker": tasks[1].marker,
            "status": "ok",
            "text": tasks[1].marker,
        },
    ]
    assert "session_mix_task_1" in validate_rows(rows, tasks)


def test_compare_arms_requires_exact_greedy_output() -> None:
    serial = {
        "metrics": {
            "wall_seconds": 8,
            "aggregate_output_tokens_per_second": 1,
        },
        "rows": [{"task": 1, "token_ids": [7], "text": "x"}],
    }
    concurrent = {
        "metrics": {
            "wall_seconds": 2,
            "aggregate_output_tokens_per_second": 4,
        },
        "rows": [{"task": 1, "token_ids": [7], "text": "x"}],
    }
    result = compare_arms(serial, concurrent)
    assert result["all_exact"] is True
    assert result["concurrent_speedup"] == 4
    assert result["aggregate_tps_ratio"] == 4


def test_telemetry_summary_is_phase_scoped() -> None:
    samples = [
        {
            "phase": "b4-decode",
            "gpu_busy_pct": 80,
            "vram_used_bytes": 10,
            "resident_states": 4,
            "active_state_rows": 4,
            "decoding_rows": 4,
            "waiting_jobs": 0,
        },
        {
            "phase": "b4-decode",
            "gpu_busy_pct": 100,
            "vram_used_bytes": 20,
            "resident_states": 4,
            "active_state_rows": 2,
            "decoding_rows": 2,
            "waiting_jobs": 2,
        },
        {"phase": "other", "gpu_busy_pct": 0, "vram_used_bytes": 1},
    ]
    result = telemetry_summary(samples, phase="b4-decode")
    assert result["samples"] == 2
    assert result["gpu_busy_pct"]["mean"] == 90
    assert result["vram_used_bytes"]["peak"] == 20
    assert result["active_state_rows_peak"] == 4
