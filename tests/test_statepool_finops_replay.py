from __future__ import annotations

import json

import pytest

from scripts.statepool_finops_replay import ReplayConfig, build_replay, write_bundle


def test_replay_is_explicit_deterministic_and_uses_archived_evidence(tmp_path):
    replay = build_replay(ReplayConfig())

    assert replay["classification"] == "simulation_replay_from_measured_primitives"
    assert [row["scenario"] for row in replay["derived_results"]] == ["A", "B", "C"]
    assert replay["comparisons"] == {
        "statepool_gpu_hour_reduction_vs_sticky_percent": 45.151,
        "statepool_estimated_gpu_cost_reduction_vs_sticky_percent": 45.151,
        "statepool_prefill_tokens_avoided_vs_stateless": 51_200,
        "statepool_total_state_transfer_bytes": 3_873_383_100,
    }
    assert replay["derived_results"][2]["state_bytes_written"] == 2_582_255_400
    assert replay["derived_results"][2]["state_bytes_read"] == 1_291_127_700
    assert replay["not_measured"]

    write_bundle(replay, tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["comparisons"] == replay[
        "comparisons"
    ]
    assert (tmp_path / "results.csv").read_text().splitlines()[1].startswith(
        "A,sticky_worker,simulation_replay"
    )


@pytest.mark.parametrize(
    "config",
    [
        ReplayConfig(sessions=0),
        ReplayConfig(sessions_per_worker=0),
        ReplayConfig(windows=1),
        ReplayConfig(idle_seconds_between_windows=-1),
        ReplayConfig(gpu_price_cny_per_hour=-1),
    ],
)
def test_replay_rejects_invalid_inputs(config):
    with pytest.raises(ValueError):
        build_replay(config)
