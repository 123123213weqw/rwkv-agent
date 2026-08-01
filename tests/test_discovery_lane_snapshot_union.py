from __future__ import annotations

from benchmarks.replay_discovery_lane_union import (
    evaluate_replay,
    replay_candidate_sets,
)


def test_replay_uses_every_frozen_call_and_appends_lane_non_destructively() -> None:
    snapshot = {
        "calls": [
            {"raw_candidates": [{"url": "https://example.org/first"}]},
            {"raw_candidates": [{"url": "https://example.org/later"}]},
        ]
    }
    lane = {
        "scope_host": "example.org",
        "candidates": {
            "lane_only": [
                {"url": "https://example.org/later"},
                {"url": "https://example.org/new"},
                {"url": "https://noise.invalid/out-of-scope"},
            ]
        },
    }

    replay = replay_candidate_sets(snapshot, lane)

    assert replay["control_uris"] == [
        "https://example.org/first",
        "https://example.org/later",
    ]
    assert set(replay["control_uris"]).issubset(replay["union_uris"])
    assert replay["new_lane_in_scope_uris"] == ["https://example.org/new"]


def test_gold_is_not_visible_until_after_both_candidate_sets_exist() -> None:
    replay = {
        "search_invoked": True,
        "control_uris": ["https://example.org/control"],
        "lane_uris": ["https://example.org/gold"],
        "lane_in_scope_uris": ["https://example.org/gold"],
        "union_uris": [
            "https://example.org/control",
            "https://example.org/gold",
        ],
        "new_lane_uris": ["https://example.org/gold"],
        "new_lane_in_scope_uris": ["https://example.org/gold"],
    }
    case = {
        "id": "case-1",
        "language": "en",
        "gold": {"source_uris": ["https://example.org/gold"]},
    }

    row = evaluate_replay(case, replay)

    assert row["comparison"] == "win"
    assert row["control_exact_page_recall"] == 0.0
    assert row["union_exact_page_recall"] == 1.0
