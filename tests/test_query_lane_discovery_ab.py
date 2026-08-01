from __future__ import annotations

from benchmarks.run_query_lane_discovery_ab import (
    build_case_plan,
    evaluate_arm,
    fuse,
    summarize,
)


def test_case_plan_never_uses_gold_to_build_queries() -> None:
    case = {
        "id": "case",
        "prompt": "Which project won the 2024 award?",
        "language": "en",
        "metadata": {"root_url": "https://example.org/"},
        "gold": {"source_uris": ["https://example.org/private-target"]},
    }
    snapshot = {
        "case_id": "case",
        "calls": [{"effective_query": "project award site:example.org"}],
    }
    plan = build_case_plan(case, snapshot)
    assert "private-target" not in plan["original_query_lane"]
    assert plan["original_query_lane"].endswith("site:example.org")


def test_case_plan_supports_host_token_lane_without_reading_gold() -> None:
    case = {
        "id": "case-2",
        "language": "zh",
        "prompt": "2024年全民禁毒宣传教育主题班会是谁主持的？",
        "metadata": {"root_url": "https://cs.example.edu.cn/"},
        "gold": {"source_uris": ["https://private-target.invalid/answer"]},
    }
    snapshot = {
        "case_id": "case-2",
        "calls": [
            {
                "effective_query": (
                    "全民禁毒宣传教育 主持人 site:cs.example.edu.cn"
                )
            }
        ],
    }
    plan = build_case_plan(case, snapshot, lane_mode="host_token")
    assert plan["lane_mode"] == "host_token"
    assert "site:" not in plan["original_query_lane"].casefold()
    assert "private-target" not in plan["original_query_lane"]
    assert plan["original_query_lane"].endswith("cs.example.edu.cn")


def test_case_plan_supports_anchor_phrase_lane() -> None:
    case = {
        "id": "case-3",
        "language": "zh",
        "prompt": "2024年学院举办的“全民禁毒宣传教育”主题班会是谁主持的？",
        "metadata": {"root_url": "https://cs.example.edu.cn/"},
        "gold": {"source_uris": ["https://private-target.invalid/answer"]},
    }
    snapshot = {
        "case_id": "case-3",
        "calls": [
            {"effective_query": "全民禁毒宣传教育 主持人 site:cs.example.edu.cn"}
        ],
    }
    plan = build_case_plan(case, snapshot, lane_mode="anchor_phrase")
    assert plan["lane_mode"] == "anchor_phrase"
    assert '"全民禁毒宣传教育"' in plan["original_query_lane"]
    assert "private-target" not in plan["original_query_lane"]


def test_page_two_lane_reuses_control_query_without_gold() -> None:
    case = {
        "id": "case-4",
        "language": "en",
        "prompt": "Which project won the award?",
        "metadata": {"root_url": "https://example.org/"},
        "gold": {"source_uris": ["https://private-target.invalid/answer"]},
    }
    snapshot = {
        "case_id": "case-4",
        "calls": [{"effective_query": "project award site:example.org"}],
    }
    plan = build_case_plan(case, snapshot, lane_mode="page_two")
    assert plan["original_query_lane"] == plan["control_query"]
    assert "private-target" not in plan["original_query_lane"]


def test_fuse_and_evaluate_use_union_without_dropping_control() -> None:
    control = [
        {"url": "https://example.org/control", "engine": "dogpile"}
    ]
    lane = [{"url": "https://example.org/target", "engine": "yandex"}]
    union = fuse([control, lane])
    case = {
        "gold": {"source_uris": ["https://example.org/target"]},
    }
    assert evaluate_arm(case, control)["target_page_hit_at_20"] is False
    assert evaluate_arm(case, union)["target_page_hit_at_20"] is True


def test_summary_separates_search_not_invoked_and_paired_wins() -> None:
    metric = {
        "nonempty": True,
        "domain_hit_at_10": True,
        "target_page_hit_at_20": False,
        "target_page_recall_at_20": 0.0,
        "matched_gold_urls": [],
    }
    winning = {
        **metric,
        "target_page_hit_at_20": True,
        "target_page_recall_at_20": 1.0,
    }
    summary = summarize(
        [
            {
                "status": "evaluated",
                "arms": {
                    "control": metric,
                    "lane_only": winning,
                    "union": winning,
                },
            },
            {"status": "search_not_invoked", "arms": {}},
        ]
    )
    assert summary["search_not_invoked"] == 1
    assert summary["paired"] == {
        "union_wins": 1,
        "union_losses": 0,
        "union_ties": 0,
    }
