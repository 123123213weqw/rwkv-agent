from __future__ import annotations

from benchmarks.run_anchor_navigation_bench import build_case_plan, evaluate_case


def test_plan_uses_all_frozen_calls_and_never_reads_gold() -> None:
    case = {
        "id": "case-1",
        "prompt": "Find the exact announcement",
        "language": "en",
        "metadata": {"root_url": "https://example.org/"},
        "gold": {"source_uris": ["https://private.invalid/gold"]},
    }
    snapshot = {
        "calls": [
            {
                "effective_query": "first query",
                "raw_candidates": [
                    {"url": "https://example.org/archive"},
                    {"url": "https://outside.invalid/result"},
                ],
            },
            {
                "effective_query": "second query",
                "raw_candidates": [{"url": "https://sub.example.org/news"}],
            },
        ]
    }

    plan = build_case_plan(case, snapshot)

    assert plan["control_uris"] == [
        "https://example.org/archive",
        "https://outside.invalid/result",
        "https://sub.example.org/news",
    ]
    assert plan["seed_urls"] == [
        "https://example.org/archive",
        "https://sub.example.org/news",
    ]
    assert "private.invalid" not in repr(plan)


def test_evaluation_is_a_non_destructive_union() -> None:
    case = {
        "id": "case-2",
        "language": "en",
        "gold": {"source_uris": ["https://example.org/gold"]},
    }
    plan = {
        "case_id": "case-2",
        "language": "en",
        "root_url": "https://example.org/",
        "query_views": ["announcement"],
        "seed_urls": [],
        "search_invoked": True,
        "control_uris": ["https://example.org/control"],
    }
    discovery = {
        "candidate_uris": ["https://example.org/gold"],
        "candidates": [],
        "fetched_urls": [],
        "requests": [],
        "error": "",
        "elapsed_ms": 1.0,
    }

    row = evaluate_case(case, plan, discovery)

    assert row["comparison"] == "win"
    assert row["control_exact_page_recall"] == 0.0
    assert row["union_exact_page_recall"] == 1.0
