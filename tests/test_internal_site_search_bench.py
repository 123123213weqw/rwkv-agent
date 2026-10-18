from __future__ import annotations

from benchmarks.run_internal_site_search_bench import build_case_plan, evaluate_case


def test_case_plan_uses_all_frozen_calls_without_gold() -> None:
    case = {
        "id": "case-1",
        "prompt": "Find the original announcement",
        "language": "en",
        "metadata": {"root_url": "https://example.org/"},
        "gold": {"source_uris": ["https://private.invalid/gold"]},
    }
    snapshot = {
        "calls": [
            {
                "effective_query": "original announcement site:example.org",
                "raw_candidates": [{"url": "https://example.org/first"}],
            },
            {
                "effective_query": "announcement archive site:example.org",
                "raw_candidates": [{"url": "https://example.org/later"}],
            },
        ]
    }

    plan = build_case_plan(case, snapshot)

    assert plan["control_uris"] == [
        "https://example.org/first",
        "https://example.org/later",
    ]
    assert "private.invalid" not in " ".join(plan["queries"])


def test_evaluation_reads_gold_after_non_destructive_union() -> None:
    case = {
        "id": "case-2",
        "language": "en",
        "gold": {"source_uris": ["https://example.org/gold"]},
    }
    plan = {
        "case_id": "case-2",
        "language": "en",
        "root_url": "https://example.org/",
        "queries": ["announcement"],
        "search_invoked": True,
        "control_uris": ["https://example.org/control"],
    }
    discovery = {
        "candidate_uris": ["https://example.org/gold"],
        "candidates": [],
        "requests": [],
        "capability": "html_form",
        "error": "",
        "elapsed_ms": 1.0,
    }

    row = evaluate_case(case, plan, discovery)

    assert row["comparison"] == "win"
    assert row["control_exact_page_recall"] == 0.0
    assert row["union_exact_page_recall"] == 1.0
