from __future__ import annotations

from benchmarks.run_scoped_tavily_discovery_bench import (
    build_case_plan,
    compile_scoped_query,
    evaluate_case,
)


def test_compiles_site_operator_into_native_domain_scope() -> None:
    value = compile_scoped_query(
        "Calton Pu award site:cs.example.edu.cn",
        root_url="https://example.invalid/",
    )

    assert value is not None
    assert value.query == "Calton Pu award"
    assert value.domains == ("cs.example.edu.cn",)


def test_plan_uses_first_last_queries_and_never_reads_gold() -> None:
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
                "effective_query": "first query site:example.org",
                "raw_candidates": [{"url": "https://example.org/control"}],
            },
            {"effective_query": "middle query site:example.org"},
            {"effective_query": "last query site:example.org"},
        ]
    }

    plan = build_case_plan(case, snapshot, max_queries=2)

    assert [row["query"] for row in plan["queries"]] == ["first query", "last query"]
    assert plan["control_uris"] == ["https://example.org/control"]
    assert "private.invalid" not in repr(plan)


def test_evaluation_appends_tavily_without_deleting_control() -> None:
    case = {
        "id": "case-2",
        "language": "en",
        "gold": {"source_uris": ["https://example.org/gold"]},
    }
    plan = {
        "case_id": "case-2",
        "language": "en",
        "root_url": "https://example.org/",
        "search_invoked": True,
        "queries": [],
        "control_uris": ["https://example.org/control"],
    }
    discovery = {
        "candidate_uris": ["https://example.org/gold"],
        "candidates": [],
        "requests": [],
    }

    row = evaluate_case(case, plan, discovery)

    assert row["comparison"] == "win"
    assert row["control_exact_page_recall"] == 0.0
    assert row["union_exact_page_recall"] == 1.0
