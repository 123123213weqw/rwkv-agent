from __future__ import annotations

import json

import pytest

from bench.query_formation import (
    QueryFormationError,
    evaluate_discovery,
    load_p4_plans,
    resolve_p4_plan_query,
    strategy_queries,
    summarize_records,
)


def _case() -> dict:
    return {
        "id": "retrieval-en-001",
        "query": "What is the current stable Python release according to python.org?",
        "language": "en",
        "category": "software_release",
        "expected_domains_any": ["python.org"],
        "target_url_patterns_any": ["/downloads/"],
        "forbidden_result_types": [],
    }


def test_strategy_queries_keep_p4_query_isolated() -> None:
    queries = strategy_queries(
        _case(),
        {"strict_success": True, "model_query": "Python latest stable release official"},
    )
    assert queries["raw"] == (
        "What is the current stable Python release according to python.org",
    )
    assert queries["rules"]
    assert queries["p4"] == ("Python latest stable release official",)
    assert queries["raw"][0] not in queries["p4"]


def test_invalid_p4_plan_uses_runtime_raw_fallback() -> None:
    queries = strategy_queries(
        _case(), {"strict_success": False, "model_query": "Python latest"}
    )
    assert queries["p4"] == (
        "What is the current stable Python release according to python.org",
    )


def test_effective_query_takes_priority_over_raw_model_output() -> None:
    resolution = resolve_p4_plan_query(
        _case()["query"],
        {
            "strict_success": True,
            "model_query": "stale model output",
            "effective_query": "Python stable release official",
        },
    )
    assert resolution["query"] == "Python stable release official"
    assert resolution["source"] == "effective_query"
    assert resolution["fallback_to_raw"] is False


def test_legacy_plan_with_invented_year_uses_current_runtime_repair() -> None:
    raw = "国家卫健委最新发布的法定传染病疫情概况。"
    resolution = resolve_p4_plan_query(
        raw,
        {
            "strict_success": True,
            "fallback_to_raw": True,
            "model_query": "国家卫健委 法定传染病 疫情概况 2025年",
        },
    )
    assert resolution["query"] == "国家卫健委 法定传染病 疫情概况"
    assert resolution["source"] == "repaired_model_query"
    assert resolution["fallback_to_raw"] is False
    assert resolution["fallback_reason"] == ""
    evaluation = resolution["constraint_evaluation"]
    assert evaluation["repair_applied"] is True
    assert evaluation["removed_absolute_terms"] == ["2025"]
    assert evaluation["original_evaluation"]["introduced_years"] == ["2025"]


def test_load_p4_plans_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "plans.jsonl"
    row = {"id": "retrieval-en-001", "strict_success": True, "model_query": "Python"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(QueryFormationError, match="duplicate id"):
        load_p4_plans(path)


def test_discovery_metrics_and_strategy_summary() -> None:
    case = _case()
    candidates = [
        {"url": "https://example.com/no", "title": "No"},
        {"url": "https://www.python.org/downloads/", "title": "Python downloads"},
    ]
    metrics = evaluate_discovery(case, candidates)
    assert metrics["domain_hit_at_5"] is True
    assert metrics["target_page_hit_at_10"] is True

    records = []
    for strategy, hit in (("raw", False), ("rules", False), ("p4", True)):
        records.append(
            {
                "strategy": strategy,
                "language": "en",
                "category": "software_release",
                "queries": ["query"],
                "elapsed_ms": 10,
                "metrics": {
                    "candidate_count": 2,
                    "nonempty_candidates": True,
                    "domain_hit_at_5": hit,
                    "domain_hit_at_10": hit,
                    "domain_hit_at_20": hit,
                    "target_page_hit_at_10": hit,
                    "target_page_hit_at_20": hit,
                },
            }
        )
    summary = summarize_records(records)
    assert summary["overall"]["p4"]["domain_hit_at_10_rate"] == 1.0
    assert summary["deltas"]["p4_vs_raw"]["domain_hit_at_10_rate"] == 1.0
