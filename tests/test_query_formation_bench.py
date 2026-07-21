from __future__ import annotations

import json

import pytest

from bench.query_formation import (
    QueryFormationError,
    evaluate_discovery,
    load_p4_plans,
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


def test_invalid_p4_plan_produces_no_p4_query() -> None:
    queries = strategy_queries(
        _case(), {"strict_success": False, "model_query": "Python latest"}
    )
    assert queries["p4"] == ()


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
