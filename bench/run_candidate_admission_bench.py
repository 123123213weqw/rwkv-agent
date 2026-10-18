#!/usr/bin/env python3
"""Replay frozen discovery candidates through the pre-fetch admission layer."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlsplit

if __package__:
    from .query_formation import evaluate_discovery
    from .retrieval_schema import load_cases
else:
    from query_formation import evaluate_discovery
    from retrieval_schema import load_cases
from rwkv_search.realtime.candidate_ranker import admit_candidates
from rwkv_search.realtime.types import DiscoveredURL
from rwkv_search.pipeline.query_compiler import normalize_source_preference


def _candidate(value: Dict[str, Any]) -> DiscoveredURL:
    return DiscoveredURL(
        url=str(value.get("url") or ""),
        title=str(value.get("title") or ""),
        snippet=str(value.get("snippet") or ""),
        engine=str(value.get("engine") or "unknown"),
        rank=int(value.get("rank") or 0),
        published_hint=value.get("published_hint"),
        rrf_score=float(value.get("rrf_score") or 0.0),
    )


def _serialize(value: DiscoveredURL, position: int) -> Dict[str, Any]:
    return {
        "position": position,
        "url": value.url,
        "title": value.title,
        "snippet": value.snippet,
        "engine": value.engine,
        "rank": value.rank,
        "rrf_score": value.rrf_score,
        "candidate_score": value.candidate_score,
        "score_components": value.score_components,
    }


def _rate(rows: List[Dict[str, Any]], section: str, metric: str) -> float:
    return round(
        sum(bool(row[section].get(metric)) for row in rows) / max(1, len(rows)), 4
    )


def _reciprocal_domain_rank(row: Dict[str, Any], key: str) -> float:
    expected = [
        str(value).casefold().removeprefix("www.")
        for value in row["expected_domains_any"]
    ]
    for position, item in enumerate(row[key], 1):
        host = (urlsplit(str(item.get("url") or "")).hostname or "").casefold()
        host = host.removeprefix("www.")
        if any(host == value or host.endswith("." + value) for value in expected):
            return 1.0 / position
    return 0.0


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = (
        "domain_hit_at_5",
        "domain_hit_at_10",
        "domain_hit_at_20",
        "target_page_hit_at_10",
        "target_page_hit_at_20",
    )
    rejected = Counter(
        reason
        for row in rows
        for reason, count in row["rejection_counts"].items()
        for _ in range(int(count))
    )
    return {
        "schema_version": "candidate-admission-bench.v1",
        "case_count": len(rows),
        "before": {metric: _rate(rows, "before_metrics", metric) for metric in metrics},
        "after": {metric: _rate(rows, "after_metrics", metric) for metric in metrics},
        "average_candidates_before": round(
            sum(len(row["before_candidates"]) for row in rows) / max(1, len(rows)), 3
        ),
        "average_candidates_after": round(
            sum(len(row["after_candidates"]) for row in rows) / max(1, len(rows)), 3
        ),
        "rejected_count": sum(rejected.values()),
        "rejection_counts": dict(sorted(rejected.items())),
        "domain_mrr_before": round(
            sum(_reciprocal_domain_rank(row, "before_candidates") for row in rows)
            / max(1, len(rows)),
            4,
        ),
        "domain_mrr_after": round(
            sum(_reciprocal_domain_rank(row, "after_candidates") for row in rows)
            / max(1, len(rows)),
            4,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="bench/runs/query_formation_v1.jsonl")
    parser.add_argument("--cases", default="bench/realtime_web_retrieval.jsonl")
    parser.add_argument("--strategy", default="p4")
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--per-domain-limit", type=int, default=3)
    parser.add_argument("--output", default="bench/runs/candidate_admission_p4_v1.jsonl")
    parser.add_argument(
        "--summary", default="bench/runs/candidate_admission_p4_v1_summary.json"
    )
    args = parser.parse_args()

    cases = {row["id"]: row for row in load_cases(Path(args.cases))}
    source_rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: List[Dict[str, Any]] = []
    for source in source_rows:
        if source.get("strategy") != args.strategy:
            continue
        case = cases[str(source["id"])]
        source_preference = normalize_source_preference(case.get("source_policy"))
        before = [_candidate(value) for value in source.get("candidates", ())]
        admission = admit_candidates(
            str(source["query"]),
            [str(value) for value in source.get("queries", ())],
            before,
            max_candidates=args.max_candidates,
            per_domain_limit=args.per_domain_limit,
            source_preference=source_preference,
        )
        before_serialized = [
            _serialize(value, position) for position, value in enumerate(before, 1)
        ]
        after_serialized = [
            _serialize(value, position)
            for position, value in enumerate(admission.admitted, 1)
        ]
        rows.append(
            {
                "schema_version": "candidate-admission-run.v1",
                "id": source["id"],
                "query": source["query"],
                "queries": source.get("queries", []),
                "source_preference": source_preference,
                "expected_domains_any": case.get("expected_domains_any", []),
                "before_candidates": before_serialized,
                "after_candidates": after_serialized,
                "rejected_candidates": [
                    {
                        "url": value.url,
                        "title": value.title,
                        "reasons": value.rejection_reasons,
                    }
                    for value in admission.rejected
                ],
                "rejection_counts": admission.rejection_counts,
                "before_metrics": evaluate_discovery(case, before_serialized),
                "after_metrics": evaluate_discovery(case, after_serialized),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = summarize(rows)
    Path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
