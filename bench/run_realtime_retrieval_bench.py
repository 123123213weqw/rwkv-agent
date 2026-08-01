#!/usr/bin/env python3
"""Evaluate live URL discovery and fetched-result recall without answer generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

if __package__:
    from .query_formation import load_p4_plans, resolve_p4_plan_query
    from .retrieval_metrics import aggregate, evaluate_candidate_stage, evaluate_case
    from .retrieval_schema import load_cases as load_validated_cases
else:
    from query_formation import load_p4_plans, resolve_p4_plan_query
    from retrieval_metrics import aggregate, evaluate_candidate_stage, evaluate_case
    from retrieval_schema import load_cases as load_validated_cases
from rwkv_search.config import AppConfig
from rwkv_search.realtime import RealtimeSearchEngine
from rwkv_search.pipeline.query_compiler import QueryHints, normalize_source_preference
from rwkv_search.search_request import SearchRequestBuilder


def load_cases(path: Path, case_ids: Sequence[str], limit: int) -> List[Dict[str, Any]]:
    rows = load_validated_cases(path)
    if case_ids:
        wanted = set(case_ids)
        available = {str(row["id"]) for row in rows}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(f"unknown case ids: {', '.join(missing)}")
        rows = [row for row in rows if row["id"] in wanted]
    if limit > 0:
        rows = rows[:limit]
    return rows


def apply_model_queries(
    cases: Sequence[Mapping[str, Any]], path: Path
) -> List[Dict[str, Any]]:
    """Attach runtime-equivalent frozen planner queries by canonical case id."""
    plans = load_p4_plans(path)
    missing = [str(case["id"]) for case in cases if str(case["id"]) not in plans]
    if missing:
        raise ValueError(f"missing model queries: {', '.join(missing)}")
    output: List[Dict[str, Any]] = []
    for case in cases:
        value = dict(case)
        resolution = resolve_p4_plan_query(
            str(case.get("query") or ""), plans[str(case["id"])]
        )
        value["model_query"] = str(resolution["query"])
        value["model_query_resolution"] = resolution
        output.append(value)
    return output


def serialize_result(item: Any, position: int) -> Dict[str, Any]:
    content = str(getattr(item, "content", "") or "")
    if hasattr(item, "to_dict"):
        value = item.to_dict(include_content=False)
    elif isinstance(item, Mapping):
        value = dict(item)
        content = str(value.pop("content", "") or content)
    else:
        value = {"value": str(item)}
    value["position"] = position
    value["content_length"] = len(content)
    return value


def _event_elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def run_case(
    case: Mapping[str, Any],
    engine: Any,
    builder: SearchRequestBuilder,
) -> Dict[str, Any]:
    started = time.perf_counter()
    query = str(case["query"])
    model_query = str(case.get("model_query") or query)
    hints = QueryHints(
        freshness=str(case.get("freshness") or "stable"),
        source_preference=normalize_source_preference(case.get("source_policy")),
        depth=str(case.get("depth") or "single"),
    )
    request = builder.build(query, model_query, hints=hints)
    effective_freshness = request.freshness
    effective_depth = request.depth
    candidates: List[Dict[str, Any]] = []
    initial_candidates: List[Dict[str, Any]] = []
    post_pivot_candidates: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    fetches: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}

    try:
        stream = engine.search_events(
            request.raw_query,
            request.execution_queries,
            freshness=effective_freshness,
            depth=effective_depth,
            source_preference=request.source_preference,
            include_candidates=True,
        )
        for event in stream:
            event_type = str(event.get("type") or "unknown")
            event_record: Dict[str, Any] = {
                "type": event_type,
                "elapsed_ms": _event_elapsed_ms(started),
            }
            if event_type == "discovery_progress":
                progress = dict(event.get("progress") or {})
                candidates = [dict(value) for value in progress.pop("candidates", ())]
                initial_candidates = [
                    dict(value)
                    for value in progress.pop("initial_candidates", candidates)
                ]
                for position, candidate in enumerate(candidates, 1):
                    candidate.setdefault("position", position)
                for position, candidate in enumerate(initial_candidates, 1):
                    candidate.setdefault("position", position)
                post_pivot_candidates = [dict(value) for value in candidates]
                event_record["progress"] = progress
            elif event_type == "discovery_enrichment":
                progress = dict(event.get("progress") or {})
                enriched = [dict(value) for value in progress.pop("candidates", ())]
                progress.pop("new_candidates", None)
                if enriched:
                    candidates = enriched
                    for position, candidate in enumerate(candidates, 1):
                        candidate.setdefault("position", position)
                event_record["progress"] = progress
            elif event_type == "fetch_progress":
                progress = dict(event.get("progress") or {})
                fetch = progress.pop("fetch", None)
                if isinstance(fetch, Mapping):
                    fetch_record = dict(fetch)
                    fetch_record["position"] = len(fetches) + 1
                    fetches.append(fetch_record)
                event_record["progress"] = progress
            elif event_type == "realtime_result":
                results = [
                    serialize_result(item, position)
                    for position, item in enumerate(event.get("results") or (), 1)
                ]
                stats = dict(event.get("stats") or {})
                event_record["stats"] = stats
            else:
                event_record.update(
                    {key: value for key, value in event.items() if key != "type"}
                )
            events.append(event_record)
    except Exception as exc:
        events.append(
            {
                "type": "runner_error",
                "elapsed_ms": _event_elapsed_ms(started),
                "error_type": type(exc).__name__,
                "message": str(exc)[:500],
            }
        )

    elapsed_ms = _event_elapsed_ms(started)
    record: Dict[str, Any] = {
        "schema_version": "realtime-retrieval-run.v1",
        "id": case["id"],
        "query": query,
        "language": case.get("language"),
        "category": case.get("category"),
        "freshness": case.get("freshness"),
        "source_policy": case.get("source_policy"),
        "expected_domains_any": list(case.get("expected_domains_any", ())),
        "target_url_patterns_any": list(case.get("target_url_patterns_any", ())),
        "forbidden_result_types": list(case.get("forbidden_result_types", ())),
        "model_query_resolution": dict(case.get("model_query_resolution") or {}),
        "search_request": request.to_dict(),
        "execution": {
            "freshness": effective_freshness,
            "depth": effective_depth,
            "queries": list(request.execution_queries),
        },
        "initial_candidates": initial_candidates,
        "post_pivot_candidates": post_pivot_candidates,
        "candidates": candidates,
        "fetches": fetches,
        "results": results,
        "events": events,
        "stats": stats,
        "total_elapsed_ms": elapsed_ms,
    }
    record["metrics"] = evaluate_case(case, candidates, results, stats)
    record["candidate_stage_metrics"] = {
        "initial": evaluate_candidate_stage(case, initial_candidates),
        "post_pivot": evaluate_candidate_stage(case, post_pivot_candidates),
        "final": evaluate_candidate_stage(case, candidates),
    }
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.json")
    parser.add_argument("--bench", default="bench/realtime_web_retrieval.jsonl")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-queries", default="")
    parser.add_argument("--output", default="bench/runs/realtime_retrieval_v1.jsonl")
    parser.add_argument(
        "--summary", default="bench/runs/realtime_retrieval_v1_summary.json"
    )
    args = parser.parse_args()

    bench_path = Path(args.bench)
    cases = load_cases(bench_path, args.case_id, args.limit)
    if args.model_queries:
        cases = apply_model_queries(cases, Path(args.model_queries))
    config = AppConfig.load(args.config)
    config.realtime_search.enabled = True
    engine = RealtimeSearchEngine(config.realtime_search, config.search)
    builder = SearchRequestBuilder()
    records: List[Dict[str, Any]] = []
    output, summary_path = Path(args.output), Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("w", encoding="utf-8") as handle:
            for case in cases:
                record = run_case(case, engine, builder)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                metrics = record["metrics"]
                print(
                    f"{case['id']} candidate@10={int(metrics['candidate_domain_hit_at_10'])} "
                    f"result@10={int(metrics['result_domain_hit_at_10'])} "
                    f"fetch={metrics['fetch_succeeded']}/{metrics['fetch_attempted']} "
                    f"garbage={metrics['garbage_result_rate']:.2%} "
                    f"elapsed={record['total_elapsed_ms']:.0f}ms",
                    flush=True,
                )
    finally:
        engine.close()

    summary = aggregate(records)
    summary.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bench": str(bench_path),
            "bench_sha256": _sha256(bench_path),
            "config": str(Path(args.config)),
            "case_count": len(records),
            "case_ids": [str(row["id"]) for row in records],
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
