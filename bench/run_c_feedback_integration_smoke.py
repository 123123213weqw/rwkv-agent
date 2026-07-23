#!/usr/bin/env python3
"""Run the default-off C feedback planner through the real retrieval engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

if __package__:
    from .retrieval_metrics import aggregate
    from .run_realtime_retrieval_bench import load_cases, run_case
else:
    from retrieval_metrics import aggregate
    from run_realtime_retrieval_bench import load_cases, run_case
from rwkv_search.config import AppConfig
from rwkv_search.g1i_native import FastRWKV7Completion
from rwkv_search.realtime import RealtimeSearchEngine
from rwkv_search.search_reasoning import CFeedbackPlanner
from rwkv_search.search_request import SearchRequestBuilder


DEFAULT_CASE_IDS = (
    "retrieval-zh-013",
    "retrieval-zh-020",
    "retrieval-en-003",
    "retrieval-en-007",
    "retrieval-en-014",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integration_checks(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    violations: List[Dict[str, str]] = []
    feedback_executed = 0
    feedback_provenance_cases = 0
    for record in records:
        case_id = str(record["id"])
        stats = dict(record.get("stats") or {})
        requests = int(stats.get("discovery_request_count") or 0)
        if requests > 2:
            violations.append(
                {"id": case_id, "reason": f"discovery_request_count={requests} > 2"}
            )
        if stats.get("feedback_query_executed"):
            feedback_executed += 1
        if any(
            "model_feedback" in (candidate.get("discovery_stages") or [])
            for candidate in record.get("candidates") or []
        ):
            feedback_provenance_cases += 1
        plans = list(stats.get("model_search_plans") or [])
        if any(
            forbidden in plan
            for plan in plans
            for forbidden in ("raw_output", "reasoning", "token_ids")
        ):
            violations.append({"id": case_id, "reason": "private model trace leaked"})
        event_types = [str(event.get("type")) for event in record.get("events") or []]
        if event_types.count("discovery_progress") != 1:
            violations.append({"id": case_id, "reason": "discovery stage repeated"})
        if event_types.count("realtime_result") != 1:
            violations.append({"id": case_id, "reason": "result stage repeated"})
    if records and feedback_executed == 0:
        violations.append({"id": "*", "reason": "no feedback query was executed"})
    if records and feedback_provenance_cases == 0:
        violations.append({"id": "*", "reason": "no feedback provenance was observed"})
    return {
        "passed": not violations,
        "case_count": len(records),
        "feedback_query_executed_cases": feedback_executed,
        "feedback_candidate_provenance_cases": feedback_provenance_cases,
        "max_discovery_requests": max(
            (int((record.get("stats") or {}).get("discovery_request_count") or 0)
             for record in records),
            default=0,
        ),
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bench-4080-c-feedback.json")
    parser.add_argument("--bench", default="bench/realtime_web_retrieval.jsonl")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--context", type=int, default=12288)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--planner-timeout", type=float, default=4.0)
    parser.add_argument(
        "--output", default="bench/runs/c_feedback_integration_smoke_v1.jsonl"
    )
    parser.add_argument(
        "--summary",
        default="bench/runs/c_feedback_integration_smoke_v1_summary.json",
    )
    args = parser.parse_args()

    bench_path = Path(args.bench)
    case_ids = args.case_id or list(DEFAULT_CASE_IDS)
    cases = load_cases(bench_path, case_ids, args.limit)
    config_path = Path(args.config)
    config = AppConfig.load(config_path)
    config.realtime_search.enabled = True
    model = FastRWKV7Completion(args.model, args.runtime, context=args.context)
    planner = CFeedbackPlanner(
        model,
        max_tokens=args.max_tokens,
        timeout_seconds=args.planner_timeout,
    )
    engine = RealtimeSearchEngine(
        config.realtime_search,
        config.search,
        feedback_planner=planner,
    )
    builder = SearchRequestBuilder()
    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    try:
        with output.open("w", encoding="utf-8") as handle:
            for case in cases:
                record = run_case(case, engine, builder)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                stats = record.get("stats") or {}
                print(
                    f"{case['id']} discoveries={stats.get('discovery_request_count', 0)} "
                    f"feedback={int(bool(stats.get('feedback_query_executed')))} "
                    f"fetch={stats.get('fetched', 0)}/{stats.get('attempted', 0)} "
                    f"elapsed={record['total_elapsed_ms']:.0f}ms",
                    flush=True,
                )
    finally:
        engine.close()

    checks = _integration_checks(records)
    summary = aggregate(records)
    summary.update(
        {
            "schema_version": "c-feedback-integration-smoke-summary.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bench": str(bench_path),
            "bench_sha256": _sha256(bench_path),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "case_ids": [str(record["id"]) for record in records],
            "protocol": {
                "planner": "CFeedbackPlanner",
                "tool_format": "p4_flat_web_search",
                "decoding": "greedy",
                "max_model_searches": 2,
                "max_feedback_searches": 1,
                "fetch_after_candidate_merge": True,
                "benchmark_labels_visible_to_runtime": False,
            },
            "integration_checks": checks,
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
