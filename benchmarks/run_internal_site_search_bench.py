#!/usr/bin/env python3
"""Paired exact-page Discovery benchmark for first-party site search.

The control is every raw candidate from every frozen Agent call.  The candidate
arm appends results obtained from a search form discovered on the case root
site.  It does not fetch result pages or call a model, and Gold is read only
after the network plan has completed.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import aiohttp

from benchmarks.agent_benchmark_metrics import canonical_uri
from benchmarks.agent_benchmark_schema import load_jsonl
from benchmarks.retrieval_snapshot import load_snapshots
from rwkv_search.realtime.internal_site_search import (
    build_internal_search_queries,
    discover_internal_site_search,
)


SCHEMA_VERSION = "rwkv-agent-internal-site-search-ab.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _uris(items: Iterable[Mapping[str, Any]]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        uri = canonical_uri(str(item.get("url") or item.get("uri") or ""))
        if not uri or uri in seen:
            continue
        seen.add(uri)
        output.append(uri)
    return output


def build_case_plan(
    case: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    max_queries: int = 3,
) -> dict[str, Any]:
    calls = [
        dict(call)
        for call in snapshot.get("calls") or ()
        if isinstance(call, Mapping)
    ]
    execution_queries = [
        str(call.get("effective_query") or call.get("query") or "")
        for call in calls
    ]
    queries = build_internal_search_queries(
        str(case.get("prompt") or ""),
        execution_queries,
        max_queries=max_queries,
    )
    raw = [
        dict(item)
        for call in calls
        for item in call.get("raw_candidates") or ()
        if isinstance(item, Mapping)
    ]
    return {
        "case_id": str(case.get("id") or ""),
        "language": str(case.get("language") or "unknown"),
        "root_url": str(dict(case.get("metadata") or {}).get("root_url") or ""),
        "original_query": str(case.get("prompt") or ""),
        "execution_queries": execution_queries,
        "queries": list(queries),
        "search_invoked": bool(calls),
        "control_uris": _uris(raw),
    }


def evaluate_case(
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only after control and candidate URL sets are immutable."""

    control = set(plan.get("control_uris") or ())
    internal = set(discovery.get("candidate_uris") or ())
    union = control | internal
    gold = {
        canonical_uri(str(value))
        for value in dict(case.get("gold") or {}).get("source_uris") or ()
        if str(value).strip()
    }

    def recall(actual: set[str]) -> float:
        return len(actual & gold) / len(gold) if gold else 1.0

    control_recall = recall(control)
    union_recall = recall(union)
    delta = union_recall - control_recall
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(plan.get("case_id") or ""),
        "language": str(plan.get("language") or "unknown"),
        "root_url": str(plan.get("root_url") or ""),
        "planned_queries": list(plan.get("queries") or ()),
        "queries": list(
            discovery.get("queries") or plan.get("queries") or ()
        ),
        "search_invoked": bool(plan.get("search_invoked")),
        "capability": str(discovery.get("capability") or "none"),
        "error": str(discovery.get("error") or ""),
        "requests": list(discovery.get("requests") or ()),
        "queue_wait_ms": float(discovery.get("queue_wait_ms") or 0.0),
        "elapsed_ms": float(discovery.get("elapsed_ms") or 0.0),
        "control_candidate_count": len(control),
        "internal_candidate_count": len(internal),
        "new_internal_candidate_count": len(internal - control),
        "gold_page_count": len(gold),
        "control_exact_page_hit": bool(control & gold),
        "union_exact_page_hit": bool(union & gold),
        "control_exact_page_recall": round(control_recall, 6),
        "union_exact_page_recall": round(union_recall, 6),
        "delta_exact_page_recall": round(delta, 6),
        "comparison": "win" if delta > 0 else "loss" if delta < 0 else "tie",
        "matched": {
            "control_gold": sorted(control & gold),
            "internal_gold": sorted(internal & gold),
            "union_gold": sorted(union & gold),
        },
        "internal_candidates": list(discovery.get("candidates") or ()),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "control_exact_page_hit_rate": 0.0,
            "union_exact_page_hit_rate": 0.0,
            "control_exact_page_macro_recall": 0.0,
            "union_exact_page_macro_recall": 0.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
        }
    count = len(rows)
    request_rows = [
        dict(request)
        for row in rows
        for request in row.get("requests") or ()
        if isinstance(request, Mapping)
    ]
    latencies = sorted(float(row.get("elapsed_ms") or 0.0) for row in rows)
    queue_waits = sorted(float(row.get("queue_wait_ms") or 0.0) for row in rows)
    return {
        "cases": count,
        "search_invoked_cases": sum(bool(row.get("search_invoked")) for row in rows),
        "capability_detected_cases": sum(
            str(row.get("capability") or "none") != "none" for row in rows
        ),
        "internal_nonempty_cases": sum(
            int(row.get("internal_candidate_count") or 0) > 0 for row in rows
        ),
        "control_exact_page_hit_rate": round(
            sum(bool(row.get("control_exact_page_hit")) for row in rows) / count,
            6,
        ),
        "union_exact_page_hit_rate": round(
            sum(bool(row.get("union_exact_page_hit")) for row in rows) / count,
            6,
        ),
        "control_exact_page_macro_recall": round(
            sum(float(row.get("control_exact_page_recall") or 0.0) for row in rows)
            / count,
            6,
        ),
        "union_exact_page_macro_recall": round(
            sum(float(row.get("union_exact_page_recall") or 0.0) for row in rows)
            / count,
            6,
        ),
        "wins": sum(row.get("comparison") == "win" for row in rows),
        "losses": sum(row.get("comparison") == "loss" for row in rows),
        "ties": sum(row.get("comparison") == "tie" for row in rows),
        "requests": len(request_rows),
        "request_failures": sum(bool(row.get("error")) for row in request_rows),
        "mean_requests_per_case": round(len(request_rows) / count, 6),
        "latency_ms": {
            "mean": round(sum(latencies) / count, 3),
            "p95": round(latencies[min(count - 1, int(count * 0.95))], 3),
            "max": round(latencies[-1], 3),
        },
        "queue_wait_ms": {
            "mean": round(sum(queue_waits) / count, 3),
            "p95": round(queue_waits[min(count - 1, int(count * 0.95))], 3),
            "max": round(queue_waits[-1], 3),
        },
        "capabilities": dict(
            sorted(Counter(str(row.get("capability") or "none") for row in rows).items())
        ),
        "errors": dict(
            sorted(Counter(str(row.get("error") or "none") for row in rows).items())
        ),
    }


async def run_discovery(
    plans: Sequence[Mapping[str, Any]],
    *,
    concurrency: int,
    timeout_seconds: float,
    max_results: int,
) -> dict[str, dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    connector = aiohttp.TCPConnector(
        limit=max(1, int(concurrency)),
        limit_per_host=1,
    )
    timeout = aiohttp.ClientTimeout(total=None)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; RWKV-Agent-Research/0.3; "
            "+https://github.com/RWKV)"
        )
    }
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers=headers,
        trust_env=True,
    ) as session:

        async def one(plan: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
            case_id = str(plan.get("case_id") or "")
            root_url = str(plan.get("root_url") or "")
            queries = tuple(str(value) for value in plan.get("queries") or ())
            if not root_url or not queries:
                return case_id, {
                    "candidate_uris": [],
                    "candidates": [],
                    "queries": list(queries),
                    "requests": [],
                    "capability": "none",
                    "error": "missing_root_or_query",
                    "elapsed_ms": 0.0,
                    "queue_wait_ms": 0.0,
                }
            queued = time.perf_counter()
            async with semaphore:
                queue_wait_ms = (time.perf_counter() - queued) * 1000.0
                started = time.perf_counter()
                result = await discover_internal_site_search(
                    session,
                    root_url=root_url,
                    queries=queries,
                    original_query=str(plan.get("original_query") or ""),
                    execution_queries=tuple(
                        str(value)
                        for value in plan.get("execution_queries") or ()
                    ),
                    timeout_seconds=timeout_seconds,
                    max_results=max_results,
                )
            candidates = [
                {
                    "url": item.url,
                    "title": item.title,
                    "snippet": item.snippet,
                    "query": item.query,
                    "protocol": item.protocol,
                }
                for item in result.candidates
            ]
            protocols = sorted({form.protocol for form in result.forms})
            return case_id, {
                "candidate_uris": _uris(candidates),
                "candidates": candidates,
                "queries": list(result.queries),
                "requests": list(result.requests),
                "capability": "+".join(protocols) if protocols else "none",
                "error": result.error,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "queue_wait_ms": round(queue_wait_ms, 3),
            }

        return dict(await asyncio.gather(*(one(plan) for plan in plans)))


def _dump_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def run(
    *,
    cases_path: Path,
    snapshots_path: Path,
    output_dir: Path,
    concurrency: int = 4,
    timeout_seconds: float = 8.0,
    max_queries: int = 3,
    max_results: int = 40,
) -> dict[str, Any]:
    cases = load_jsonl(cases_path, kind="case")
    snapshots = load_snapshots(snapshots_path)
    snapshot_by_id = {str(row["case_id"]): row for row in snapshots}
    if set(snapshot_by_id) != {str(case["id"]) for case in cases}:
        raise ValueError("case and retrieval-snapshot IDs must match exactly")
    plans = [
        build_case_plan(
            case,
            snapshot_by_id[str(case["id"])],
            max_queries=max_queries,
        )
        for case in cases
    ]
    discovery = asyncio.run(
        run_discovery(
            plans,
            concurrency=concurrency,
            timeout_seconds=timeout_seconds,
            max_results=max_results,
        )
    )
    case_by_id = {str(case["id"]): case for case in cases}
    rows = [
        evaluate_case(case_by_id[str(plan["case_id"])], plan, discovery[str(plan["case_id"])])
        for plan in plans
    ]
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[str(row.get("language") or "unknown")].append(row)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "offline_control_live_candidate": True,
        "control": "all_raw_candidates_from_all_frozen_agent_calls",
        "candidate": "control_plus_detected_first_party_site_search_results",
        "gold_read_after_network_and_both_candidate_sets": True,
        "production_enabled": False,
        "inputs": {
            "cases": str(cases_path.resolve()),
            "cases_sha256": _sha256(cases_path),
            "snapshots": str(snapshots_path.resolve()),
            "snapshots_sha256": _sha256(snapshots_path),
        },
        "limits": {
            "concurrency": int(concurrency),
            "per_host_concurrency": 1,
            "timeout_seconds_per_request": float(timeout_seconds),
            "max_queries_per_case": int(max_queries),
            "max_results_per_case": int(max_results),
            "page_result_fetches": 0,
            "model_calls": 0,
        },
        "overall": _aggregate(rows),
        "by_language": {
            language: _aggregate(values)
            for language, values in sorted(by_language.items())
        },
    }
    _dump_jsonl(output_dir / "rows.jsonl", rows)
    _dump_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument("--max-queries", type=int, default=3)
    parser.add_argument("--max-results", type=int, default=40)
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if not 1 <= args.max_queries <= 5:
        parser.error("--max-queries must be between 1 and 5")
    summary = run(
        cases_path=args.cases.expanduser().resolve(),
        snapshots_path=args.snapshots.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        max_queries=args.max_queries,
        max_results=args.max_results,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
