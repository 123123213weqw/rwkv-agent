#!/usr/bin/env python3
"""Paired exact-page benchmark for bounded anchor-aware site navigation.

The control arm is every raw URL from every call in the frozen Dev snapshot.
The candidate arm non-destructively appends links observed while traversing a
small first-party graph.  Traversal never receives Gold; evaluation begins only
after both URL sets are final.
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
from urllib.parse import urlsplit

import aiohttp

from benchmarks.agent_benchmark_metrics import canonical_uri
from benchmarks.agent_benchmark_schema import load_jsonl
from benchmarks.retrieval_snapshot import load_snapshots
from rwkv_search.realtime.anchor_navigation import (
    AnchorPageCache,
    discover_anchor_navigation,
)


SCHEMA_VERSION = "rwkv-agent-anchor-navigation-ab.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _host(value: str) -> str:
    return (urlsplit(value).hostname or "").casefold().removeprefix("www.")


def _same_host_or_subdomain(url: str, expected: str) -> bool:
    actual = _host(url)
    return bool(
        actual
        and expected
        and (actual == expected or actual.endswith("." + expected))
    )


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
) -> dict[str, Any]:
    """Build a traversal plan without consulting the case Gold object."""

    calls = [
        dict(call)
        for call in snapshot.get("calls") or ()
        if isinstance(call, Mapping)
    ]
    raw = [
        dict(item)
        for call in calls
        for item in call.get("raw_candidates") or ()
        if isinstance(item, Mapping)
    ]
    control = _uris(raw)
    root_url = str(dict(case.get("metadata") or {}).get("root_url") or "")
    scope_host = _host(root_url)
    seeds = [
        uri
        for uri in control
        if _same_host_or_subdomain(uri, scope_host)
    ][:2]
    query_views = [
        str(call.get("effective_query") or call.get("query") or "")
        for call in calls
        if str(call.get("effective_query") or call.get("query") or "").strip()
    ]
    return {
        "case_id": str(case.get("id") or ""),
        "language": str(case.get("language") or "unknown"),
        "root_url": root_url,
        "question": str(case.get("prompt") or ""),
        "query_views": query_views,
        "seed_urls": seeds,
        "search_invoked": bool(calls),
        "control_uris": control,
    }


def evaluate_case(
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    """Read Gold only after the traversal and control sets are immutable."""

    control = set(plan.get("control_uris") or ())
    navigation = set(discovery.get("candidate_uris") or ())
    union = control | navigation
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
        "query_views": list(plan.get("query_views") or ()),
        "seed_urls": list(plan.get("seed_urls") or ()),
        "search_invoked": bool(plan.get("search_invoked")),
        "error": str(discovery.get("error") or ""),
        "requests": list(discovery.get("requests") or ()),
        "fetched_urls": list(discovery.get("fetched_urls") or ()),
        "queue_wait_ms": float(discovery.get("queue_wait_ms") or 0.0),
        "elapsed_ms": float(discovery.get("elapsed_ms") or 0.0),
        "control_candidate_count": len(control),
        "navigation_candidate_count": len(navigation),
        "new_navigation_candidate_count": len(navigation - control),
        "gold_page_count": len(gold),
        "control_exact_page_hit": bool(control & gold),
        "union_exact_page_hit": bool(union & gold),
        "control_exact_page_recall": round(control_recall, 6),
        "union_exact_page_recall": round(union_recall, 6),
        "delta_exact_page_recall": round(delta, 6),
        "comparison": "win" if delta > 0 else "loss" if delta < 0 else "tie",
        "matched": {
            "control_gold": sorted(control & gold),
            "navigation_gold": sorted(navigation & gold),
            "union_gold": sorted(union & gold),
        },
        "navigation_candidates": list(discovery.get("candidates") or ()),
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
    network_request_rows = [
        request
        for request in request_rows
        if bool(request.get("network_request", True))
    ]
    latencies = sorted(float(row.get("elapsed_ms") or 0.0) for row in rows)
    queue_waits = sorted(float(row.get("queue_wait_ms") or 0.0) for row in rows)
    return {
        "cases": count,
        "search_invoked_cases": sum(bool(row.get("search_invoked")) for row in rows),
        "navigation_nonempty_cases": sum(
            int(row.get("navigation_candidate_count") or 0) > 0 for row in rows
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
        "network_requests": len(network_request_rows),
        "cache_hits": len(request_rows) - len(network_request_rows),
        "request_failures": sum(
            bool(row.get("error")) for row in network_request_rows
        ),
        "mean_requests_per_case": round(len(request_rows) / count, 6),
        "mean_network_requests_per_case": round(
            len(network_request_rows) / count,
            6,
        ),
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
        "errors": dict(
            sorted(Counter(str(row.get("error") or "none") for row in rows).items())
        ),
    }


async def run_discovery(
    plans: Sequence[Mapping[str, Any]],
    *,
    concurrency: int,
    timeout_seconds: float,
    max_page_fetches: int,
    max_frontier_per_page: int,
    max_depth: int,
) -> dict[str, dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    connector = aiohttp.TCPConnector(
        limit=max(1, int(concurrency)),
        limit_per_host=1,
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; RWKV-Agent-Research/0.3; "
            "+https://github.com/RWKV)"
        )
    }
    page_cache = AnchorPageCache()
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=None),
        headers=headers,
        trust_env=True,
    ) as session:

        async def one(plan: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
            case_id = str(plan.get("case_id") or "")
            root_url = str(plan.get("root_url") or "")
            if not root_url:
                return case_id, {
                    "candidate_uris": [],
                    "candidates": [],
                    "fetched_urls": [],
                    "requests": [],
                    "error": "missing_root_url",
                    "elapsed_ms": 0.0,
                    "queue_wait_ms": 0.0,
                }
            queued = time.perf_counter()
            async with semaphore:
                queue_wait_ms = (time.perf_counter() - queued) * 1000.0
                started = time.perf_counter()
                result = await discover_anchor_navigation(
                    session,
                    root_url=root_url,
                    seed_urls=tuple(str(value) for value in plan.get("seed_urls") or ()),
                    question=str(plan.get("question") or ""),
                    query_views=tuple(
                        str(value) for value in plan.get("query_views") or ()
                    ),
                    timeout_seconds=timeout_seconds,
                    max_page_fetches=max_page_fetches,
                    max_frontier_per_page=max_frontier_per_page,
                    max_depth=max_depth,
                    page_cache=page_cache,
                )
            candidates = [
                {
                    "url": item.url,
                    "title": item.title,
                    "snippet": item.context,
                    "parent_url": item.parent_url,
                    "position": item.position,
                    "pagination": item.pagination,
                }
                for item in result.candidates
            ]
            return case_id, {
                "candidate_uris": _uris(candidates),
                "candidates": candidates,
                "fetched_urls": list(result.fetched_urls),
                "requests": list(result.requests),
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
    timeout_seconds: float = 7.0,
    max_page_fetches: int = 2,
    max_frontier_per_page: int = 24,
    max_depth: int = 1,
) -> dict[str, Any]:
    cases = load_jsonl(cases_path, kind="case")
    snapshots = load_snapshots(snapshots_path)
    snapshot_by_id = {str(row["case_id"]): row for row in snapshots}
    if set(snapshot_by_id) != {str(case["id"]) for case in cases}:
        raise ValueError("case and retrieval-snapshot IDs must match exactly")
    plans = [build_case_plan(case, snapshot_by_id[str(case["id"])]) for case in cases]
    discovery = asyncio.run(
        run_discovery(
            plans,
            concurrency=concurrency,
            timeout_seconds=timeout_seconds,
            max_page_fetches=max_page_fetches,
            max_frontier_per_page=max_frontier_per_page,
            max_depth=max_depth,
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
        "candidate": "control_plus_bounded_anchor_context_navigation",
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
            "max_page_fetches_per_case": int(max_page_fetches),
            "max_frontier_per_page": int(max_frontier_per_page),
            "max_depth": int(max_depth),
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
    parser.add_argument("--timeout-seconds", type=float, default=7.0)
    parser.add_argument("--max-page-fetches", type=int, default=2)
    parser.add_argument("--max-frontier-per-page", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=1)
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.max_page_fetches < 1:
        parser.error("concurrency and page fetch budget must be positive")
    summary = run(
        cases_path=args.cases.expanduser().resolve(),
        snapshots_path=args.snapshots.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        max_page_fetches=args.max_page_fetches,
        max_frontier_per_page=args.max_frontier_per_page,
        max_depth=args.max_depth,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
