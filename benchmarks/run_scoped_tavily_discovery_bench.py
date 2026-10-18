#!/usr/bin/env python3
"""Paired exact-page benchmark for domain-scoped Tavily discovery.

WebWalker queries carry a ``site:`` constraint.  The current structured-source
selector skips every such query, so merely configuring Tavily does not exercise
it.  This isolated benchmark converts that visible constraint into Tavily's
native ``include_domains`` field, removes the operator from the query text, and
appends results to every raw URL from the frozen Agent snapshot.  No result
pages or models are invoked, and Gold is read only after both URL sets are
complete.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from benchmarks.agent_benchmark_metrics import canonical_uri
from benchmarks.agent_benchmark_schema import load_jsonl
from benchmarks.retrieval_snapshot import load_snapshots
from rwkv_search.realtime.source_api import parse_tavily_results


SCHEMA_VERSION = "rwkv-agent-scoped-tavily-discovery-ab.v1"
_SITE_OPERATOR = re.compile(r"(?<!\w)site:([^\s]+)", re.I)


@dataclass(frozen=True)
class ScopedQuery:
    query: str
    domains: tuple[str, ...]
    source_query: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _host(value: str) -> str:
    raw = str(value or "").strip().strip(".,;:()[]{}<>\"'")
    if not raw:
        return ""
    if "://" not in raw:
        raw = "//" + raw
    return (urlsplit(raw).hostname or "").casefold().removeprefix("www.")


def compile_scoped_query(
    source_query: str,
    *,
    root_url: str = "",
    fallback_query: str = "",
) -> ScopedQuery | None:
    """Translate visible site constraints without consulting labels or Gold."""

    source = " ".join(str(source_query or "").split()).strip()
    domains: list[str] = []
    for match in _SITE_OPERATOR.finditer(source):
        host = _host(match.group(1))
        if host and host not in domains:
            domains.append(host)
    if not domains:
        host = _host(root_url)
        if host:
            domains.append(host)
    cleaned = " ".join(_SITE_OPERATOR.sub(" ", source).split()).strip()
    if not cleaned:
        cleaned = " ".join(str(fallback_query or "").split()).strip()
    if not cleaned or not domains:
        return None
    return ScopedQuery(
        query=cleaned[:400],
        domains=tuple(domains[:4]),
        source_query=source,
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
    *,
    max_queries: int = 2,
) -> dict[str, Any]:
    """Select first/last scoped views without ever reading ``case.gold``."""

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
    root_url = str(dict(case.get("metadata") or {}).get("root_url") or "")
    question = str(case.get("prompt") or "")
    compiled: list[ScopedQuery] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for call in calls:
        source = str(call.get("effective_query") or call.get("query") or "")
        query = compile_scoped_query(
            source,
            root_url=root_url,
            fallback_query=question,
        )
        if query is None:
            continue
        key = (query.query.casefold(), query.domains)
        if key in seen:
            continue
        seen.add(key)
        compiled.append(query)
    selected: list[ScopedQuery] = []
    if compiled and max_queries > 0:
        selected.append(compiled[0])
    if len(compiled) > 1 and max_queries > 1:
        selected.append(compiled[-1])
    return {
        "case_id": str(case.get("id") or ""),
        "language": str(case.get("language") or "unknown"),
        "root_url": root_url,
        "search_invoked": bool(calls),
        "control_uris": _uris(raw),
        "queries": [
            {
                "query": query.query,
                "domains": list(query.domains),
                "source_query": query.source_query,
            }
            for query in selected[: max(0, int(max_queries))]
        ],
    }


def _request_tavily(
    *,
    api_key: str,
    query: str,
    domains: Sequence[str],
    timeout_seconds: float,
    max_results: int,
) -> tuple[list[dict[str, Any]], int, int, str]:
    payload = json.dumps(
        {
            "query": query,
            "search_depth": "basic",
            "topic": "general",
            "max_results": max(1, min(20, int(max_results))),
            "include_domains": list(domains),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
    ).encode("utf-8")
    request = Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=max(0.25, timeout_seconds)) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
            status = int(getattr(response, "status", 200))
    except HTTPError as exc:
        return [], int(exc.code), 0, f"HTTP {exc.code}"
    except Exception as exc:
        return [], 0, 0, f"{type(exc).__name__}: {exc}"[:300]
    if len(raw) > 4 * 1024 * 1024:
        return [], status, len(raw), "response_exceeds_4_mib"
    try:
        value = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        return [], status, len(raw), f"JSONDecodeError: {exc}"[:300]
    candidates = parse_tavily_results(value if isinstance(value, Mapping) else {})
    return (
        [
            {
                "url": item.url,
                "title": item.title,
                "snippet": item.snippet,
                "rank": item.rank,
                "engine_score": item.engine_score,
            }
            for item in candidates
        ],
        status,
        len(raw),
        "",
    )


async def run_discovery(
    plans: Sequence[Mapping[str, Any]],
    *,
    api_key: str,
    concurrency: int,
    timeout_seconds: float,
    max_results: int,
) -> dict[str, dict[str, Any]]:
    if not api_key.strip():
        raise RuntimeError("TAVILY_API_KEY is required for this benchmark")
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    tasks: dict[tuple[str, tuple[str, ...]], asyncio.Task[dict[str, Any]]] = {}

    async def exchange(query: str, domains: tuple[str, ...]) -> dict[str, Any]:
        key = (query.casefold(), domains)
        task = tasks.get(key)
        if task is not None:
            result = dict(await asyncio.shield(task))
            result["cache_hit"] = True
            return result

        async def request_once() -> dict[str, Any]:
            queued = time.perf_counter()
            async with semaphore:
                queue_wait_ms = (time.perf_counter() - queued) * 1000.0
                started = time.perf_counter()
                candidates, status, response_bytes, error = await asyncio.to_thread(
                    _request_tavily,
                    api_key=api_key,
                    query=query,
                    domains=domains,
                    timeout_seconds=timeout_seconds,
                    max_results=max_results,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            return {
                "query": query,
                "domains": list(domains),
                "status": status,
                "response_bytes": response_bytes,
                "error": error,
                "elapsed_ms": round(elapsed_ms, 3),
                "queue_wait_ms": round(queue_wait_ms, 3),
                "cache_hit": False,
                "candidates": candidates,
            }

        task = asyncio.create_task(request_once())
        tasks[key] = task
        return dict(await asyncio.shield(task))

    async def one(plan: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        results = []
        for row in plan.get("queries") or ():
            if not isinstance(row, Mapping):
                continue
            results.append(
                await exchange(
                    str(row.get("query") or ""),
                    tuple(str(value) for value in row.get("domains") or ()),
                )
            )
        candidates = [
            dict(candidate)
            for result in results
            for candidate in result.get("candidates") or ()
            if isinstance(candidate, Mapping)
        ]
        return str(plan.get("case_id") or ""), {
            "candidate_uris": _uris(candidates),
            "candidates": candidates,
            "requests": [
                {key: value for key, value in result.items() if key != "candidates"}
                for result in results
            ],
        }

    return dict(await asyncio.gather(*(one(plan) for plan in plans)))


def evaluate_case(
    case: Mapping[str, Any],
    plan: Mapping[str, Any],
    discovery: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only after the network and both candidate sets are complete."""

    control = set(plan.get("control_uris") or ())
    tavily = set(discovery.get("candidate_uris") or ())
    union = control | tavily
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
        "search_invoked": bool(plan.get("search_invoked")),
        "queries": list(plan.get("queries") or ()),
        "requests": list(discovery.get("requests") or ()),
        "control_candidate_count": len(control),
        "tavily_candidate_count": len(tavily),
        "new_tavily_candidate_count": len(tavily - control),
        "gold_page_count": len(gold),
        "control_exact_page_hit": bool(control & gold),
        "union_exact_page_hit": bool(union & gold),
        "control_exact_page_recall": round(control_recall, 6),
        "union_exact_page_recall": round(union_recall, 6),
        "delta_exact_page_recall": round(delta, 6),
        "comparison": "win" if delta > 0 else "loss" if delta < 0 else "tie",
        "matched": {
            "control_gold": sorted(control & gold),
            "tavily_gold": sorted(tavily & gold),
            "union_gold": sorted(union & gold),
        },
        "tavily_candidates": list(discovery.get("candidates") or ()),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"cases": 0, "wins": 0, "losses": 0, "ties": 0}
    count = len(rows)
    requests = [
        dict(request)
        for row in rows
        for request in row.get("requests") or ()
        if isinstance(request, Mapping)
    ]
    network = [request for request in requests if not bool(request.get("cache_hit"))]
    latencies = sorted(float(request.get("elapsed_ms") or 0.0) for request in network)
    return {
        "cases": count,
        "search_invoked_cases": sum(bool(row.get("search_invoked")) for row in rows),
        "tavily_nonempty_cases": sum(
            int(row.get("tavily_candidate_count") or 0) > 0 for row in rows
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
        "logical_requests": len(requests),
        "network_requests": len(network),
        "cache_hits": len(requests) - len(network),
        "request_failures": sum(bool(request.get("error")) for request in network),
        "http_statuses": dict(
            sorted(Counter(int(request.get("status") or 0) for request in network).items())
        ),
        "mean_network_requests_per_case": round(len(network) / count, 6),
        "service_latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p95": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 3)
            if latencies
            else 0.0,
            "max": round(latencies[-1], 3) if latencies else 0.0,
        },
    }


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
    api_key: str,
    concurrency: int = 8,
    timeout_seconds: float = 8.0,
    max_queries: int = 2,
    max_results: int = 20,
) -> dict[str, Any]:
    cases = load_jsonl(cases_path, kind="case")
    snapshots = load_snapshots(snapshots_path)
    snapshot_by_id = {str(row["case_id"]): row for row in snapshots}
    if set(snapshot_by_id) != {str(case["id"]) for case in cases}:
        raise ValueError("case and retrieval-snapshot IDs must match exactly")
    plans = [
        build_case_plan(case, snapshot_by_id[str(case["id"])], max_queries=max_queries)
        for case in cases
    ]
    discovery = asyncio.run(
        run_discovery(
            plans,
            api_key=api_key,
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
        "control": "all_raw_candidates_from_all_frozen_agent_calls",
        "candidate": "control_plus_basic_tavily_include_domains_first_last_queries",
        "gold_read_after_network_and_both_candidate_sets": True,
        "production_enabled": False,
        "inputs": {
            "cases": str(cases_path.resolve()),
            "cases_sha256": _sha256(cases_path),
            "snapshots": str(snapshots_path.resolve()),
            "snapshots_sha256": _sha256(snapshots_path),
        },
        "limits": {
            "query_selection": "first_and_last_distinct_frozen_agent_queries",
            "max_queries_per_case": int(max_queries),
            "max_results_per_query": int(max_results),
            "search_depth": "basic",
            "credits_per_network_request": 1,
            "concurrency": int(concurrency),
            "timeout_seconds": float(timeout_seconds),
            "page_fetches": 0,
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
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument("--max-queries", type=int, default=2)
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--api-key-env", default="TAVILY_API_KEY")
    args = parser.parse_args(argv)
    if args.concurrency < 1 or not 1 <= args.max_queries <= 8:
        parser.error("concurrency must be positive and max-queries must be 1..8")
    summary = run(
        cases_path=args.cases.expanduser().resolve(),
        snapshots_path=args.snapshots.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        api_key=os.getenv(args.api_key_env, ""),
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        max_queries=args.max_queries,
        max_results=args.max_results,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
