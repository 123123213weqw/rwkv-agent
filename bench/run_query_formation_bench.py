#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import aiohttp

if __package__:
    from .query_formation import (
        STRATEGIES,
        evaluate_discovery,
        load_p4_plans,
        strategy_queries,
        summarize_records,
    )
    from .retrieval_schema import load_cases
else:
    from query_formation import (
        STRATEGIES,
        evaluate_discovery,
        load_p4_plans,
        strategy_queries,
        summarize_records,
    )
    from retrieval_schema import load_cases
from rwkv_search.analysis import QueryAnalyzer
from rwkv_search.config import AppConfig
from rwkv_search.realtime.discovery import URLDiscovery


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialize_candidate(value: Any, position: int) -> Dict[str, Any]:
    return {
        "position": position,
        "url": str(value.url),
        "title": str(value.title),
        "snippet": str(value.snippet),
        "engine": str(value.engine),
        "rank": int(value.rank),
        "rrf_score": round(float(value.rrf_score), 8),
        "published_hint": value.published_hint,
    }


async def run(
    cases: Sequence[Mapping[str, Any]],
    p4_plans: Mapping[str, Mapping[str, Any]],
    config: AppConfig,
    *,
    delay_seconds: float,
) -> List[Dict[str, Any]]:
    realtime = config.realtime_search
    timeout = aiohttp.ClientTimeout(
        total=None,
        connect=realtime.connect_timeout_seconds,
        sock_connect=realtime.connect_timeout_seconds,
        sock_read=realtime.page_timeout_seconds,
    )
    connector = aiohttp.TCPConnector(
        limit=max(1, realtime.global_concurrency),
        family=socket.AF_INET if realtime.force_ipv4 else socket.AF_UNSPEC,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    headers = {
        "User-Agent": realtime.user_agent,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.2",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
        "Accept-Encoding": "gzip, deflate",
    }
    analyzer = QueryAnalyzer()
    records: List[Dict[str, Any]] = []
    discovery_cache: Dict[
        tuple[str, tuple[str, ...]], tuple[List[Any], List[Dict[str, str]], float, str]
    ] = {}
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        auto_decompress=False,
        headers=headers,
        trust_env=False,
    ) as session:
        discovery = URLDiscovery(realtime, session)
        for case_index, case in enumerate(cases):
            planned = strategy_queries(case, p4_plans.get(str(case["id"])), analyzer=analyzer)
            for strategy in STRATEGIES:
                queries = planned[strategy]
                diagnostics: List[Dict[str, str]] = []
                candidates: List[Any] = []
                shared_with = ""
                network_request = False
                cache_key = (str(case["freshness"]), tuple(queries))
                wall_started = time.perf_counter()
                if queries and cache_key in discovery_cache:
                    cached_candidates, cached_diagnostics, elapsed_ms, shared_with = (
                        discovery_cache[cache_key]
                    )
                    candidates = list(cached_candidates)
                    diagnostics = [dict(item) for item in cached_diagnostics]
                elif queries:
                    network_request = True
                    started = time.perf_counter()
                    try:
                        candidates = await discovery.discover(
                            queries,
                            freshness=str(case["freshness"]),
                            max_candidates=30,
                            diagnostics=diagnostics,
                        )
                    except Exception as exc:
                        diagnostics.append(
                            {
                                "query": " | ".join(queries),
                                "engine": "runner",
                                "error_type": type(exc).__name__,
                                "message": str(exc)[:300],
                            }
                        )
                    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
                    discovery_cache[cache_key] = (
                        list(candidates),
                        [dict(item) for item in diagnostics],
                        elapsed_ms,
                        f"{case['id']}:{strategy}",
                    )
                else:
                    elapsed_ms = 0.0
                wall_elapsed_ms = round((time.perf_counter() - wall_started) * 1000.0, 3)
                serialized = [
                    _serialize_candidate(item, position)
                    for position, item in enumerate(candidates, 1)
                ]
                record = {
                    "schema_version": "query-formation-run.v1",
                    "id": case["id"],
                    "query": case["query"],
                    "language": case["language"],
                    "category": case["category"],
                    "freshness": case["freshness"],
                    "source_policy": case["source_policy"],
                    "expected_domains_any": list(case["expected_domains_any"]),
                    "target_url_patterns_any": list(case["target_url_patterns_any"]),
                    "strategy": strategy,
                    "queries": list(queries),
                    "query_valid": bool(queries),
                    "candidates": serialized,
                    "diagnostics": diagnostics,
                    "elapsed_ms": elapsed_ms,
                    "wall_elapsed_ms": wall_elapsed_ms,
                    "network_request": network_request,
                    "shared_with": shared_with,
                    "metrics": evaluate_discovery(case, serialized),
                }
                records.append(record)
                metrics = record["metrics"]
                print(
                    f"{case['id']} {strategy:<5} q={list(queries)!r} "
                    f"domain@10={int(metrics['domain_hit_at_10'])} "
                    f"target@20={int(metrics['target_page_hit_at_20'])} "
                    f"candidates={metrics['candidate_count']} elapsed={elapsed_ms:.0f}ms",
                    flush=True,
                )
                if network_request and delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/server-v100.json")
    parser.add_argument("--bench", default="bench/realtime_web_retrieval.jsonl")
    parser.add_argument("--p4-plans", required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--output", default="bench/runs/query_formation_v1.jsonl")
    parser.add_argument("--summary", default="bench/runs/query_formation_v1_summary.json")
    args = parser.parse_args()

    bench_path, p4_path = Path(args.bench), Path(args.p4_plans)
    cases = load_cases(bench_path)
    if args.case_id:
        wanted = set(args.case_id)
        missing = wanted - {str(row["id"]) for row in cases}
        if missing:
            raise ValueError(f"unknown case ids: {', '.join(sorted(missing))}")
        cases = [row for row in cases if row["id"] in wanted]
    if args.limit > 0:
        cases = cases[: args.limit]
    p4_plans = load_p4_plans(p4_path)
    missing_plans = [str(row["id"]) for row in cases if str(row["id"]) not in p4_plans]
    if missing_plans:
        raise ValueError(f"missing P4 plans: {', '.join(missing_plans)}")

    config = AppConfig.load(args.config)
    config.realtime_search.enabled = True
    records = asyncio.run(
        run(cases, p4_plans, config, delay_seconds=max(0.0, args.delay_seconds))
    )
    output, summary_path = Path(args.output), Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_records(records)
    p4_strict = sum(bool(p4_plans[str(row["id"])].get("strict_success")) for row in cases)
    summary.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bench": str(bench_path),
            "bench_sha256": _sha256(bench_path),
            "p4_plans": str(p4_path),
            "p4_plans_sha256": _sha256(p4_path),
            "config": args.config,
            "case_count": len(cases),
            "p4_strict_success": p4_strict,
            "p4_strict_success_rate": round(p4_strict / max(1, len(cases)), 4),
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["deltas"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
