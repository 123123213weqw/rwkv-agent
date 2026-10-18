#!/usr/bin/env python3
"""Paired live Discovery A/B for a complementary original-query lane.

Gold URLs are read only after both query arms have returned.  The runner does
not fetch result pages or call the answer model; it isolates URL discovery.
Raw rows contain URLs and therefore belong in a private, ignored run folder.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import aiohttp

from benchmarks.agent_benchmark_metrics import canonical_uri
from benchmarks.agent_benchmark_schema import load_jsonl
from benchmarks.retrieval_snapshot import load_snapshots
from rwkv_search.analysis.query import QueryAnalyzer
from rwkv_search.realtime.discovery import parse_searxng_results
from rwkv_search.realtime.precision_discovery import (
    build_anchor_phrase_query_lane,
    build_host_token_query_lane,
    build_original_query_lane,
)


SCHEMA_VERSION = "rwkv-agent-query-lane-discovery-ab.v1"
_CJK = re.compile(r"[\u3400-\u9fff]")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _host(value: str) -> str:
    return (urlsplit(str(value or "")).hostname or "").casefold().removeprefix(
        "www."
    )


def _domain_match(actual: str, expected: str) -> bool:
    return bool(
        actual
        and expected
        and (
            actual == expected
            or actual.endswith("." + expected)
            or expected.endswith("." + actual)
        )
    )


def build_case_plan(
    case: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    lane_mode: str = "site_operator",
    analyzer: QueryAnalyzer | None = None,
) -> dict[str, Any]:
    calls = [dict(value) for value in snapshot.get("calls") or ()]
    root_url = str(dict(case.get("metadata") or {}).get("root_url") or "")
    scope_host = _host(root_url)
    control = ""
    if calls:
        control = str(
            calls[0].get("effective_query") or calls[0].get("query") or ""
        ).strip()
    builders = {
        "anchor_phrase": build_anchor_phrase_query_lane,
        "site_operator": build_original_query_lane,
        "host_token": build_host_token_query_lane,
    }
    if lane_mode not in {*builders, "page_two"}:
        raise ValueError(f"unsupported query lane mode: {lane_mode}")
    lane = ""
    if control and scope_host:
        if lane_mode == "page_two":
            lane = control
        else:
            lane = builders[lane_mode](
                str(case.get("prompt") or ""),
                control,
                site=scope_host,
                analyzer=analyzer,
            )
    return {
        "case_id": str(case.get("id") or ""),
        "language": str(case.get("language") or "unknown"),
        "scope_host": scope_host,
        "control_query": control,
        "original_query_lane": lane,
        "lane_mode": lane_mode,
        "search_invoked": bool(calls),
    }


def fuse(groups: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group_index, group in enumerate(groups):
        for position, source in enumerate(group, 1):
            url = canonical_uri(str(source.get("url") or ""))
            if not url:
                continue
            contribution = 1.0 / (60.0 + position) + 0.002 / (group_index + 1)
            existing = merged.get(url)
            if existing is None:
                existing = dict(source)
                existing["url"] = url
                existing["rrf_score"] = 0.0
                existing["engines"] = []
                merged[url] = existing
            existing["rrf_score"] += contribution
            engine = str(source.get("engine") or "")
            if engine and engine not in existing["engines"]:
                existing["engines"].append(engine)
            if len(str(source.get("snippet") or "")) > len(
                str(existing.get("snippet") or "")
            ):
                existing["snippet"] = str(source.get("snippet") or "")
    return sorted(
        merged.values(),
        key=lambda value: (
            float(value.get("rrf_score") or 0.0),
            str(value.get("url") or ""),
        ),
        reverse=True,
    )


def evaluate_arm(
    case: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gold = {
        canonical_uri(str(value))
        for value in dict(case.get("gold") or {}).get("source_uris") or ()
        if str(value).strip()
    }
    gold_domains = {_host(value) for value in gold if _host(value)}
    top10 = [canonical_uri(str(value.get("url") or "")) for value in candidates[:10]]
    top20 = [canonical_uri(str(value.get("url") or "")) for value in candidates[:20]]
    matched = gold.intersection(top20)
    return {
        "nonempty": bool(candidates),
        "domain_hit_at_10": any(
            _domain_match(_host(actual), expected)
            for actual in top10
            for expected in gold_domains
        ),
        "target_page_hit_at_20": bool(matched),
        "target_page_recall_at_20": (
            round(len(matched) / len(gold), 6) if gold else 1.0
        ),
        "matched_gold_urls": sorted(matched),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluated = [value for value in rows if value.get("status") == "evaluated"]

    def mean(
        arm: str,
        metric: str,
        values: Sequence[Mapping[str, Any]] = evaluated,
    ) -> float:
        return round(
            sum(float(dict(dict(row["arms"])[arm])[metric]) for row in values)
            / max(1, len(values)),
            6,
        )

    def arm_summary(
        arm: str,
        values: Sequence[Mapping[str, Any]] = evaluated,
    ) -> dict[str, float]:
        return {
            "domain_hit_at_10": mean(arm, "domain_hit_at_10", values),
            "target_page_hit_at_20": mean(
                arm, "target_page_hit_at_20", values
            ),
            "target_page_recall_at_20": mean(
                arm, "target_page_recall_at_20", values
            ),
        }

    control_hits = [
        bool(dict(dict(row["arms"])["control"])["target_page_hit_at_20"])
        for row in evaluated
    ]
    union_hits = [
        bool(dict(dict(row["arms"])["union"])["target_page_hit_at_20"])
        for row in evaluated
    ]
    request_rows = [
        dict(request)
        for row in evaluated
        for request in row.get("requests") or ()
        if isinstance(request, Mapping)
    ]

    def distribution(name: str) -> dict[str, float]:
        values = sorted(float(row.get(name) or 0.0) for row in request_rows)
        if not values:
            return {"mean": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "mean": round(sum(values) / len(values), 3),
            "p95": round(values[min(len(values) - 1, int(len(values) * 0.95))], 3),
            "max": round(values[-1], 3),
        }

    languages = sorted({str(row.get("language") or "unknown") for row in evaluated})
    return {
        "schema_version": SCHEMA_VERSION,
        "cases": len(rows),
        "evaluated_cases": len(evaluated),
        "search_not_invoked": sum(
            value.get("status") == "search_not_invoked" for value in rows
        ),
        "control": arm_summary("control"),
        "lane_only": arm_summary("lane_only"),
        "union": arm_summary("union"),
        "paired": {
            "union_wins": sum(right and not left for left, right in zip(control_hits, union_hits)),
            "union_losses": sum(left and not right for left, right in zip(control_hits, union_hits)),
            "union_ties": sum(left == right for left, right in zip(control_hits, union_hits)),
        },
        "by_language": {
            language: {
                "cases": len(values),
                "control": arm_summary("control", values),
                "union": arm_summary("union", values),
            }
            for language in languages
            if (
                values := [
                    row
                    for row in evaluated
                    if str(row.get("language") or "unknown") == language
                ]
            )
        },
        "requests": {
            "count": len(request_rows),
            "failures": sum(bool(row.get("error")) for row in request_rows),
            "service_ms": distribution("elapsed_ms"),
            "queue_wait_ms": distribution("queue_wait_ms"),
        },
    }


def _query_engines(
    query: str,
    base: Sequence[str],
    language_engines: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    language = "zh" if _CJK.search(query) else "default"
    additions = language_engines.get(language) or language_engines.get("default", ())
    return tuple(dict.fromkeys([*base, *additions]))


async def run(
    cases: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    endpoint: str,
    engines: Sequence[str],
    language_engines: Mapping[str, Sequence[str]],
    concurrency: int,
    timeout_seconds: float,
    lane_mode: str = "site_operator",
) -> list[dict[str, Any]]:
    snapshot_by_id = {str(value["case_id"]): value for value in snapshots}
    analyzer = QueryAnalyzer(max_queries=1)
    plans = [
        build_case_plan(
            case,
            snapshot_by_id[str(case["id"])],
            lane_mode=lane_mode,
            analyzer=analyzer,
        )
        for case in cases
    ]
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def request(
        session: aiohttp.ClientSession,
        case_id: str,
        arm: str,
        query: str,
        engine: str,
    ) -> tuple[str, str, str, list[dict[str, Any]], float, float, str]:
        queued = time.monotonic()
        async with semaphore:
            queue_wait_ms = (time.monotonic() - queued) * 1000.0
            started = time.monotonic()
            try:
                async with session.get(
                    endpoint.rstrip("/") + "/search",
                    params={
                        "q": query,
                        "format": "json",
                        "language": "zh-CN" if _CJK.search(query) else "en",
                        "safesearch": "1",
                        "engines": engine,
                        "pageno": (
                            "2"
                            if lane_mode == "page_two" and arm == "lane_only"
                            else "1"
                        ),
                    },
                ) as response:
                    value = await response.json(content_type=None)
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}")
                    rows = [
                        {
                            "url": item.url,
                            "title": item.title,
                            "snippet": item.snippet,
                            "engine": engine,
                        }
                        for item in parse_searxng_results(value)
                    ]
                    return (
                        case_id,
                        arm,
                        engine,
                        rows,
                        round((time.monotonic() - started) * 1000.0, 3),
                        round(queue_wait_ms, 3),
                        "",
                    )
            except Exception as exc:
                return (
                    case_id,
                    arm,
                    engine,
                    [],
                    round((time.monotonic() - started) * 1000.0, 3),
                    round(queue_wait_ms, 3),
                    f"{type(exc).__name__}: {exc}"[:300],
                )

    timeout = aiohttp.ClientTimeout(total=max(0.2, timeout_seconds))
    connector = aiohttp.TCPConnector(limit=max(1, concurrency))
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        trust_env=True,
    ) as session:
        tasks = []
        for plan in plans:
            for arm, query in (
                ("control", str(plan["control_query"])),
                ("lane_only", str(plan["original_query_lane"])),
            ):
                if not query:
                    continue
                for engine in _query_engines(query, engines, language_engines):
                    tasks.append(
                        request(
                            session,
                            str(plan["case_id"]),
                            arm,
                            query,
                            engine,
                        )
                    )
        responses = await asyncio.gather(*tasks)

    grouped: dict[tuple[str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    diagnostics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case_id, arm, engine, values, elapsed_ms, queue_wait_ms, error in responses:
        grouped[(case_id, arm)].append(values)
        diagnostics[case_id].append(
            {
                "arm": arm,
                "engine": engine,
                "result_count": len(values),
                "elapsed_ms": elapsed_ms,
                "queue_wait_ms": queue_wait_ms,
                "error": error,
            }
        )

    case_by_id = {str(value["id"]): value for value in cases}
    rows: list[dict[str, Any]] = []
    for plan in plans:
        case_id = str(plan["case_id"])
        if not plan["search_invoked"]:
            rows.append({**plan, "status": "search_not_invoked", "arms": {}})
            continue
        control_groups = grouped[(case_id, "control")]
        lane_groups = grouped[(case_id, "lane_only")]
        control = fuse(control_groups)
        lane = fuse(lane_groups)
        union = fuse([*control_groups, *lane_groups])
        case = case_by_id[case_id]
        rows.append(
            {
                **plan,
                "status": "evaluated",
                "arms": {
                    "control": evaluate_arm(case, control),
                    "lane_only": evaluate_arm(case, lane),
                    "union": evaluate_arm(case, union),
                },
                "candidates": {
                    "control": control,
                    "lane_only": lane,
                    "union": union,
                },
                "requests": diagnostics[case_id],
            }
        )
    return rows


def _language_engines(values: Sequence[str]) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for raw in values:
        key, separator, engines = raw.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"invalid --language-engine: {raw}")
        output[key.strip()] = tuple(
            value.strip() for value in engines.split(",") if value.strip()
        )
    return output


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8888")
    parser.add_argument("--engine", action="append", default=[])
    parser.add_argument("--language-engine", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--lane-mode",
        choices=("site_operator", "host_token", "anchor_phrase", "page_two"),
        default="site_operator",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)

    cases = load_jsonl(args.cases, kind="case")
    if args.case_id:
        wanted = set(args.case_id)
        cases = [value for value in cases if str(value["id"]) in wanted]
        missing = wanted - {str(value["id"]) for value in cases}
        if missing:
            raise ValueError(f"unknown case IDs: {sorted(missing)}")
    snapshots = load_snapshots(args.snapshots)
    snapshot_ids = {str(value["case_id"]) for value in snapshots}
    missing_snapshots = {str(value["id"]) for value in cases} - snapshot_ids
    if missing_snapshots:
        raise ValueError(f"missing snapshots: {sorted(missing_snapshots)}")

    engines = tuple(args.engine or ("dogpile", "naver"))
    language_engines = _language_engines(
        args.language_engine or ("zh=baidu", "default=yandex")
    )
    rows = asyncio.run(
        run(
            cases,
            snapshots,
            endpoint=args.endpoint,
            engines=engines,
            language_engines=language_engines,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout_seconds,
            lane_mode=args.lane_mode,
        )
    )
    summary = summarize(rows)
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "cases_sha256": sha256(args.cases),
                "snapshots_sha256": sha256(args.snapshots),
            },
            "runtime": {
                "endpoint": args.endpoint,
                "engines": list(engines),
                "language_engines": {
                    key: list(value) for key, value in language_engines.items()
                },
                "concurrency": args.concurrency,
                "timeout_seconds": args.timeout_seconds,
                "gold_query_leak": False,
                "page_fetches": 0,
                "answer_model_calls": 0,
                "production_enabled": False,
            },
            "lane_mode": args.lane_mode,
        }
    )
    _write(
        args.output,
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in rows),
    )
    _write(
        args.summary,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
