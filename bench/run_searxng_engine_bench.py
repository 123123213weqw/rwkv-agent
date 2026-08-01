#!/usr/bin/env python3
"""Benchmark each enabled SearXNG engine without changing its configuration."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import aiohttp

if __package__:
    from .retrieval_schema import load_cases as load_validated_cases
    from .run_realtime_retrieval_bench import apply_model_queries, load_cases
    from .searxng_engine_bench import (
        SCHEMA_VERSION,
        enabled_search_engines,
        engine_is_unresponsive,
        evaluate_engine_record,
        public_case_matrix,
        rank_general_engines,
        summarize_records,
    )
else:
    from retrieval_schema import load_cases as load_validated_cases  # type: ignore
    from run_realtime_retrieval_bench import apply_model_queries, load_cases  # type: ignore
    from searxng_engine_bench import (  # type: ignore
        SCHEMA_VERSION,
        enabled_search_engines,
        engine_is_unresponsive,
        evaluate_engine_record,
        public_case_matrix,
        rank_general_engines,
        summarize_records,
    )

from rwkv_search.realtime.discovery import parse_searxng_results, searxng_search_params


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_candidate(value: Any, position: int) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        item = value.to_dict()
    elif isinstance(value, Mapping):
        item = dict(value)
    elif hasattr(value, "__dataclass_fields__"):
        item = {
            name: getattr(value, name)
            for name in value.__dataclass_fields__
            if hasattr(value, name)
        }
    else:
        item = {"url": str(value)}
    item.pop("content", None)
    item["position"] = position
    return item


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    params: Mapping[str, str] | None = None,
) -> tuple[int, Mapping[str, Any]]:
    async with session.get(url, params=params) as response:
        status = response.status
        data = await response.json(content_type=None)
        if not isinstance(data, Mapping):
            raise ValueError("SearXNG response is not an object")
        return status, data


async def run_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    case: Mapping[str, Any],
    engine: str,
    repetition: int,
) -> Dict[str, Any]:
    query = str(case.get("model_query") or case["query"])
    params = searxng_search_params(query, str(case.get("freshness") or "none"))
    params.pop("categories", None)
    params["engines"] = engine
    queued_at = time.perf_counter()
    candidates: List[Dict[str, Any]] = []
    status = 0
    unresponsive: Sequence[Any] = ()
    error_type = ""
    error_message = ""
    async with semaphore:
        queue_elapsed_ms = round((time.perf_counter() - queued_at) * 1000.0, 3)
        started = time.perf_counter()
        try:
            status, data = await fetch_json(session, endpoint + "/search", params=params)
            unresponsive = data.get("unresponsive_engines", ())
            candidates = [
                serialize_candidate(value, position)
                for position, value in enumerate(parse_searxng_results(data)[:20], 1)
            ]
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)[:300]
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    request_success = (
        status == 200
        and not error_type
        and not engine_is_unresponsive(engine, unresponsive)
    )
    metrics = evaluate_engine_record(
        case, candidates, request_success=request_success
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "id": case["id"],
        "language": case["language"],
        "category": case["category"],
        "engine": engine,
        "repetition": repetition,
        "request": {
            "query": query,
            "language": params["language"],
            "safesearch": params["safesearch"],
            "engine": engine,
        },
        "http_status": status,
        "unresponsive_engines": list(unresponsive),
        "error_type": error_type,
        "error_message": error_message,
        "queue_elapsed_ms": queue_elapsed_ms,
        "elapsed_ms": elapsed_ms,
        "candidates": candidates,
        "metrics": metrics,
    }


async def run(args: argparse.Namespace) -> None:
    bench_path = Path(args.bench)
    cases = load_cases(bench_path, args.case_id, args.limit)
    model_queries_path = Path(args.model_queries)
    cases = apply_model_queries(cases, model_queries_path)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(1, args.concurrency))
    headers = {"User-Agent": "rwkv-search-engine-bench/1.0"}
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:
        status, config = await fetch_json(session, args.endpoint.rstrip("/") + "/config")
        if status != 200:
            raise RuntimeError(f"SearXNG /config returned HTTP {status}")
        specs = enabled_search_engines(config)
        enabled_names = {value["name"] for value in specs}
        engines = list(dict.fromkeys(args.engine or sorted(enabled_names)))
        unknown = sorted(set(engines) - enabled_names)
        if unknown:
            raise ValueError("engines are not enabled by SearXNG: " + ", ".join(unknown))

        semaphore = asyncio.Semaphore(max(1, args.concurrency))
        records: List[Dict[str, Any]] = []
        for repetition in range(1, args.repetitions + 1):
            for engine in engines:
                if args.request_delay > 0:
                    engine_rows = []
                    for index, case in enumerate(cases):
                        engine_rows.append(
                            await run_one(
                                session,
                                semaphore,
                                args.endpoint.rstrip("/"),
                                case,
                                engine,
                                repetition,
                            )
                        )
                        if index + 1 != len(cases):
                            await asyncio.sleep(args.request_delay)
                else:
                    tasks = [
                        run_one(
                            session,
                            semaphore,
                            args.endpoint.rstrip("/"),
                            case,
                            engine,
                            repetition,
                        )
                        for case in cases
                    ]
                    engine_rows = await asyncio.gather(*tasks)
                records.extend(engine_rows)
                hit_count = sum(
                    bool(row["metrics"]["candidate_domain_hit_at_10"])
                    for row in engine_rows
                )
                success_count = sum(
                    bool(row["metrics"]["request_success"]) for row in engine_rows
                )
                print(
                    f"repeat={repetition} engine={engine} "
                    f"domain@10={hit_count}/{len(engine_rows)} "
                    f"success={success_count}/{len(engine_rows)}",
                    flush=True,
                )
            if repetition != args.repetitions and args.repeat_delay > 0:
                await asyncio.sleep(args.repeat_delay)

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    public_path = Path(args.public_summary) if args.public_summary else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_records(records, specs)
    summary.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "endpoint": args.endpoint,
            "bench": str(bench_path),
            "bench_sha256": sha256(bench_path),
            "model_queries": str(model_queries_path),
            "model_queries_sha256": sha256(model_queries_path),
            "case_count": len(cases),
            "repetition_count": args.repetitions,
            "selected_engines": engines,
            "general_engine_ranking": {
                language: rank_general_engines(summary, language)
                for language in ("zh", "en")
            },
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if public_path:
        public_path.parent.mkdir(parents=True, exist_ok=True)
        public = dict(summary)
        public.pop("endpoint", None)
        public["case_matrix"] = public_case_matrix(records)
        public_path.write_text(
            json.dumps(public, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary["general_engine_ranking"], ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8888")
    parser.add_argument("--bench", default="bench/realtime_web_retrieval.jsonl")
    parser.add_argument("--model-queries", required=True)
    parser.add_argument("--engine", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--repeat-delay", type=float, default=5.0)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="pace an engine sequentially; recommended for stability runs",
    )
    parser.add_argument("--output", default="bench/runs/searxng_engines_v1.jsonl")
    parser.add_argument("--summary", default="bench/runs/searxng_engines_v1_summary.json")
    parser.add_argument("--public-summary", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    # Validate early so malformed public cases never reach the live endpoint.
    load_validated_cases(Path(args.bench))
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
