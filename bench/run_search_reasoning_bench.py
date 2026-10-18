#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import socket
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import aiohttp

if __package__:
    from .query_formation import evaluate_discovery
    from .retrieval_schema import load_cases
    from .search_reasoning_bench import STRATEGIES, summarize_records
else:
    from query_formation import evaluate_discovery
    from retrieval_schema import load_cases
    from search_reasoning_bench import STRATEGIES, summarize_records
from rwkv_search.config import AppConfig
from rwkv_search.g1i_native import FastRWKV7Completion
from rwkv_search.realtime.discovery import (
    bing_search_headers,
    bing_search_params,
    parse_search_html,
)
from rwkv_search.realtime.types import DiscoveredURL
from rwkv_search.search_reasoning import (
    feedback_gate,
    generate_search_action,
    merge_query_candidates,
    render_observation,
    serialize_candidates,
    validate_generated_query,
)
from rwkv_search.search_request import SearchRequestBuilder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_body(raw: bytes, encoding: str) -> bytes:
    normalized = encoding.casefold().strip()
    try:
        if normalized in {"gzip", "x-gzip"}:
            return zlib.decompress(raw, 16 + zlib.MAX_WBITS, 4 * 1024 * 1024)
        if normalized == "deflate":
            return zlib.decompress(raw, zlib.MAX_WBITS, 4 * 1024 * 1024)
    except zlib.error:
        return b""
    return raw if normalized in {"", "identity"} else b""


class BingCNDiscovery:
    """Experiment-only direct Bing Web HTML client with per-run query cache."""

    def __init__(self, session: aiohttp.ClientSession, timeout_seconds: float) -> None:
        self.session = session
        self.timeout_seconds = timeout_seconds
        self.cache: Dict[str, Tuple[List[DiscoveredURL], float, List[Dict[str, str]]]] = {}

    async def discover(
        self, query: str, *, max_candidates: int = 30
    ) -> Tuple[List[DiscoveredURL], float, bool, List[Dict[str, str]]]:
        key = " ".join(query.casefold().split())
        if key in self.cache:
            candidates, elapsed_ms, diagnostics = self.cache[key]
            return copy.deepcopy(candidates), elapsed_ms, True, copy.deepcopy(diagnostics)
        started = time.perf_counter()
        diagnostics: List[Dict[str, str]] = []
        candidates: List[DiscoveredURL] = []
        try:
            response = await asyncio.wait_for(
                self.session.get(
                    "https://cn.bing.com/search",
                    params=bing_search_params(query),
                    headers=bing_search_headers(query),
                    allow_redirects=True,
                ),
                timeout=self.timeout_seconds,
            )
            async with response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                raw = await response.read()
                if len(raw) > 2 * 1024 * 1024:
                    raise RuntimeError("response_too_large")
                raw = _decode_body(raw, response.headers.get("Content-Encoding", ""))
                charset = response.charset or "utf-8"
            candidates = parse_search_html(raw.decode(charset, "replace"), "bing")
            for position, item in enumerate(candidates, 1):
                item.rank = position
                item.rrf_score = 1.0 / (60.0 + position)
                item.engines = ["bing"]
                item.positions = [position]
                item.discovery_stage = "initial"
                item.discovery_stages = ["initial"]
        except Exception as exc:
            diagnostics.append(
                {
                    "query": query,
                    "engine": "bing_cn",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:300],
                }
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        candidates = candidates[: max(0, max_candidates)]
        self.cache[key] = (copy.deepcopy(candidates), elapsed_ms, copy.deepcopy(diagnostics))
        return candidates, elapsed_ms, False, diagnostics


def _action_record(action: Any, validation: Any, execution_query: str) -> Dict[str, Any]:
    value = action.to_dict()
    value["validation"] = validation.to_dict() if validation else None
    value["execution_query"] = execution_query
    return value


def _successful_action(action: Any, validation: Any) -> bool:
    return action.kind == "search" and bool(validation and validation.accepted)


async def _search_action(
    discovery: BingCNDiscovery,
    query: str,
    *,
    delay_seconds: float,
) -> Tuple[List[DiscoveredURL], Dict[str, Any]]:
    candidates, elapsed_ms, cache_hit, diagnostics = await discovery.discover(query)
    for position, item in enumerate(candidates, 1):
        item.matched_queries = list(dict.fromkeys([*item.matched_queries, query]))
        item.query_positions[query] = min(
            item.query_positions.get(query, position), position
        )
    if not cache_hit and delay_seconds > 0:
        await asyncio.sleep(delay_seconds)
    return candidates, {
        "query": query,
        "elapsed_ms": elapsed_ms,
        "cache_hit": cache_hit,
        "diagnostics": diagnostics,
        "candidate_count": len(candidates),
    }


async def run_case(
    case: Mapping[str, Any],
    model: FastRWKV7Completion,
    discovery: BingCNDiscovery,
    *,
    delay_seconds: float,
    max_tokens: int,
    react_rounds: int,
    strategies: Sequence[str] = STRATEGIES,
) -> List[Dict[str, Any]]:
    user_query = str(case["query"])
    freshness = str(case.get("freshness") or "stable")
    builder = SearchRequestBuilder()
    records: List[Dict[str, Any]] = []
    selected = tuple(dict.fromkeys(str(value) for value in strategies))
    if not selected or "direct" not in selected:
        raise ValueError("strategy selection requires direct")
    unknown = set(selected) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies: {', '.join(sorted(unknown))}")

    async def first_pass(strategy: str) -> Tuple[Dict[str, Any], List[DiscoveredURL]]:
        started = time.perf_counter()
        action = generate_search_action(
            model, strategy, user_query, max_tokens=max_tokens
        )
        execution_query = ""
        validation = validate_generated_query(user_query, action.query)
        searches: List[Dict[str, Any]] = []
        candidates: List[DiscoveredURL] = []
        if _successful_action(action, validation):
            request = builder.build(user_query, action.query)
            execution_query = request.execution_queries[0]
            candidates, search = await _search_action(
                discovery, execution_query, delay_seconds=delay_seconds
            )
            searches.append(search)
        metrics = evaluate_discovery(case, serialize_candidates(candidates))
        model_elapsed_ms = round(action.elapsed_ms, 3)
        search_elapsed_ms = round(sum(item["elapsed_ms"] for item in searches), 3)
        wall_elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        record = {
            "schema_version": "search-reasoning-run.v1",
            "id": case["id"],
            "query": user_query,
            "language": case.get("language"),
            "category": case.get("category"),
            "freshness": freshness,
            "strategy": strategy,
            "actions": [_action_record(action, validation, execution_query)],
            "searches": searches,
            "query_count": len(searches),
            "model_call_count": 1,
            "token_count": len(action.token_ids),
            "model_elapsed_ms": model_elapsed_ms,
            "search_elapsed_ms": search_elapsed_ms,
            "total_elapsed_ms": round(model_elapsed_ms + search_elapsed_ms, 3),
            "wall_elapsed_ms": wall_elapsed_ms,
            "case_success": bool(searches),
            "stop_reason": "single_pass",
            "candidates": serialize_candidates(candidates),
            "metrics": metrics,
        }
        return record, candidates

    direct_record, direct_candidates = await first_pass("direct")
    if "direct" in selected:
        records.append(direct_record)
    if "short_cot" in selected:
        short_cot_record, _ = await first_pass("short_cot")
        records.append(short_cot_record)

    if "feedback" in selected:
        feedback_started = time.perf_counter()
        feedback_record = copy.deepcopy(direct_record)
        feedback_record["strategy"] = "feedback"
        feedback_record["schema_version"] = "search-reasoning-run.v1"
        feedback_record["total_elapsed_ms"] = direct_record["total_elapsed_ms"]
        gate = feedback_gate(
            user_query,
            str(direct_record["actions"][0].get("execution_query") or ""),
            direct_candidates,
        )
        feedback_record["feedback_gate"] = gate.to_dict()
        feedback_groups: List[Tuple[str, Sequence[DiscoveredURL]]] = []
        direct_execution = str(
            direct_record["actions"][0].get("execution_query") or ""
        )
        if direct_execution and direct_candidates:
            feedback_groups.append((direct_execution, direct_candidates))
        if gate.trigger and direct_execution:
            action = generate_search_action(
                model,
                "feedback",
                user_query,
                previous_query=direct_execution,
                candidates=direct_candidates,
                max_tokens=max_tokens,
            )
            observation = render_observation(direct_candidates)
            validation = validate_generated_query(
                user_query,
                action.query,
                previous_queries=(direct_execution,),
                observation=observation,
                allow_observation_grounding=True,
            )
            execution_query = ""
            if _successful_action(action, validation):
                request = builder.build(user_query, action.query)
                execution_query = request.execution_queries[0]
                execution_validation = validate_generated_query(
                    user_query,
                    execution_query,
                    previous_queries=(direct_execution,),
                    observation=observation,
                    allow_observation_grounding=True,
                )
                if execution_validation.accepted:
                    extra_candidates, search = await _search_action(
                        discovery, execution_query, delay_seconds=delay_seconds
                    )
                    feedback_record["searches"].append(search)
                    feedback_groups.append((execution_query, extra_candidates))
                else:
                    validation = execution_validation
            feedback_record["actions"].append(
                _action_record(action, validation, execution_query)
            )
            feedback_record["model_call_count"] += 1
            feedback_record["token_count"] += len(action.token_ids)
            feedback_record["model_elapsed_ms"] = round(
                feedback_record["model_elapsed_ms"] + action.elapsed_ms, 3
            )
        feedback_candidates = merge_query_candidates(feedback_groups)
        feedback_record["candidates"] = serialize_candidates(feedback_candidates)
        feedback_record["metrics"] = evaluate_discovery(
            case, feedback_record["candidates"]
        )
        feedback_record["query_count"] = len(feedback_record["searches"])
        feedback_record["search_elapsed_ms"] = round(
            sum(item["elapsed_ms"] for item in feedback_record["searches"]), 3
        )
        feedback_record["total_elapsed_ms"] = round(
            feedback_record["model_elapsed_ms"]
            + feedback_record["search_elapsed_ms"],
            3,
        )
        feedback_record["wall_elapsed_ms"] = round(
            direct_record.get("wall_elapsed_ms", 0.0)
            + (time.perf_counter() - feedback_started) * 1000.0,
            3,
        )
        feedback_record["case_success"] = bool(feedback_groups)
        if not gate.trigger:
            feedback_record["stop_reason"] = "gate_not_triggered"
        elif len(feedback_record["searches"]) > len(direct_record["searches"]):
            feedback_record["stop_reason"] = "feedback_complete"
        else:
            feedback_record["stop_reason"] = "feedback_invalid"
        records.append(feedback_record)

    if "react" not in selected:
        return records

    react_started = time.perf_counter()
    trajectory: List[Dict[str, Any]] = []
    react_actions: List[Dict[str, Any]] = []
    react_searches: List[Dict[str, Any]] = []
    react_groups: List[Tuple[str, Sequence[DiscoveredURL]]] = []
    stop_reason = "max_rounds"
    token_count, model_elapsed = 0, 0.0
    for _ in range(max(1, react_rounds)):
        action = generate_search_action(
            model,
            "react",
            user_query,
            trajectory=trajectory,
            max_tokens=max_tokens,
        )
        token_count += len(action.token_ids)
        model_elapsed += action.elapsed_ms
        if action.kind == "final":
            react_actions.append(_action_record(action, None, ""))
            stop_reason = "model_final"
            break
        previous = tuple(query for query, _ in react_groups)
        previous_observation = trajectory[-1]["observation"] if trajectory else ""
        validation = validate_generated_query(
            user_query,
            action.query,
            previous_queries=previous,
            observation=previous_observation,
            allow_observation_grounding=bool(trajectory),
        )
        execution_query = ""
        if not _successful_action(action, validation):
            react_actions.append(_action_record(action, validation, execution_query))
            stop_reason = "invalid_action"
            break
        request = builder.build(user_query, action.query)
        execution_query = request.execution_queries[0]
        execution_validation = validate_generated_query(
            user_query,
            execution_query,
            previous_queries=previous,
            observation=previous_observation,
            allow_observation_grounding=bool(trajectory),
        )
        if not execution_validation.accepted:
            react_actions.append(
                _action_record(action, execution_validation, execution_query)
            )
            stop_reason = "invalid_execution_query"
            break
        candidates, search = await _search_action(
            discovery, execution_query, delay_seconds=delay_seconds
        )
        react_actions.append(_action_record(action, validation, execution_query))
        react_searches.append(search)
        react_groups.append((execution_query, candidates))
        trajectory.append(
            {
                "query": execution_query,
                "observation": render_observation(candidates),
            }
        )
    react_candidates = merge_query_candidates(react_groups)
    react_serialized = serialize_candidates(react_candidates)
    react_search_elapsed = round(
        sum(item["elapsed_ms"] for item in react_searches), 3
    )
    react_model_elapsed = round(model_elapsed, 3)
    react_record = {
        "schema_version": "search-reasoning-run.v1",
        "id": case["id"],
        "query": user_query,
        "language": case.get("language"),
        "category": case.get("category"),
        "freshness": freshness,
        "strategy": "react",
        "actions": react_actions,
        "searches": react_searches,
        "trajectory": trajectory,
        "query_count": len(react_searches),
        "model_call_count": len(react_actions),
        "token_count": token_count,
        "model_elapsed_ms": react_model_elapsed,
        "search_elapsed_ms": react_search_elapsed,
        "total_elapsed_ms": round(react_model_elapsed + react_search_elapsed, 3),
        "wall_elapsed_ms": round((time.perf_counter() - react_started) * 1000.0, 3),
        "case_success": bool(react_searches),
        "stop_reason": stop_reason,
        "candidates": react_serialized,
        "metrics": evaluate_discovery(case, react_serialized),
    }
    records.append(react_record)
    return records


async def run(
    cases: Sequence[Mapping[str, Any]],
    model: FastRWKV7Completion,
    config: AppConfig,
    *,
    delay_seconds: float,
    max_tokens: int,
    react_rounds: int,
    strategies: Sequence[str] = STRATEGIES,
) -> List[Dict[str, Any]]:
    realtime = config.realtime_search
    timeout = aiohttp.ClientTimeout(
        total=None,
        connect=realtime.connect_timeout_seconds,
        sock_connect=realtime.connect_timeout_seconds,
        sock_read=realtime.discovery_timeout_seconds,
    )
    connector = aiohttp.TCPConnector(
        limit=2,
        family=socket.AF_INET if realtime.force_ipv4 else socket.AF_UNSPEC,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    headers = {
        "User-Agent": realtime.user_agent,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.2",
        "Accept-Encoding": "gzip, deflate",
    }
    records: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        auto_decompress=False,
        headers=headers,
        trust_env=False,
    ) as session:
        discovery = BingCNDiscovery(session, realtime.discovery_timeout_seconds)
        for index, case in enumerate(cases, 1):
            case_records = await run_case(
                case,
                model,
                discovery,
                delay_seconds=delay_seconds,
                max_tokens=max_tokens,
                react_rounds=react_rounds,
                strategies=strategies,
            )
            records.extend(case_records)
            line = " ".join(
                f"{row['strategy']}:{int(row['metrics']['domain_hit_at_10'])}/"
                f"{row['query_count']}q"
                for row in case_records
            )
            print(f"[{index}/{len(cases)}] {case['id']} {line}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--config", default="configs/benchmark.json")
    parser.add_argument("--bench", default="bench/realtime_web_retrieval.jsonl")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--react-rounds", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument(
        "--strategies",
        default=",".join(STRATEGIES),
        help="Comma-separated subset; direct is required (for example direct,feedback)",
    )
    parser.add_argument("--output", default="bench/runs/search_reasoning_abcd_v1.jsonl")
    parser.add_argument(
        "--summary", default="bench/runs/search_reasoning_abcd_v1_summary.json"
    )
    args = parser.parse_args()

    bench_path = Path(args.bench)
    cases = load_cases(bench_path)
    if args.case_id:
        wanted = set(args.case_id)
        missing = wanted - {str(row["id"]) for row in cases}
        if missing:
            raise ValueError(f"unknown case ids: {', '.join(sorted(missing))}")
        cases = [row for row in cases if str(row["id"]) in wanted]
    if args.limit > 0:
        cases = cases[: args.limit]
    config = AppConfig.load(args.config)
    model = FastRWKV7Completion(args.model, args.runtime_dir)
    strategies = tuple(
        dict.fromkeys(
            value.strip() for value in args.strategies.split(",") if value.strip()
        )
    )
    if not strategies or "direct" not in strategies:
        raise ValueError("--strategies requires direct")
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies: {', '.join(sorted(unknown))}")
    records = asyncio.run(
        run(
            cases,
            model,
            config,
            delay_seconds=max(0.0, args.delay_seconds),
            max_tokens=max(32, args.max_tokens),
            react_rounds=max(1, min(3, args.react_rounds)),
            strategies=strategies,
        )
    )
    output, summary_path = Path(args.output), Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize_records(records, strategies=strategies)
    summary.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bench": str(bench_path),
            "bench_sha256": _sha256(bench_path),
            "model": args.model,
            "runtime_dir": args.runtime_dir,
            "config": args.config,
            "case_count": len(cases),
            "react_rounds": max(1, min(3, args.react_rounds)),
            "search_engine": "https://cn.bing.com/search",
            "decoder": "greedy",
            "strategies": list(strategies),
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["deltas"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
