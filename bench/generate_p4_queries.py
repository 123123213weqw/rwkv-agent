#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

if __package__:
    from .retrieval_schema import load_cases
else:
    from retrieval_schema import load_cases
from rwkv_search.g1i_native import FastRWKV7Completion
from rwkv_search.p4_search import P4SearchPlanner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--bench", default="bench/realtime_web_retrieval.jsonl")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--output", default="bench/runs/query_formation_p4_queries_v1.jsonl")
    parser.add_argument("--summary", default="bench/runs/query_formation_p4_queries_v1_summary.json")
    args = parser.parse_args()

    bench_path = Path(args.bench)
    cases = load_cases(bench_path)
    if args.case_id:
        wanted = set(args.case_id)
        missing = wanted - {str(row["id"]) for row in cases}
        if missing:
            raise ValueError(f"unknown case ids: {', '.join(sorted(missing))}")
        cases = [row for row in cases if row["id"] in wanted]
    if args.limit > 0:
        cases = cases[: args.limit]

    model = FastRWKV7Completion(args.model, args.runtime_dir)
    planner = P4SearchPlanner(model, max_tokens=args.max_tokens)
    output, summary_path = Path(args.output), Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            plan = planner.plan(str(case["query"]))
            evaluation = plan.format_evaluation
            record = {
                "schema_version": "query-formation-p4-plan.v1",
                "id": case["id"],
                "user_query": case["query"],
                "strict_success": bool(evaluation.get("strict_success")),
                "model_query": str(evaluation.get("query") or ""),
                "plan": plan.to_dict(),
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{case['id']} strict={int(record['strict_success'])} "
                f"tokens={len(plan.token_ids)} query={record['model_query']!r}",
                flush=True,
            )

    strict = sum(bool(row["strict_success"]) for row in records)
    summary = {
        "schema_version": "query-formation-p4-summary.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bench": str(bench_path),
        "bench_sha256": _sha256(bench_path),
        "model": args.model,
        "runtime_dir": args.runtime_dir,
        "total": len(records),
        "strict_success": strict,
        "strict_success_rate": round(strict / max(1, len(records)), 4),
        "average_elapsed_ms": round(
            sum(float(row["plan"].get("elapsed_ms", 0.0)) for row in records)
            / max(1, len(records)),
            3,
        ),
        "average_tokens": round(
            sum(int(row["plan"].get("token_count", 0)) for row in records)
            / max(1, len(records)),
            3,
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
