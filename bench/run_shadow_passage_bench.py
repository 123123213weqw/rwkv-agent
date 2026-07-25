from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from rwkv_search.candidate_index import CandidateIndexClient
from rwkv_search.config import ShadowSearchConfig
from rwkv_search.passage_hydration import (
    PagePassageClient,
    PassageHydrator,
    TransformersPassageScorer,
)
from rwkv_search.shadow_search import FineWikiShadowSearch

from bench.long_knowledge_schema import load_cases


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = min(
        len(ordered) - 1,
        max(0, math.ceil(len(ordered) * fraction) - 1),
    )
    return ordered[position]


def _evidence_page_ids(evidence: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str((item.get("metadata") or {}).get("page_id") or "")
        for item in evidence
        if str((item.get("metadata") or {}).get("page_id") or "")
    ]


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty shadow passage results")
    compared = sum(int(row.get("compared_count") or 0) for row in rows)
    changed = sum(int(row.get("changed_text_count") or 0) for row in rows)
    latencies = [
        float((row.get("passage_hydration") or {}).get("latency_ms") or 0.0)
        for row in rows
    ]
    query_latencies = [
        float((row.get("passage_hydration") or {}).get("query_latency_ms") or 0.0)
        for row in rows
    ]
    rerank_latencies = [
        float((row.get("passage_hydration") or {}).get("rerank_latency_ms") or 0.0)
        for row in rows
    ]
    warm_latencies = latencies[1:]
    warm_query_latencies = query_latencies[1:]
    warm_rerank_latencies = rerank_latencies[1:]
    deltas = [int(row.get("character_delta") or 0) for row in rows]
    statuses: dict[str, int] = {}
    for row in rows:
        status = str((row.get("passage_hydration") or {}).get("status") or "")
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "schema_version": "shadow-passage-bench-summary.v1",
        "status": "isolated_shadow_not_production",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": len(rows),
        "hydration_statuses": dict(sorted(statuses.items())),
        "page_order_identity_rate": statistics.fmean(
            float(row.get("page_order_identical", False)) for row in rows
        ),
        "legacy_hit_at_8": statistics.fmean(
            float(row.get("legacy_hit_at_8", False)) for row in rows
        ),
        "hydrated_hit_at_8": statistics.fmean(
            float(row.get("hydrated_hit_at_8", False)) for row in rows
        ),
        "compared_evidence": compared,
        "changed_evidence": changed,
        "changed_evidence_rate": changed / max(1, compared),
        "character_delta": {
            "mean": statistics.fmean(deltas),
            "min": min(deltas),
            "max": max(deltas),
        },
        "latency_ms": {
            "cold_start": latencies[0],
            "mean": statistics.fmean(latencies),
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies),
            "query_mean": statistics.fmean(query_latencies),
            "rerank_mean": statistics.fmean(rerank_latencies),
            "warm_mean": (
                statistics.fmean(warm_latencies)
                if warm_latencies
                else latencies[0]
            ),
            "warm_p50": _percentile(warm_latencies or latencies, 0.5),
            "warm_p95": _percentile(warm_latencies or latencies, 0.95),
            "warm_query_mean": (
                statistics.fmean(warm_query_latencies)
                if warm_query_latencies
                else query_latencies[0]
            ),
            "warm_rerank_mean": (
                statistics.fmean(warm_rerank_latencies)
                if warm_rerank_latencies
                else rerank_latencies[0]
            ),
        },
        "empty_legacy_evidence_cases": sum(
            int(not row.get("legacy_page_ids")) for row in rows
        ),
        "visible_output_changed": False,
        "answer_generation_executed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pair legacy and lead+CrossEncoder Evidence inside the existing "
            "FineWiki Shadow path without changing visible answers."
        )
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--language", required=True, choices=("zh", "en"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:19220")
    parser.add_argument("--index", required=True)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--channel-size", type=int, default=50)
    parser.add_argument("--chunks-per-page", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=3200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = [
        case
        for case in load_cases(args.cases)
        if case.language == args.language and case.expectation == "relevant"
    ]
    if not cases:
        raise SystemExit("no positive cases remain after filtering")
    scorer = TransformersPassageScorer(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        fp16=args.fp16,
        local_files_only=not args.allow_model_download,
    )
    hydrator = PassageHydrator(
        PagePassageClient(args.endpoint, timeout=45.0),
        scorer,
        max_pages=max(1, args.limit),
        chunks_per_page=max(1, args.chunks_per_page),
        max_chars=max(512, args.max_chars),
    )
    config = ShadowSearchConfig(
        enabled=True,
        endpoint=args.endpoint,
        index=args.index,
        timeout_seconds=45.0,
        limit=max(1, args.limit),
        channel_size=max(10, args.channel_size),
        passage_hydration_enabled=True,
        passage_max_pages=max(1, args.limit),
        passage_chunks_per_page=max(1, args.chunks_per_page),
        passage_max_chars=max(512, args.max_chars),
        passage_model=args.model,
        passage_device=args.device,
        passage_batch_size=max(1, args.batch_size),
        passage_max_length=max(64, args.max_length),
        passage_fp16=bool(args.fp16),
        passage_local_files_only=not args.allow_model_download,
    )
    runner = FineWikiShadowSearch(
        config,
        client=CandidateIndexClient(args.endpoint, timeout=45.0),
        passage_hydrator=hydrator,
    )
    rows: list[dict[str, Any]] = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("w", encoding="utf-8") as handle:
            for position, case in enumerate(cases, start=1):
                future = runner.start(
                    case.query,
                    {
                        "intent": "knowledge",
                        "shadow_benchmark": True,
                        "language": case.language,
                    },
                )
                if future is None:
                    raise RuntimeError("shadow request was unexpectedly skipped")
                payload = future.result(timeout=120.0)
                legacy = list(payload.get("legacy_evidence") or ())
                hydrated = list(payload.get("evidence") or ())
                comparison = FineWikiShadowSearch._compare_evidence_variants(
                    legacy,
                    hydrated,
                )
                legacy_pages = _evidence_page_ids(legacy)
                hydrated_pages = _evidence_page_ids(hydrated)
                positives = {page.page_id for page in case.relevant_pages}
                row = {
                    "schema_version": "shadow-passage-bench-row.v1",
                    "case_id": case.id,
                    "query": case.query,
                    "language": case.language,
                    "query_type": case.query_type,
                    "relevant_page_ids": sorted(positives),
                    "legacy_page_ids": legacy_pages,
                    "hydrated_page_ids": hydrated_pages,
                    "page_order_identical": legacy_pages == hydrated_pages,
                    "legacy_hit_at_8": bool(
                        positives.intersection(legacy_pages[:8])
                    ),
                    "hydrated_hit_at_8": bool(
                        positives.intersection(hydrated_pages[:8])
                    ),
                    "legacy_evidence": legacy,
                    "hydrated_evidence": hydrated,
                    "evidence_variant": payload.get("evidence_variant"),
                    "passage_hydration": payload.get("passage_hydration"),
                    "shadow_latency_ms": payload.get("latency_ms"),
                    **comparison,
                }
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                rows.append(row)
                print(
                    f"[{position}/{len(cases)}] {case.id} "
                    f"status={row['passage_hydration'].get('status')} "
                    f"changed={row['changed_text_count']}/"
                    f"{row['compared_count']}",
                    flush=True,
                )
    finally:
        runner.close()
    summary = summarize(rows)
    summary.update(
        {
            "language": args.language,
            "index": args.index,
            "model": args.model,
            "chunks_per_page": max(1, args.chunks_per_page),
            "max_pages": max(1, args.limit),
            "max_chars": max(512, args.max_chars),
        }
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
