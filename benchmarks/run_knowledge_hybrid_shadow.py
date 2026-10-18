from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from rwkv_agent.tools.hybrid_knowledge import (
    CrossEncoderScorer,
    E5QueryEncoder,
    HybridKnowledgeRetriever,
)
from rwkv_agent.tools.knowledge import KnowledgeSearchAdapter


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row.get("id") or "") for row in rows]
    if not rows or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("cases require unique, non-empty IDs")
    for row in rows:
        if row.get("schema_version") != "agent-knowledge-shadow-case.v1":
            raise ValueError(f"unsupported schema in {row.get('id')}")
        if row.get("language") not in {"zh", "en"}:
            raise ValueError(f"unsupported language in {row.get('id')}")
        if not str(row.get("query") or "").strip():
            raise ValueError(f"missing query in {row.get('id')}")
        if not row.get("relevant_page_ids"):
            raise ValueError(f"missing qrels in {row.get('id')}")
    return rows


def _page_ids(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(item.get("page_id") or "")
        for item in items
        if str(item.get("page_id") or "")
    ]


def _hit(page_ids: Sequence[str], relevant: Sequence[str], depth: int) -> bool:
    return bool(set(page_ids[:depth]) & set(relevant))


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = max(
        0,
        min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1),
    )
    return ordered[position]


def _arm_summary(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    latencies = [
        float(row[name].get("latency_ms") or 0.0)
        for row in rows
        if row[name].get("status") == "ok"
    ]
    return {
        "cases": len(rows),
        "ok": sum(row[name].get("status") == "ok" for row in rows),
        "empty": sum(not row[name].get("page_ids") for row in rows),
        "hit_at_1": sum(bool(row[name].get("hit_at_1")) for row in rows),
        "hit_at_5": sum(bool(row[name].get("hit_at_5")) for row in rows),
        "hit_at_1_rate": (
            sum(bool(row[name].get("hit_at_1")) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "hit_at_5_rate": (
            sum(bool(row[name].get("hit_at_5")) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": statistics.median(latencies) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else 0.0,
        },
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_language: dict[str, Any] = {}
    for language in ("zh", "en"):
        selected = [row for row in rows if row.get("language") == language]
        by_language[language] = {
            "legacy": _arm_summary(selected, "legacy"),
            "hybrid": _arm_summary(selected, "hybrid"),
        }
    legacy = _arm_summary(rows, "legacy")
    hybrid = _arm_summary(rows, "hybrid")
    return {
        "schema_version": "agent-knowledge-shadow-summary.v1",
        "cases": len(rows),
        "legacy": legacy,
        "hybrid": hybrid,
        "paired": {
            "hybrid_hit_at_1_wins": sum(
                bool(row["hybrid"].get("hit_at_1"))
                and not bool(row["legacy"].get("hit_at_1"))
                for row in rows
            ),
            "legacy_hit_at_1_wins": sum(
                bool(row["legacy"].get("hit_at_1"))
                and not bool(row["hybrid"].get("hit_at_1"))
                for row in rows
            ),
            "hybrid_hit_at_5_wins": sum(
                bool(row["hybrid"].get("hit_at_5"))
                and not bool(row["legacy"].get("hit_at_5"))
                for row in rows
            ),
            "legacy_hit_at_5_wins": sum(
                bool(row["legacy"].get("hit_at_5"))
                and not bool(row["hybrid"].get("hit_at_5"))
                for row in rows
            ),
            "page_order_changed": sum(
                row["legacy"].get("page_ids") != row["hybrid"].get("page_ids")
                for row in rows
            ),
            "hydrated_text_changed": sum(
                bool(row["hybrid"].get("hydrated_text_changed"))
                for row in rows
            ),
            "hybrid_fallbacks": sum(
                row["hybrid"].get("status") != "ok" for row in rows
            ),
        },
        "by_language": by_language,
    }


def run_case(
    case: Mapping[str, Any],
    *,
    legacy: KnowledgeSearchAdapter,
    hybrid: HybridKnowledgeRetriever,
) -> dict[str, Any]:
    query = str(case["query"])
    language = str(case["language"])
    relevant = [str(value) for value in case["relevant_page_ids"]]

    legacy_started = time.perf_counter()
    legacy_result = legacy.execute(query, language=language)
    legacy_wall_ms = (time.perf_counter() - legacy_started) * 1000.0
    legacy_evidence = list(legacy_result.get("evidence") or ())
    legacy_ids = _page_ids(legacy_evidence)

    hybrid_started = time.perf_counter()
    try:
        hybrid_result = hybrid.search(query, language=language)
        hybrid_wall_ms = (time.perf_counter() - hybrid_started) * 1000.0
        hybrid_hits = list(hybrid_result.hits)
        hybrid_ids = _page_ids(hybrid_hits)
        hybrid_arm = {
            "status": "ok",
            "page_ids": hybrid_ids,
            "hit_at_1": _hit(hybrid_ids, relevant, 1),
            "hit_at_5": _hit(hybrid_ids, relevant, 5),
            "latency_ms": hybrid_wall_ms,
            "stats": hybrid_result.stats,
            "hits": hybrid_hits,
            "hydrated_text_changed": bool(
                hybrid_result.stats.get("hydration", {}).get("changed_pages")
            ),
        }
    except Exception as exc:
        hybrid_wall_ms = (time.perf_counter() - hybrid_started) * 1000.0
        hybrid_arm = {
            "status": "fallback_legacy",
            "page_ids": legacy_ids,
            "hit_at_1": _hit(legacy_ids, relevant, 1),
            "hit_at_5": _hit(legacy_ids, relevant, 5),
            "latency_ms": hybrid_wall_ms,
            "stats": {},
            "hits": [],
            "hydrated_text_changed": False,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    return {
        "schema_version": "agent-knowledge-shadow-row.v1",
        "case_id": str(case["id"]),
        "source_case_id": str(case["source_case_id"]),
        "language": language,
        "query_type": str(case.get("query_type") or ""),
        "query": query,
        "relevant_page_ids": relevant,
        "legacy": {
            "status": str(legacy_result.get("status") or ""),
            "page_ids": legacy_ids,
            "hit_at_1": _hit(legacy_ids, relevant, 1),
            "hit_at_5": _hit(legacy_ids, relevant, 5),
            "latency_ms": legacy_wall_ms,
            "evidence": legacy_evidence,
        },
        "hybrid": hybrid_arm,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="benchmarks/data/knowledge_shadow_cases_v1.jsonl",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:19220")
    parser.add_argument(
        "--embedding-model",
        default="models/multilingual-e5-small",
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--raw-output",
        default="var/knowledge-hybrid-shadow-v1.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        default="var/knowledge-hybrid-shadow-v1-summary.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    case_path = Path(args.cases)
    cases = load_cases(case_path)
    if args.limit > 0:
        cases = cases[: args.limit]
    encoder = E5QueryEncoder(args.embedding_model, device=args.device)
    scorer = CrossEncoderScorer(args.reranker_model, device=args.device)
    hybrid = HybridKnowledgeRetriever(
        args.endpoint,
        encoder=encoder,
        scorer=scorer,
    )
    legacy = KnowledgeSearchAdapter(args.endpoint, shadow=False)
    rows = [
        run_case(case, legacy=legacy, hybrid=hybrid)
        for case in cases
    ]
    summary = summarize(rows)
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "path": case_path.name,
                "sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
                "cases": len(cases),
            },
            "runtime": {
                "endpoint": args.endpoint,
                "embedding_model": args.embedding_model,
                "reranker_model": args.reranker_model,
                "device": args.device,
                "answer_model_called": False,
                "visible_strategy": "legacy",
                "production_enabled": False,
            },
        }
    )
    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
