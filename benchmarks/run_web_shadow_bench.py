from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.web_retrieval_metrics import (
    aggregate,
    evaluate_candidate_stage,
    evaluate_case,
)
from rwkv_agent.tools.web import WebSearchAdapter


DATASET_SHA256 = "6900404d43deac290b599f10ee3b1f6e2fb8d8db06f821b346809049ab2e57dc"


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row.get("id") or "") for row in rows]
    if not rows or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("web cases require unique, non-empty IDs")
    for row in rows:
        if row.get("schema_version") != "realtime-retrieval-case.v1":
            raise ValueError(f"unsupported schema in {row.get('id')}")
        if row.get("language") not in {"zh", "en"}:
            raise ValueError(f"unsupported language in {row.get('id')}")
        for key in (
            "query",
            "category",
            "freshness",
            "source_policy",
            "expected_domains_any",
            "forbidden_result_types",
        ):
            if not row.get(key):
                raise ValueError(f"missing {key} in {row.get('id')}")
    return rows


def _failed_trace(profile: str, query: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": "agent-web-trace.v1",
        "status": "error",
        "profile": profile,
        "query": query,
        "initial_candidates": [],
        "post_pivot_candidates": [],
        "candidates": [],
        "results": [],
        "fetches": [],
        "warnings": [],
        "stats": {},
        "events": [],
        "latency_ms": 0.0,
        "error": f"{type(exc).__name__}: {exc}"[:500],
    }


def classify_failure_stage(trace: Mapping[str, Any]) -> str:
    stats = dict(trace.get("stats") or {})
    if trace.get("status") == "error":
        return "runtime_error"
    if int(stats.get("raw_candidates") or 0) == 0:
        return "discovery_empty"
    if int(stats.get("candidates") or 0) == 0:
        return "candidate_admission_empty"
    if int(stats.get("attempted") or 0) == 0:
        return "fetch_not_attempted"
    if int(stats.get("fetched") or 0) == 0:
        return "fetch_failed"
    if int(stats.get("usable") or 0) == 0:
        return "post_fetch_rejected"
    if int(stats.get("selected") or 0) == 0:
        return "final_ranking_empty"
    return "ok"


def _run_arm(
    case: Mapping[str, Any],
    adapter: WebSearchAdapter,
) -> dict[str, Any]:
    query = str(case["query"])
    try:
        public, trace = adapter.execute_with_trace(query)
    except Exception as exc:
        public = {
            "status": "error",
            "evidence": [],
            "retrieval": {},
        }
        trace = _failed_trace(adapter.profile, query, exc)
    candidates = list(trace.get("candidates") or ())
    results = list(trace.get("results") or ())
    stats = dict(trace.get("stats") or {})
    record = {
        "schema_version": "agent-web-bench-arm.v1",
        "id": str(case["id"]),
        "language": case.get("language"),
        "category": case.get("category"),
        "freshness": case.get("freshness"),
        "source_policy": case.get("source_policy"),
        "profile": adapter.profile,
        "status": str(public.get("status") or ""),
        "evidence": list(public.get("evidence") or ()),
        "evidence_nonempty": bool(public.get("evidence")),
        "failure_stage": classify_failure_stage(trace),
        "discovery_engines": sorted(
            {
                str(engine)
                for item in candidates
                for engine in (
                    item.get("engines")
                    or ([item.get("engine")] if item.get("engine") else [])
                )
                if engine
            }
        ),
        "trace": trace,
        "metrics": evaluate_case(case, candidates, results, stats),
        "candidate_stage_metrics": {
            "initial": evaluate_candidate_stage(
                case,
                list(trace.get("initial_candidates") or ()),
            ),
            "post_pivot": evaluate_candidate_stage(
                case,
                list(trace.get("post_pivot_candidates") or ()),
            ),
            "final": evaluate_candidate_stage(case, candidates),
        },
        "total_elapsed_ms": float(trace.get("latency_ms") or 0.0),
    }
    return record


def run_case(
    case: Mapping[str, Any],
    *,
    legacy: WebSearchAdapter,
    enhanced: WebSearchAdapter,
    enhanced_first: bool,
) -> dict[str, Any]:
    order = (
        (("enhanced", enhanced), ("legacy", legacy))
        if enhanced_first
        else (("legacy", legacy), ("enhanced", enhanced))
    )
    arms = {name: _run_arm(case, adapter) for name, adapter in order}
    enhanced_evidence = list(arms["enhanced"].get("evidence") or ())
    legacy_evidence = list(arms["legacy"].get("evidence") or ())
    fallback_used = not enhanced_evidence and bool(legacy_evidence)
    arms["enhanced"]["effective_evidence"] = (
        legacy_evidence if fallback_used else enhanced_evidence
    )
    arms["enhanced"]["fallback_used"] = fallback_used
    arms["enhanced"]["fallback_reason"] = (
        "enhanced_evidence_empty" if fallback_used else ""
    )
    return {
        "schema_version": "agent-web-shadow-row.v1",
        "case": dict(case),
        "execution_order": [name for name, _ in order],
        "legacy": arms["legacy"],
        "enhanced": arms["enhanced"],
    }


def _paired(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "candidate_domain_hit_at_5",
        "candidate_domain_hit_at_10",
        "candidate_domain_hit_at_20",
        "candidate_target_page_hit_at_10",
        "candidate_target_page_hit_at_20",
        "result_domain_hit_at_5",
        "result_domain_hit_at_10",
        "result_domain_hit_at_20",
        "result_target_page_hit_at_10",
        "result_target_page_hit_at_20",
        "nonempty_result",
    )
    output: dict[str, Any] = {}
    for metric in metrics:
        output[metric] = {
            "enhanced_wins": sum(
                bool(row["enhanced"]["metrics"].get(metric))
                and not bool(row["legacy"]["metrics"].get(metric))
                for row in rows
            ),
            "legacy_wins": sum(
                bool(row["legacy"]["metrics"].get(metric))
                and not bool(row["enhanced"]["metrics"].get(metric))
                for row in rows
            ),
        }
    output["enhanced_errors"] = sum(
        row["enhanced"].get("status") == "error" for row in rows
    )
    output["legacy_errors"] = sum(
        row["legacy"].get("status") == "error" for row in rows
    )
    return output


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    legacy = [dict(row["legacy"]) for row in rows]
    enhanced = [dict(row["enhanced"]) for row in rows]
    by_order: dict[str, Any] = {}
    failure_stages: dict[str, dict[str, int]] = {}
    evidence_nonempty: dict[str, float] = {}
    effective_evidence_nonempty: dict[str, float] = {}
    discovery_engines: dict[str, list[str]] = {}
    for arm in ("legacy", "enhanced"):
        counts: dict[str, int] = {}
        engines: set[str] = set()
        nonempty = 0
        for row in rows:
            value = row[arm]
            stage = str(value.get("failure_stage") or "unknown")
            counts[stage] = counts.get(stage, 0) + 1
            nonempty += int(bool(value.get("evidence_nonempty")))
            engines.update(str(item) for item in value.get("discovery_engines") or ())
        failure_stages[arm] = dict(sorted(counts.items()))
        evidence_nonempty[arm] = round(nonempty / max(1, len(rows)), 4)
        effective_evidence_nonempty[arm] = round(
            sum(
                bool(
                    row[arm].get("effective_evidence")
                    if arm == "enhanced"
                    else row[arm].get("evidence")
                )
                for row in rows
            )
            / max(1, len(rows)),
            4,
        )
        discovery_engines[arm] = sorted(engines)
    for order in ("legacy_first", "enhanced_first"):
        expected = (
            ["legacy", "enhanced"]
            if order == "legacy_first"
            else ["enhanced", "legacy"]
        )
        selected = [
            row for row in rows if list(row.get("execution_order") or ()) == expected
        ]
        by_order[order] = {
            "cases": len(selected),
            "legacy_nonempty": sum(
                bool(row["legacy"]["metrics"].get("nonempty_result"))
                for row in selected
            ),
            "enhanced_nonempty": sum(
                bool(row["enhanced"]["metrics"].get("nonempty_result"))
                for row in selected
            ),
            "legacy_candidate_domain_at_10": sum(
                bool(row["legacy"]["metrics"].get("candidate_domain_hit_at_10"))
                for row in selected
            ),
            "enhanced_candidate_domain_at_10": sum(
                bool(row["enhanced"]["metrics"].get("candidate_domain_hit_at_10"))
                for row in selected
            ),
        }
    return {
        "schema_version": "agent-web-shadow-summary.v1",
        "cases": len(rows),
        "legacy": aggregate(legacy),
        "enhanced": aggregate(enhanced),
        "paired": _paired(rows),
        "by_execution_order": by_order,
        "failure_stages": failure_stages,
        "evidence_nonempty_rate": evidence_nonempty,
        "effective_evidence_nonempty_rate": effective_evidence_nonempty,
        "enhanced_fallback_count": sum(
            bool(row["enhanced"].get("fallback_used")) for row in rows
        ),
        "discovery_engines": discovery_engines,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="benchmarks/data/realtime_web_retrieval_v1.jsonl",
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--raw-output",
        default="var/web-shadow-bench-v1.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        default="var/web-shadow-bench-v1-summary.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    case_path = Path(args.cases)
    cases = load_cases(case_path)
    if args.limit > 0:
        cases = cases[: args.limit]
    legacy = WebSearchAdapter(
        args.config,
        profile="legacy",
        shadow=False,
    )
    enhanced = WebSearchAdapter(
        args.config,
        profile="enhanced",
        shadow=False,
    )
    rows: list[dict[str, Any]] = []
    try:
        for position, case in enumerate(cases):
            row = run_case(
                case,
                legacy=legacy,
                enhanced=enhanced,
                enhanced_first=bool(position % 2),
            )
            rows.append(row)
            left = row["legacy"]["metrics"]
            right = row["enhanced"]["metrics"]
            print(
                f"{case['id']} legacy-domain10="
                f"{int(left['candidate_domain_hit_at_10'])} enhanced-domain10="
                f"{int(right['candidate_domain_hit_at_10'])} "
                f"legacy-result={int(left['nonempty_result'])} "
                f"enhanced-result={int(right['nonempty_result'])}",
                flush=True,
            )
    finally:
        legacy.close()
        enhanced.close()
    summary = summarize(rows)
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "path": case_path.name,
                "sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
                "expected_sha256": DATASET_SHA256,
                "cases": len(cases),
            },
            "runtime": {
                "legacy_profile": "all precision discovery flags disabled",
                "enhanced_profile": (
                    "generic English query compaction + recall-protected "
                    "candidate admission + source channels + domain pivot + "
                    "one-hop link expansion"
                ),
                "query_mode": "Agent web_search(query), latest, single",
                "empty_evidence_policy": (
                    "record enhanced raw evidence separately; reuse legacy "
                    "evidence only as an auditable effective fallback"
                ),
                "shared_discovery_cache": True,
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
