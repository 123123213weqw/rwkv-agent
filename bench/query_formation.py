from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from rwkv_search.analysis import QueryAnalyzer

try:
    from .retrieval_metrics import evaluate_case
except ImportError:  # Direct execution through another bench script.
    from retrieval_metrics import evaluate_case


STRATEGIES = ("raw", "rules", "p4")


class QueryFormationError(ValueError):
    pass


def _clean_query(value: str) -> str:
    return " ".join(str(value or "").strip(" ？?。.!！,，;；").split())


def _unique_queries(values: Iterable[str]) -> Tuple[str, ...]:
    output: List[str] = []
    seen = set()
    for value in values:
        clean = _clean_query(value)
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            output.append(clean)
    return tuple(output)


def load_p4_plans(path: Path) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QueryFormationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise QueryFormationError(f"{path}:{line_number}: row must be an object")
            identifier = str(value.get("id") or "")
            if not identifier:
                raise QueryFormationError(f"{path}:{line_number}: id is required")
            if identifier in output:
                raise QueryFormationError(f"{path}:{line_number}: duplicate id {identifier!r}")
            output[identifier] = value
    return output


def strategy_queries(
    case: Mapping[str, Any],
    p4_plan: Mapping[str, Any] | None,
    *,
    analyzer: QueryAnalyzer | None = None,
) -> Dict[str, Tuple[str, ...]]:
    raw_query = _clean_query(str(case.get("query") or ""))
    analyzer = analyzer or QueryAnalyzer()
    analysis = analyzer.analyze(raw_query)
    rule_queries = _unique_queries(analysis.search_queries or (analysis.resolved_query,))

    p4_query = ""
    if p4_plan and bool(p4_plan.get("strict_success")):
        p4_query = str(p4_plan.get("model_query") or "")
    return {
        "raw": _unique_queries((raw_query,)),
        "rules": rule_queries,
        "p4": _unique_queries((p4_query,)),
    }


def evaluate_discovery(
    case: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    metrics = evaluate_case(case, candidates, (), {})
    return {
        "candidate_count": metrics["candidate_count"],
        "nonempty_candidates": bool(candidates),
        "domain_hit_at_5": metrics["candidate_domain_hit_at_5"],
        "domain_hit_at_10": metrics["candidate_domain_hit_at_10"],
        "domain_hit_at_20": metrics["candidate_domain_hit_at_20"],
        "target_page_hit_at_10": metrics["candidate_target_page_hit_at_10"],
        "target_page_hit_at_20": metrics["candidate_target_page_hit_at_20"],
    }


def _percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def summarize_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(records)

    def summarize(group: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not group:
            return {"total": 0}
        valid = [row for row in group if row.get("queries")]
        elapsed = [float(row.get("elapsed_ms", 0.0)) for row in group]
        output: Dict[str, Any] = {
            "total": len(group),
            "valid_query_rate": round(len(valid) / len(group), 4),
            "nonempty_candidate_rate": round(
                sum(bool(row["metrics"].get("nonempty_candidates")) for row in group)
                / len(group),
                4,
            ),
            "average_query_count": round(
                sum(len(row.get("queries") or ()) for row in group) / len(group), 3
            ),
            "average_candidate_count": round(
                sum(int(row["metrics"].get("candidate_count", 0)) for row in group)
                / len(group),
                3,
            ),
            "average_elapsed_ms": round(sum(elapsed) / len(group), 3),
            "p95_elapsed_ms": round(_percentile_95(elapsed), 3),
        }
        for name in (
            "domain_hit_at_5",
            "domain_hit_at_10",
            "domain_hit_at_20",
            "target_page_hit_at_10",
            "target_page_hit_at_20",
        ):
            output[name + "_rate"] = round(
                sum(bool(row["metrics"].get(name)) for row in group) / len(group), 4
            )
        return output

    by_strategy: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_group: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        strategy = str(row.get("strategy") or "unknown")
        by_strategy[strategy].append(row)
        by_group[strategy][f"language:{row.get('language', 'unknown')}"].append(row)
        by_group[strategy][f"category:{row.get('category', 'unknown')}"].append(row)

    overall = {name: summarize(by_strategy.get(name, ())) for name in STRATEGIES}
    raw = overall.get("raw", {})
    deltas: Dict[str, Dict[str, float]] = {}
    for strategy in STRATEGIES[1:]:
        current = overall.get(strategy, {})
        deltas[strategy + "_vs_raw"] = {
            metric: round(float(current.get(metric, 0.0)) - float(raw.get(metric, 0.0)), 4)
            for metric in (
                "domain_hit_at_5_rate",
                "domain_hit_at_10_rate",
                "domain_hit_at_20_rate",
                "target_page_hit_at_10_rate",
                "target_page_hit_at_20_rate",
                "nonempty_candidate_rate",
            )
        }
    return {
        "schema_version": "query-formation-bench.v1",
        "overall": overall,
        "deltas": deltas,
        "groups": {
            strategy: {
                key: summarize(group) for key, group in sorted(groups.items())
            }
            for strategy, groups in sorted(by_group.items())
        },
    }
