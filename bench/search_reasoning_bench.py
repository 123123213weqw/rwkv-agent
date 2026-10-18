from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence


STRATEGIES = ("direct", "short_cot", "feedback", "react")


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def summarize_group(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"total": 0}
    metrics = [dict(row.get("metrics") or {}) for row in records]
    model_elapsed = [float(row.get("model_elapsed_ms") or 0.0) for row in records]
    search_elapsed = [float(row.get("search_elapsed_ms") or 0.0) for row in records]
    total_elapsed = [float(row.get("total_elapsed_ms") or 0.0) for row in records]
    wall_elapsed = [float(row.get("wall_elapsed_ms") or 0.0) for row in records]
    action_count = sum(len(row.get("actions") or ()) for row in records)
    valid_actions = sum(
        bool(action.get("validation", {}).get("accepted"))
        for row in records
        for action in row.get("actions") or ()
        if action.get("kind") == "search"
    )
    search_actions = sum(
        action.get("kind") == "search"
        for row in records
        for action in row.get("actions") or ()
    )
    strict_actions = sum(
        bool(action.get("format_evaluation", {}).get("strict_success"))
        for row in records
        for action in row.get("actions") or ()
    )
    reasoning_actions = sum(
        bool(str(action.get("reasoning") or "").strip())
        for row in records
        for action in row.get("actions") or ()
    )
    search_action_rows = [
        action
        for row in records
        for action in row.get("actions") or ()
        if action.get("kind") == "search"
    ]
    validation_rows = [
        dict(action.get("validation") or {}) for action in search_action_rows
    ]
    stop_reason_counts: Dict[str, int] = defaultdict(int)
    for row in records:
        stop_reason_counts[str(row.get("stop_reason") or "unknown")] += 1
    output: Dict[str, Any] = {
        "total": len(records),
        "case_success_rate": round(
            sum(bool(row.get("case_success")) for row in records) / len(records), 4
        ),
        "nonempty_candidate_rate": round(
            sum(bool(metric.get("nonempty_candidates")) for metric in metrics)
            / len(records),
            4,
        ),
        "average_query_count": round(
            _mean([float(row.get("query_count") or 0) for row in records]), 3
        ),
        "average_model_call_count": round(
            _mean([float(row.get("model_call_count") or 0) for row in records]), 3
        ),
        "average_token_count": round(
            _mean([float(row.get("token_count") or 0) for row in records]), 3
        ),
        "action_count": action_count,
        "strict_action_rate": round(strict_actions / max(1, action_count), 4),
        "max_token_stop_rate": round(
            sum(
                str(action.get("stop") or "") == "max_tokens"
                for row in records
                for action in row.get("actions") or ()
            )
            / max(1, action_count),
            4,
        ),
        "reasoning_action_rate": round(reasoning_actions / max(1, action_count), 4),
        "query_validation_rate": round(valid_actions / max(1, search_actions), 4),
        "average_entity_retention_rate": round(
            _mean(
                [
                    float(value.get("entity_retention_rate") or 0.0)
                    for value in validation_rows
                ]
            ),
            4,
        ),
        "subject_drift_rate": round(
            sum("subject_drift" in (value.get("reasons") or ()) for value in validation_rows)
            / max(1, search_actions),
            4,
        ),
        "duplicate_query_rate": round(
            sum("duplicate_query" in (value.get("reasons") or ()) for value in validation_rows)
            / max(1, search_actions),
            4,
        ),
        "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
        "average_candidate_count": round(
            _mean([float(metric.get("candidate_count") or 0) for metric in metrics]), 3
        ),
        "average_model_elapsed_ms": round(_mean(model_elapsed), 3),
        "p95_model_elapsed_ms": round(_p95(model_elapsed), 3),
        "average_search_elapsed_ms": round(_mean(search_elapsed), 3),
        "p95_search_elapsed_ms": round(_p95(search_elapsed), 3),
        "average_total_elapsed_ms": round(_mean(total_elapsed), 3),
        "p95_total_elapsed_ms": round(_p95(total_elapsed), 3),
        "average_wall_elapsed_ms": round(_mean(wall_elapsed), 3),
        "p95_wall_elapsed_ms": round(_p95(wall_elapsed), 3),
        "feedback_trigger_rate": round(
            sum(bool(row.get("feedback_gate", {}).get("trigger")) for row in records)
            / len(records),
            4,
        ),
        "model_stop_rate": round(
            sum(str(row.get("stop_reason")) == "model_final" for row in records)
            / len(records),
            4,
        ),
    }
    for metric_name in (
        "domain_hit_at_5",
        "domain_hit_at_10",
        "domain_hit_at_20",
        "target_page_hit_at_10",
        "target_page_hit_at_20",
    ):
        output[metric_name + "_rate"] = round(
            sum(bool(metric.get(metric_name)) for metric in metrics) / len(records), 4
        )
    return output


def summarize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    strategies: Sequence[str] = STRATEGIES,
) -> Dict[str, Any]:
    rows = list(records)
    selected = tuple(dict.fromkeys(str(value) for value in strategies))
    if not selected or "direct" not in selected:
        raise ValueError("summaries require the direct baseline")
    unknown = set(selected) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies: {', '.join(sorted(unknown))}")
    by_strategy: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_language: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_category: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        strategy = str(row.get("strategy") or "unknown")
        by_strategy[strategy].append(row)
        by_language[strategy][str(row.get("language") or "unknown")].append(row)
        by_category[strategy][str(row.get("category") or "unknown")].append(row)
    overall = {
        strategy: summarize_group(by_strategy.get(strategy, ()))
        for strategy in selected
    }
    baseline = overall["direct"]
    deltas: Dict[str, Dict[str, float]] = {}
    for strategy in selected:
        if strategy == "direct":
            continue
        current = overall[strategy]
        deltas[strategy + "_vs_direct"] = {
            name: round(float(current.get(name, 0.0)) - float(baseline.get(name, 0.0)), 4)
            for name in (
                "domain_hit_at_5_rate",
                "domain_hit_at_10_rate",
                "domain_hit_at_20_rate",
                "target_page_hit_at_10_rate",
                "target_page_hit_at_20_rate",
                "nonempty_candidate_rate",
                "average_query_count",
                "average_total_elapsed_ms",
            )
        }
    return {
        "schema_version": "search-reasoning-bench.v1",
        "overall": overall,
        "deltas": deltas,
        "by_language": {
            strategy: {
                language: summarize_group(group)
                for language, group in sorted(groups.items())
            }
            for strategy, groups in sorted(by_language.items())
        },
        "by_category": {
            strategy: {
                category: summarize_group(group)
                for category, group in sorted(groups.items())
            }
            for strategy, groups in sorted(by_category.items())
        },
    }
