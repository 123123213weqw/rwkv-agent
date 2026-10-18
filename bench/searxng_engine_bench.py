from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

if __package__:
    from .retrieval_metrics import (
        classify_garbage_types,
        evaluate_candidate_stage,
        normalized_domain,
    )
else:
    from retrieval_metrics import (  # type: ignore
        classify_garbage_types,
        evaluate_candidate_stage,
        normalized_domain,
    )


SCHEMA_VERSION = "searxng-engine-run.v1"


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def enabled_search_engines(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return enabled search engines exposed by the running SearXNG instance."""
    output: List[Dict[str, Any]] = []
    for value in config.get("engines", ()):
        if not isinstance(value, Mapping) or not value.get("enabled"):
            continue
        name = str(value.get("name") or "").strip()
        categories = sorted(
            {str(item).strip() for item in value.get("categories", ()) if item}
        )
        if not name or not categories:
            continue
        output.append(
            {
                "name": name,
                "categories": categories,
                "language_support": bool(value.get("language_support")),
                "paging": bool(value.get("paging")),
                "time_range_support": bool(value.get("time_range_support")),
                "timeout": float(value.get("timeout") or 0.0),
            }
        )
    return sorted(output, key=lambda item: item["name"])


def engine_is_unresponsive(engine: str, values: Sequence[Any]) -> bool:
    wanted = engine.casefold()
    for value in values:
        if isinstance(value, (list, tuple)):
            name = str(value[0] if value else "")
        elif isinstance(value, Mapping):
            name = str(value.get("engine") or value.get("name") or "")
        else:
            name = str(value)
        if name.casefold() == wanted:
            return True
    return False


def evaluate_engine_record(
    case: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    request_success: bool,
) -> Dict[str, Any]:
    metrics = evaluate_candidate_stage(case, candidates)
    garbage = [
        kind for candidate in candidates for kind in classify_garbage_types(candidate)
    ]
    domains = [
        normalized_domain(str(candidate.get("url") or "")) for candidate in candidates
    ]
    domains = [value for value in domains if value]
    metrics.update(
        {
            "request_success": bool(request_success),
            "nonempty": bool(candidates),
            "garbage_result_count": len(garbage),
            "garbage_result_rate": round(len(garbage) / max(1, len(candidates)), 4),
            "unique_domain_count": len(set(domains)),
            "unique_domain_ratio": round(len(set(domains)) / max(1, len(domains)), 4),
        }
    )
    return metrics


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"requests": 0}
    metrics = [row.get("metrics", {}) for row in rows]
    elapsed = [float(row.get("elapsed_ms") or 0.0) for row in rows]
    candidate_count = sum(int(value.get("candidate_count") or 0) for value in metrics)
    garbage_count = sum(int(value.get("garbage_result_count") or 0) for value in metrics)

    def rate(name: str) -> float:
        return round(sum(bool(value.get(name)) for value in metrics) / len(metrics), 4)

    return {
        "requests": len(rows),
        "request_success_rate": rate("request_success"),
        "nonempty_rate": rate("nonempty"),
        "domain_recall_at_10": rate("candidate_domain_hit_at_10"),
        "target_page_recall_at_20": rate("candidate_target_page_hit_at_20"),
        "candidate_count": candidate_count,
        "average_candidate_count": round(candidate_count / len(rows), 3),
        "garbage_result_count": garbage_count,
        "garbage_result_rate": round(garbage_count / max(1, candidate_count), 4),
        "average_unique_domain_ratio": round(
            mean(float(value.get("unique_domain_ratio") or 0.0) for value in metrics),
            4,
        ),
        "average_elapsed_ms": round(mean(elapsed), 3),
        "p95_elapsed_ms": round(percentile(elapsed, 0.95), 3),
    }


def _stable_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_case: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["id"])].append(row)
    if not by_case:
        return {"case_count": 0}

    def stable(name: str) -> int:
        return sum(
            bool(case_rows) and all(bool(row["metrics"].get(name)) for row in case_rows)
            for case_rows in by_case.values()
        )

    def intermittent(name: str) -> int:
        count = 0
        for case_rows in by_case.values():
            values = {bool(row["metrics"].get(name)) for row in case_rows}
            count += len(values) > 1
        return count

    total = len(by_case)
    stable_domain = stable("candidate_domain_hit_at_10")
    stable_target = stable("candidate_target_page_hit_at_20")
    stable_nonempty = stable("nonempty")
    stable_success = stable("request_success")
    return {
        "case_count": total,
        "stable_request_success_rate": round(stable_success / total, 4),
        "stable_nonempty_rate": round(stable_nonempty / total, 4),
        "stable_domain_recall_at_10": round(stable_domain / total, 4),
        "stable_target_page_recall_at_20": round(stable_target / total, 4),
        "intermittent_domain_hit_cases": intermittent("candidate_domain_hit_at_10"),
        "intermittent_target_hit_cases": intermittent(
            "candidate_target_page_hit_at_20"
        ),
        "intermittent_nonempty_cases": intermittent("nonempty"),
    }


def summarize_records(
    records: Iterable[Mapping[str, Any]],
    engine_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = list(records)
    specs = {str(value["name"]): dict(value) for value in engine_specs}
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["engine"])].append(row)

    engines: Dict[str, Any] = {}
    for engine, engine_rows in sorted(grouped.items()):
        repetitions = sorted({int(row["repetition"]) for row in engine_rows})
        per_repeat = {
            str(repetition): _summary(
                [row for row in engine_rows if int(row["repetition"]) == repetition]
            )
            for repetition in repetitions
        }
        languages = {
            language: {
                **_summary(
                    [row for row in engine_rows if row.get("language") == language]
                ),
                **_stable_metrics(
                    [row for row in engine_rows if row.get("language") == language]
                ),
            }
            for language in ("zh", "en")
        }
        engines[engine] = {
            "spec": specs.get(engine, {"name": engine, "categories": []}),
            "overall": {**_summary(engine_rows), **_stable_metrics(engine_rows)},
            "languages": languages,
            "repetitions": per_repeat,
        }

    return {
        "schema_version": "searxng-engine-bench-summary.v1",
        "record_count": len(rows),
        "engine_count": len(engines),
        "engines": engines,
    }


def rank_general_engines(summary: Mapping[str, Any], language: str) -> List[str]:
    candidates = []
    for name, value in summary.get("engines", {}).items():
        categories = set(value.get("spec", {}).get("categories", ()))
        if "general" not in categories:
            continue
        metrics = value.get("languages", {}).get(language, {})
        candidates.append(
            (
                str(name),
                float(metrics.get("stable_domain_recall_at_10") or 0.0),
                float(metrics.get("stable_target_page_recall_at_20") or 0.0),
                float(metrics.get("stable_nonempty_rate") or 0.0),
                float(metrics.get("request_success_rate") or 0.0),
                -float(metrics.get("garbage_result_rate") or 0.0),
                -float(metrics.get("p95_elapsed_ms") or 0.0),
            )
        )
    candidates.sort(key=lambda value: value[1:], reverse=True)
    return [value[0] for value in candidates]


def public_case_matrix(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Preserve reviewable outcome flags without publishing result URLs or text."""
    output = []
    for row in records:
        metrics = row.get("metrics", {})
        output.append(
            {
                "id": row.get("id"),
                "engine": row.get("engine"),
                "repetition": row.get("repetition"),
                "language": row.get("language"),
                "category": row.get("category"),
                "request_success": bool(metrics.get("request_success")),
                "nonempty": bool(metrics.get("nonempty")),
                "domain_hit_at_10": bool(
                    metrics.get("candidate_domain_hit_at_10")
                ),
                "target_page_hit_at_20": bool(
                    metrics.get("candidate_target_page_hit_at_20")
                ),
                "candidate_count": int(metrics.get("candidate_count") or 0),
                "garbage_result_count": int(
                    metrics.get("garbage_result_count") or 0
                ),
                "elapsed_ms": float(row.get("elapsed_ms") or 0.0),
                "error_type": row.get("error_type") or "",
            }
        )
    return output
