from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    from .retrieval_metrics import (
        domain_matches,
        normalized_domain,
        target_page_matches,
    )
except ImportError:
    from retrieval_metrics import (  # type: ignore
        domain_matches,
        normalized_domain,
        target_page_matches,
    )


ATTRIBUTION_SCHEMA_VERSION = "realtime-retrieval-attribution.v1"


def load_run(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def attribute_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    expected_domains = [
        str(value) for value in record.get("expected_domains_any", ())
    ]
    target_patterns = [
        str(value) for value in record.get("target_url_patterns_any", ())
    ]
    initial = _mapping_list(record.get("initial_candidates"))
    post_pivot = _mapping_list(record.get("post_pivot_candidates"))
    final = _mapping_list(record.get("candidates"))
    fetches = _mapping_list(record.get("fetches"))
    results = _mapping_list(record.get("results"))

    initial_domain = _has_domain(initial, expected_domains)
    post_pivot_domain = _has_domain(post_pivot, expected_domains)
    final_domain = _has_domain(final, expected_domains)
    result_domain = _has_domain(results, expected_domains)
    initial_target = _has_target(initial, expected_domains, target_patterns)
    post_pivot_target = _has_target(
        post_pivot, expected_domains, target_patterns
    )
    final_target = _has_target(final, expected_domains, target_patterns)
    result_target = _has_target(results, expected_domains, target_patterns)

    domain_fetches = [
        item
        for item in fetches
        if _fetch_matches_domain(item, expected_domains)
    ]
    target_fetches = [
        item
        for item in fetches
        if _fetch_matches_target(item, expected_domains, target_patterns)
    ]
    domain_fetch_succeeded = any(
        str(item.get("status")) == "succeeded" for item in domain_fetches
    )
    target_fetch_succeeded = any(
        str(item.get("status")) == "succeeded" for item in target_fetches
    )
    domain_post_fetch_rejected = any(
        item.get("admission_rejection_reasons") for item in domain_fetches
    )
    target_post_fetch_rejected = any(
        item.get("admission_rejection_reasons") for item in target_fetches
    )
    domain_failure_types = _failure_types(domain_fetches)
    target_failure_types = _failure_types(target_fetches)

    return {
        "id": str(record.get("id") or ""),
        "language": str(record.get("language") or ""),
        "category": str(record.get("category") or ""),
        "initial_domain_hit": initial_domain,
        "post_pivot_domain_hit": post_pivot_domain,
        "final_candidate_domain_hit": final_domain,
        "result_domain_hit": result_domain,
        "domain_stage": _first_stage(
            initial_domain, post_pivot_domain, final_domain
        ),
        "domain_scheduled": bool(domain_fetches),
        "domain_fetch_succeeded": domain_fetch_succeeded,
        "domain_failure_types": domain_failure_types,
        "domain_post_fetch_rejected": domain_post_fetch_rejected,
        "domain_outcome": _domain_outcome(
            initial_hit=initial_domain,
            post_pivot_hit=post_pivot_domain,
            final_hit=final_domain,
            scheduled=bool(domain_fetches),
            fetch_succeeded=domain_fetch_succeeded,
            post_fetch_rejected=domain_post_fetch_rejected,
            result_hit=result_domain,
        ),
        "initial_target_hit": initial_target,
        "post_pivot_target_hit": post_pivot_target,
        "final_candidate_target_hit": final_target,
        "result_target_hit": result_target,
        "target_stage": _first_stage(
            initial_target, post_pivot_target, final_target
        ),
        "target_scheduled": bool(target_fetches),
        "target_fetch_succeeded": target_fetch_succeeded,
        "target_failure_types": target_failure_types,
        "target_post_fetch_rejected": target_post_fetch_rejected,
        "target_outcome": _target_outcome(
            initial_domain_hit=initial_domain,
            initial_hit=initial_target,
            post_pivot_hit=post_pivot_target,
            final_hit=final_target,
            scheduled=bool(target_fetches),
            fetch_succeeded=target_fetch_succeeded,
            post_fetch_rejected=target_post_fetch_rejected,
            result_hit=result_target,
        ),
        "discovery_request_count": int(
            (record.get("stats") or {}).get("discovery_request_count", 0) or 0
        ),
        "fetch_attempted": int(
            (record.get("stats") or {}).get("attempted", 0) or 0
        ),
        "fetch_usable": int(
            (record.get("stats") or {}).get("usable", 0) or 0
        ),
        "total_elapsed_ms": float(record.get("total_elapsed_ms", 0.0) or 0.0),
    }


def aggregate_attributions(
    run_paths: Sequence[Path],
    *,
    reference: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for run_index, path in enumerate(run_paths, 1):
        records = load_run(path)
        attributions = [attribute_record(record) for record in records]
        run_summary = _aggregate_one(attributions)
        run_summary.update(
            {
                "run_index": run_index,
                "path": str(path),
                "case_count": len(records),
            }
        )
        runs.append(run_summary)
    output: Dict[str, Any] = {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "run_count": len(runs),
        "runs": runs,
    }
    if runs:
        output["across_runs"] = _aggregate_across_runs(runs)
    if reference:
        output["reference"] = dict(reference)
    return output


def case_matrix(run_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for run_index, path in enumerate(run_paths, 1):
        for record in load_run(path):
            attributed = attribute_record(record)
            attributed["run_index"] = run_index
            output.append(attributed)
    return output


def _aggregate_one(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    domain_outcomes = Counter(str(row["domain_outcome"]) for row in rows)
    target_outcomes = Counter(str(row["target_outcome"]) for row in rows)
    elapsed = [float(row["total_elapsed_ms"]) for row in rows]
    return {
        "domain_outcomes": dict(sorted(domain_outcomes.items())),
        "target_outcomes": dict(sorted(target_outcomes.items())),
        "domain_fetch_failure_types": _aggregate_failure_types(
            rows, "domain_failure_types"
        ),
        "target_fetch_failure_types": _aggregate_failure_types(
            rows, "target_failure_types"
        ),
        "stage_recall": {
            "initial_domain": _rate(rows, "initial_domain_hit"),
            "post_pivot_domain": _rate(rows, "post_pivot_domain_hit"),
            "final_candidate_domain": _rate(rows, "final_candidate_domain_hit"),
            "result_domain": _rate(rows, "result_domain_hit"),
            "initial_target": _rate(rows, "initial_target_hit"),
            "post_pivot_target": _rate(rows, "post_pivot_target_hit"),
            "final_candidate_target": _rate(
                rows, "final_candidate_target_hit"
            ),
            "result_target": _rate(rows, "result_target_hit"),
        },
        "stage_gains": {
            "pivot_domain_additions": sum(
                bool(row["post_pivot_domain_hit"])
                and not bool(row["initial_domain_hit"])
                for row in rows
            ),
            "one_hop_domain_additions": sum(
                bool(row["final_candidate_domain_hit"])
                and not bool(row["post_pivot_domain_hit"])
                for row in rows
            ),
            "pivot_target_additions": sum(
                bool(row["post_pivot_target_hit"])
                and not bool(row["initial_target_hit"])
                for row in rows
            ),
            "one_hop_target_additions": sum(
                bool(row["final_candidate_target_hit"])
                and not bool(row["post_pivot_target_hit"])
                for row in rows
            ),
        },
        "resource_totals": {
            "discovery_requests": sum(
                int(row["discovery_request_count"]) for row in rows
            ),
            "fetch_attempted": sum(int(row["fetch_attempted"]) for row in rows),
            "fetch_usable": sum(int(row["fetch_usable"]) for row in rows),
        },
        "average_total_elapsed_ms": round(statistics.mean(elapsed), 3)
        if elapsed
        else 0.0,
        "p95_total_elapsed_ms": _percentile(elapsed, 0.95),
        "case_count": count,
    }


def _aggregate_across_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metric_names = tuple((runs[0].get("stage_recall") or {}).keys())
    stage_ranges: Dict[str, Any] = {}
    for name in metric_names:
        values = [float(run["stage_recall"][name]) for run in runs]
        stage_ranges[name] = {
            "mean": round(statistics.mean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
    return {"stage_recall_ranges": stage_ranges}


def _domain_outcome(
    *,
    initial_hit: bool,
    post_pivot_hit: bool,
    final_hit: bool,
    scheduled: bool,
    fetch_succeeded: bool,
    post_fetch_rejected: bool,
    result_hit: bool,
) -> str:
    if result_hit:
        return "success_result"
    if not final_hit:
        if not initial_hit and not post_pivot_hit:
            return "domain_not_discovered"
        return "candidate_lost_after_discovery"
    if not scheduled:
        return "domain_not_scheduled"
    if not fetch_succeeded:
        return "domain_fetch_or_extraction_failed"
    if post_fetch_rejected:
        return "domain_post_fetch_rejected"
    return "domain_final_ranking_drop"


def _target_outcome(
    *,
    initial_domain_hit: bool,
    initial_hit: bool,
    post_pivot_hit: bool,
    final_hit: bool,
    scheduled: bool,
    fetch_succeeded: bool,
    post_fetch_rejected: bool,
    result_hit: bool,
) -> str:
    if result_hit:
        return "success_result"
    if not final_hit:
        if not initial_domain_hit:
            return "initial_domain_miss"
        if initial_hit or post_pivot_hit:
            return "candidate_lost_after_discovery"
        return "exact_page_not_discovered_after_precision"
    if not scheduled:
        return "target_not_scheduled"
    if not fetch_succeeded:
        return "target_fetch_or_extraction_failed"
    if post_fetch_rejected:
        return "target_post_fetch_rejected"
    return "target_final_ranking_drop"


def _first_stage(initial: bool, post_pivot: bool, final: bool) -> str:
    if initial:
        return "initial"
    if post_pivot:
        return "domain_pivot"
    if final:
        return "one_hop_link"
    return "none"


def _has_domain(
    items: Sequence[Mapping[str, Any]], expected_domains: Sequence[str]
) -> bool:
    return any(
        domain_matches(normalized_domain(str(item.get("url") or "")), expected)
        for item in items
        for expected in expected_domains
    )


def _has_target(
    items: Sequence[Mapping[str, Any]],
    expected_domains: Sequence[str],
    target_patterns: Sequence[str],
) -> bool:
    return any(
        target_page_matches(
            str(item.get("url") or ""), expected_domains, target_patterns
        )
        for item in items
    )


def _fetch_matches_domain(
    item: Mapping[str, Any], expected_domains: Sequence[str]
) -> bool:
    return any(
        domain_matches(normalized_domain(url), expected)
        for url in _fetch_urls(item)
        for expected in expected_domains
    )


def _fetch_matches_target(
    item: Mapping[str, Any],
    expected_domains: Sequence[str],
    target_patterns: Sequence[str],
) -> bool:
    return any(
        target_page_matches(url, expected_domains, target_patterns)
        for url in _fetch_urls(item)
    )


def _fetch_urls(item: Mapping[str, Any]) -> List[str]:
    return [
        value
        for value in (
            str(item.get("requested_url") or ""),
            str(item.get("final_url") or ""),
        )
        if value
    ]


def _failure_types(items: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted(
        str(item.get("error_type") or "unknown_failure")
        for item in items
        if str(item.get("status") or "") != "succeeded"
    )


def _aggregate_failure_types(
    rows: Sequence[Mapping[str, Any]], key: str
) -> Dict[str, int]:
    values = Counter(
        str(value)
        for row in rows
        for value in row.get(key, ())
    )
    return dict(sorted(values.items()))


def _mapping_list(value: Any) -> List[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return round(sum(bool(row[key]) for row in rows) / max(1, len(rows)), 4)


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))
    return round(ordered[index], 3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attribute realtime retrieval failures by observable stage"
    )
    parser.add_argument("run", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-matrix", type=Path, required=True)
    args = parser.parse_args()
    reference = (
        json.loads(args.reference.read_text(encoding="utf-8"))
        if args.reference
        else None
    )
    summary = aggregate_attributions(args.run, reference=reference)
    matrix = case_matrix(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.case_matrix.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.case_matrix.write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
