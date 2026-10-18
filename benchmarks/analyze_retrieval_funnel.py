#!/usr/bin/env python3
"""Replay frozen retrieval snapshots into an exact-page loss funnel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from benchmarks.agent_benchmark_metrics import canonical_uri
from benchmarks.agent_benchmark_schema import load_jsonl
from benchmarks.retrieval_snapshot import FUNNEL_SCHEMA_VERSION, load_snapshots


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def _domain(value: str) -> str:
    return (urlsplit(str(value or "")).hostname or "").casefold().strip(".")


def _domain_matches(actual: str, expected: str) -> bool:
    return bool(
        actual
        and expected
        and (
            actual == expected
            or actual.endswith("." + expected)
            or expected.endswith("." + actual)
        )
    )


def _urls(values: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        canonical_uri(str(value.get("url") or value.get("uri") or ""))
        for value in values
        if str(value.get("url") or value.get("uri") or "").strip()
    }


def _matched(actual: set[str], expected: set[str]) -> set[str]:
    return actual & expected


def _recall(actual: set[str], expected: set[str]) -> float:
    return len(_matched(actual, expected)) / len(expected) if expected else 1.0


def _metric(row: Mapping[str, Any], name: str) -> float:
    value = dict(row.get("metrics") or {}).get(name)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def evaluate_funnel_case(
    case: Mapping[str, Any],
    result: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    gold = {
        canonical_uri(str(value))
        for value in dict(case.get("gold") or {}).get("source_uris") or ()
        if str(value).strip()
    }
    gold_domains = {_domain(value) for value in gold if _domain(value)}
    calls = [dict(value) for value in snapshot.get("calls") or ()]

    def union(stage: str) -> set[str]:
        return set().union(
            *(
                _urls(value for value in call.get(stage) or () if isinstance(value, Mapping))
                for call in calls
            ),
            set(),
        )

    raw = union("raw_candidates")
    initial = union("initial_candidates")
    admitted = union("candidates")
    fetched = union("results")
    call_evidence = union("evidence")
    final_evidence = _urls(
        value
        for value in result.get("evidence") or ()
        if isinstance(value, Mapping)
    )
    raw_for_domain = raw or initial or admitted
    domain_hit = any(
        _domain_matches(_domain(actual), expected)
        for actual in raw_for_domain
        for expected in gold_domains
    )
    exact_raw = bool(_matched(raw, gold))
    exact_initial = bool(_matched(initial, gold))
    exact_admitted = bool(_matched(admitted, gold))
    exact_fetched = bool(_matched(fetched, gold))
    exact_call_evidence = bool(_matched(call_evidence, gold))
    exact_final_evidence = bool(_matched(final_evidence, gold))
    answer_f1 = _metric(evaluation, "answer_token_f1")
    citation_recall = _metric(evaluation, "citation_exact_page_recall")
    exact_discovered = exact_raw or exact_initial or exact_admitted
    exact_retrieved = exact_fetched or exact_call_evidence

    search_invoked = bool(calls)
    if not search_invoked:
        blocker = "search_not_invoked"
    elif not domain_hit:
        blocker = "domain_discovery_miss"
    elif not exact_discovered:
        blocker = "exact_page_discovery_miss"
    elif (exact_raw or exact_initial) and not exact_admitted:
        blocker = "candidate_admission_or_rerank_loss"
    elif exact_admitted and not exact_retrieved:
        blocker = "fetch_or_extraction_loss"
    elif exact_retrieved and not exact_final_evidence:
        blocker = "evidence_selection_loss"
    elif exact_final_evidence and answer_f1 <= 0.0:
        blocker = "answer_synthesis_failure"
    elif answer_f1 > 0.0 and citation_recall <= 0.0:
        blocker = "citation_binding_failure"
    else:
        blocker = "partial_or_pass"

    stages = {
        "search_invoked": search_invoked,
        "domain_candidate_hit": domain_hit,
        "exact_raw_candidate_hit": exact_raw,
        "exact_initial_candidate_hit": exact_initial,
        "exact_admitted_candidate_hit": exact_admitted,
        "exact_fetched_result_hit": exact_fetched,
        "exact_call_evidence_hit": exact_call_evidence,
        "exact_final_evidence_hit": exact_final_evidence,
        "answer_overlap_hit": answer_f1 > 0.0,
        "exact_citation_hit": citation_recall > 0.0,
    }
    recalls = {
        "raw_candidate_recall": round(_recall(raw, gold), 6),
        "admitted_candidate_recall": round(_recall(admitted, gold), 6),
        "fetched_result_recall": round(_recall(fetched, gold), 6),
        "final_evidence_recall": round(_recall(final_evidence, gold), 6),
        "answer_token_f1": round(answer_f1, 6),
        "citation_exact_page_recall": round(citation_recall, 6),
    }
    return {
        "schema_version": FUNNEL_SCHEMA_VERSION,
        "case_id": str(case.get("id") or ""),
        "language": str(case.get("language") or ""),
        "primary_blocker": blocker,
        "gold_page_count": len(gold),
        "call_count": len(calls),
        "stages": stages,
        "recalls": recalls,
        "matched": {
            "raw_candidates": sorted(_matched(raw, gold)),
            "admitted_candidates": sorted(_matched(admitted, gold)),
            "fetched_results": sorted(_matched(fetched, gold)),
            "final_evidence": sorted(_matched(final_evidence, gold)),
        },
    }


def build_funnel(
    cases: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = {
        "cases": {str(value["id"]): value for value in cases},
        "results": {str(value["case_id"]): value for value in results},
        "evaluations": {str(value["case_id"]): value for value in evaluations},
        "snapshots": {str(value["case_id"]): value for value in snapshots},
    }
    expected = set(groups["cases"])
    for label, values in groups.items():
        if set(values) != expected:
            raise ValueError(
                f"{label} IDs differ; missing={sorted(expected-set(values))} "
                f"extra={sorted(set(values)-expected)}"
            )
    rows = [
        evaluate_funnel_case(
            groups["cases"][case_id],
            groups["results"][case_id],
            groups["evaluations"][case_id],
            groups["snapshots"][case_id],
        )
        for case_id in sorted(expected)
    ]
    blockers = Counter(str(row["primary_blocker"]) for row in rows)
    stage_counts = Counter()
    recall_values: dict[str, list[float]] = defaultdict(list)
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[str(row["language"] or "unknown")].append(row)
        for name, passed in dict(row["stages"]).items():
            stage_counts[name] += bool(passed)
        for name, value in dict(row["recalls"]).items():
            recall_values[name].append(float(value))

    def language_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "cases": len(values),
            "blockers": dict(
                sorted(Counter(str(row["primary_blocker"]) for row in values).items())
            ),
            "stage_hit_rates": {
                name: round(
                    sum(bool(dict(row["stages"])[name]) for row in values)
                    / max(1, len(values)),
                    6,
                )
                for name in sorted(stage_counts)
            },
        }

    summary = {
        "schema_version": FUNNEL_SCHEMA_VERSION,
        "cases": len(rows),
        "primary_blockers": dict(sorted(blockers.items())),
        "stage_hit_rates": {
            name: round(count / max(1, len(rows)), 6)
            for name, count in sorted(stage_counts.items())
        },
        "stage_macro_recalls": {
            name: round(sum(values) / max(1, len(values)), 6)
            for name, values in sorted(recall_values.items())
        },
        "by_language": {
            name: language_summary(values)
            for name, values in sorted(by_language.items())
        },
    }
    return summary, rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay frozen retrieval snapshots into a loss funnel."
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args(argv)
    cases = load_jsonl(args.cases, kind="case")
    results = load_jsonl(args.results, kind="result")
    evaluations = _load_objects(args.evaluations)
    snapshots = load_snapshots(args.snapshots)
    summary, rows = build_funnel(cases, results, evaluations, snapshots)
    summary["inputs"] = {
        "cases_sha256": sha256(args.cases),
        "results_sha256": sha256(args.results),
        "evaluations_sha256": sha256(args.evaluations),
        "snapshots_sha256": sha256(args.snapshots),
    }
    _write_json(args.output, summary)
    _write_jsonl(args.rows_output, rows)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
