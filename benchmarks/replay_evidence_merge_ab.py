#!/usr/bin/env python3
"""Offline paired A/B for final Evidence selection on frozen Web snapshots.

Both selectors run before Gold is read.  The experiment changes only the final
merge policy: the control is the existing global query-view MMR selector; the
candidate reserves a bounded set of generated-query representatives and then
uses the same global selector for the remaining slots.  It performs no search,
fetch, model call, or production-state change.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from benchmarks.agent_benchmark_metrics import canonical_uri
from benchmarks.agent_benchmark_schema import load_jsonl
from benchmarks.retrieval_snapshot import load_snapshots
from rwkv_agent.state_agent import _merge_evidence


SCHEMA_VERSION = "rwkv-agent-evidence-merge-ab.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _uris(items: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        canonical_uri(str(item.get("uri") or item.get("url") or ""))
        for item in items
        if str(item.get("uri") or item.get("url") or "").strip()
    }


def _recall(actual: set[str], expected: set[str]) -> float:
    return len(actual & expected) / len(expected) if expected else 1.0


def replay_selectors(
    question: str,
    snapshot: Mapping[str, Any],
    *,
    limit: int = 8,
) -> dict[str, Any]:
    """Run both selectors without accepting or reading Gold labels."""

    results: list[dict[str, Any]] = []
    for call in snapshot.get("calls") or ():
        if not isinstance(call, Mapping):
            continue
        query = " ".join(
            str(call.get("effective_query") or call.get("query") or "").split()
        ).strip()
        evidence = [
            dict(item)
            for item in call.get("evidence") or ()
            if isinstance(item, Mapping)
        ]
        if not evidence:
            continue
        results.append({"query": query, "evidence": evidence})

    control = _merge_evidence(
        results,
        question=question,
        limit=limit,
        preserve_query_views=False,
    )
    candidate = _merge_evidence(
        results,
        question=question,
        limit=limit,
        preserve_query_views=True,
    )
    return {
        "call_count": len(snapshot.get("calls") or ()),
        "nonempty_query_views": len(results),
        "call_evidence_uris": sorted(
            set().union(*(_uris(result["evidence"]) for result in results), set())
        ),
        "control_uris": sorted(_uris(control)),
        "candidate_uris": sorted(_uris(candidate)),
    }


def evaluate_replay(
    case: Mapping[str, Any],
    replay: Mapping[str, Any],
    *,
    actual_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read Gold only after both frozen-snapshot selections have completed."""

    gold = {
        canonical_uri(str(value))
        for value in dict(case.get("gold") or {}).get("source_uris") or ()
        if str(value).strip()
    }
    call_evidence = set(replay.get("call_evidence_uris") or ())
    control = set(replay.get("control_uris") or ())
    candidate = set(replay.get("candidate_uris") or ())
    actual = (
        _uris(actual_result.get("evidence") or ())
        if actual_result is not None
        else set()
    )
    control_recall = _recall(control, gold)
    candidate_recall = _recall(candidate, gold)
    delta = candidate_recall - control_recall
    actual_recall = _recall(actual, gold) if actual_result is not None else None
    control_actual_union = control | actual
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(case.get("id") or ""),
        "language": str(case.get("language") or ""),
        "call_count": int(replay.get("call_count") or 0),
        "nonempty_query_views": int(replay.get("nonempty_query_views") or 0),
        "gold_page_count": len(gold),
        "reachable_gold_count": len(gold & call_evidence),
        "control_exact_page_recall": round(control_recall, 6),
        "candidate_exact_page_recall": round(candidate_recall, 6),
        "delta_exact_page_recall": round(delta, 6),
        "comparison": "win" if delta > 0 else "loss" if delta < 0 else "tie",
        "call_evidence_uri_count": len(call_evidence),
        "control_uri_count": len(control),
        "candidate_uri_count": len(candidate),
        **(
            {
                "actual_final_exact_page_recall": round(
                    float(actual_recall), 6
                ),
                "actual_final_uri_count": len(actual),
                "control_actual_uri_jaccard": round(
                    len(control & actual) / len(control_actual_union)
                    if control_actual_union
                    else 1.0,
                    6,
                ),
            }
            if actual_result is not None
            else {}
        ),
        "control_call_evidence_retention": round(
            len(control & call_evidence) / len(call_evidence)
            if call_evidence
            else 1.0,
            6,
        ),
        "candidate_call_evidence_retention": round(
            len(candidate & call_evidence) / len(call_evidence)
            if call_evidence
            else 1.0,
            6,
        ),
        "matched": {
            "reachable_gold": sorted(gold & call_evidence),
            "control_gold": sorted(gold & control),
            "candidate_gold": sorted(gold & candidate),
            **(
                {"actual_final_gold": sorted(gold & actual)}
                if actual_result is not None
                else {}
            ),
        },
        "selected": {
            "control": list(replay.get("control_uris") or ()),
            "candidate": list(replay.get("candidate_uris") or ()),
        },
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "control_exact_page_recall": 0.0,
            "candidate_exact_page_recall": 0.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
        }
    output = {
        "cases": len(rows),
        "search_invoked_cases": sum(int(row.get("call_count") or 0) > 0 for row in rows),
        "reachable_cases": sum(int(row.get("reachable_gold_count") or 0) > 0 for row in rows),
        "control_exact_page_recall": round(
            sum(float(row["control_exact_page_recall"]) for row in rows)
            / len(rows),
            6,
        ),
        "candidate_exact_page_recall": round(
            sum(float(row["candidate_exact_page_recall"]) for row in rows)
            / len(rows),
            6,
        ),
        "wins": sum(row.get("comparison") == "win" for row in rows),
        "losses": sum(row.get("comparison") == "loss" for row in rows),
        "ties": sum(row.get("comparison") == "tie" for row in rows),
        "mean_control_call_evidence_retention": round(
            sum(float(row["control_call_evidence_retention"]) for row in rows)
            / len(rows),
            6,
        ),
        "mean_candidate_call_evidence_retention": round(
            sum(float(row["candidate_call_evidence_retention"]) for row in rows)
            / len(rows),
            6,
        ),
    }
    actual_rows = [
        row for row in rows if "actual_final_exact_page_recall" in row
    ]
    if actual_rows:
        output.update(
            {
                "actual_final_exact_page_recall": round(
                    sum(
                        float(row["actual_final_exact_page_recall"])
                        for row in actual_rows
                    )
                    / len(actual_rows),
                    6,
                ),
                "mean_control_actual_uri_jaccard": round(
                    sum(
                        float(row["control_actual_uri_jaccard"])
                        for row in actual_rows
                    )
                    / len(actual_rows),
                    6,
                ),
            }
        )
    return output


def run(
    *,
    cases_path: Path,
    snapshots_path: Path,
    output_dir: Path,
    results_path: Path | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    cases = load_jsonl(cases_path, kind="case")
    snapshots = load_snapshots(snapshots_path)
    snapshot_by_id = {str(row["case_id"]): row for row in snapshots}
    result_by_id = (
        {
            str(row["case_id"]): row
            for row in load_jsonl(results_path, kind="result")
        }
        if results_path is not None
        else {}
    )
    if set(snapshot_by_id) != {str(case["id"]) for case in cases}:
        raise ValueError("case and retrieval-snapshot IDs must match exactly")
    if result_by_id and set(result_by_id) != set(snapshot_by_id):
        raise ValueError("result and retrieval-snapshot IDs must match exactly")

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        # This call is intentionally complete before evaluate_replay can access
        # case.gold, preventing label-dependent selector behavior.
        replay = replay_selectors(
            str(case["prompt"]),
            snapshot_by_id[case_id],
            limit=limit,
        )
        rows.append(
            evaluate_replay(
                case,
                replay,
                actual_result=result_by_id.get(case_id),
            )
        )

    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[str(row.get("language") or "unknown")].append(row)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "selector_control": "query_view_mmr_v1",
        "selector_candidate": "cross_view_confirmed_append_mmr_v1",
        "evidence_limit": int(limit),
        "offline_only": True,
        "gold_read_after_both_selections": True,
        "snapshot_content_limitation": (
            "snapshots omit fetched page bodies, so replayed control ranking is "
            "paired and deterministic but need not reproduce the recorded "
            "content-aware final Evidence set"
        ),
        "inputs": {
            "cases": str(cases_path.resolve()),
            "cases_sha256": _sha256(cases_path),
            "snapshots": str(snapshots_path.resolve()),
            "snapshots_sha256": _sha256(snapshots_path),
            **(
                {
                    "results": str(results_path.resolve()),
                    "results_sha256": _sha256(results_path),
                }
                if results_path is not None
                else {}
            ),
        },
        "overall": _aggregate(rows),
        "by_language": {
            language: _aggregate(values)
            for language, values in sorted(by_language.items())
        },
    }
    _dump_jsonl(output_dir / "rows.jsonl", rows)
    _dump_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay frozen Web call Evidence through paired merge policies"
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args(argv)
    if not 2 <= args.limit <= 32:
        parser.error("--limit must be between 2 and 32")
    summary = run(
        cases_path=args.cases.expanduser().resolve(),
        snapshots_path=args.snapshots.expanduser().resolve(),
        results_path=(
            args.results.expanduser().resolve()
            if args.results is not None
            else None
        ),
        output_dir=args.output_dir.expanduser().resolve(),
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
