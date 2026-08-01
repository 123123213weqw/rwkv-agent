#!/usr/bin/env python3
"""Measure whether a live query lane adds URLs beyond a frozen raw snapshot.

The live query-lane runner compares one historical model query with one new
query.  That can overstate gains when the original Agent already discovered
the same page in a later call.  This replay therefore treats every frozen raw
candidate from every call as the control ceiling and appends the lane URLs
without deleting or reranking control URLs.  Gold is read only after both URL
sets have been constructed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from benchmarks.agent_benchmark_metrics import canonical_uri
from benchmarks.agent_benchmark_schema import load_jsonl
from benchmarks.retrieval_snapshot import load_snapshots


SCHEMA_VERSION = "rwkv-agent-discovery-lane-snapshot-union.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(dict(value))
    ids = [str(row.get("case_id") or "") for row in rows]
    if any(not value for value in ids):
        raise ValueError("lane rows must contain non-empty case_id values")
    if len(ids) != len(set(ids)):
        raise ValueError("lane rows contain duplicate case_id values")
    return rows


def _uris(items: Iterable[Mapping[str, Any]]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        uri = canonical_uri(str(item.get("url") or item.get("uri") or ""))
        if not uri or uri in seen:
            continue
        seen.add(uri)
        output.append(uri)
    return output


def _host(uri: str) -> str:
    return (urlsplit(uri).hostname or "").casefold().removeprefix("www.")


def _same_host_or_subdomain(actual: str, expected: str) -> bool:
    return bool(
        actual
        and expected
        and (actual == expected or actual.endswith("." + expected))
    )


def replay_candidate_sets(
    snapshot: Mapping[str, Any],
    lane_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Build both arms without accepting a case or any Gold labels."""

    raw_items = [
        dict(item)
        for call in snapshot.get("calls") or ()
        if isinstance(call, Mapping)
        for item in call.get("raw_candidates") or ()
        if isinstance(item, Mapping)
    ]
    lane_items = [
        dict(item)
        for item in dict(lane_row.get("candidates") or {}).get("lane_only") or ()
        if isinstance(item, Mapping)
    ]
    control = _uris(raw_items)
    lane = _uris(lane_items)
    union = list(dict.fromkeys([*control, *lane]))
    control_set = set(control)
    scope_host = str(lane_row.get("scope_host") or "").casefold().removeprefix(
        "www."
    )
    lane_in_scope = [
        uri
        for uri in lane
        if not scope_host or _same_host_or_subdomain(_host(uri), scope_host)
    ]
    return {
        "search_invoked": bool(snapshot.get("calls")),
        "control_uris": control,
        "lane_uris": lane,
        "lane_in_scope_uris": lane_in_scope,
        "union_uris": union,
        "new_lane_uris": [uri for uri in lane if uri not in control_set],
        "new_lane_in_scope_uris": [
            uri for uri in lane_in_scope if uri not in control_set
        ],
    }


def _recall(actual: set[str], expected: set[str]) -> float:
    return len(actual & expected) / len(expected) if expected else 1.0


def evaluate_replay(
    case: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Read Gold only after the non-destructive candidate replay is complete."""

    gold = {
        canonical_uri(str(value))
        for value in dict(case.get("gold") or {}).get("source_uris") or ()
        if str(value).strip()
    }
    control = set(replay.get("control_uris") or ())
    lane = set(replay.get("lane_uris") or ())
    union = set(replay.get("union_uris") or ())
    control_recall = _recall(control, gold)
    union_recall = _recall(union, gold)
    delta = union_recall - control_recall
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(case.get("id") or ""),
        "language": str(case.get("language") or "unknown"),
        "search_invoked": bool(replay.get("search_invoked")),
        "gold_page_count": len(gold),
        "control_raw_candidate_count": len(control),
        "lane_candidate_count": len(lane),
        "lane_in_scope_count": len(replay.get("lane_in_scope_uris") or ()),
        "new_lane_candidate_count": len(replay.get("new_lane_uris") or ()),
        "new_lane_in_scope_count": len(
            replay.get("new_lane_in_scope_uris") or ()
        ),
        "control_exact_page_hit": bool(control & gold),
        "union_exact_page_hit": bool(union & gold),
        "control_exact_page_recall": round(control_recall, 6),
        "union_exact_page_recall": round(union_recall, 6),
        "delta_exact_page_recall": round(delta, 6),
        "comparison": "win" if delta > 0 else "loss" if delta < 0 else "tie",
        "matched": {
            "control_gold": sorted(control & gold),
            "lane_gold": sorted(lane & gold),
            "union_gold": sorted(union & gold),
        },
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "control_exact_page_hit_rate": 0.0,
            "union_exact_page_hit_rate": 0.0,
            "control_exact_page_macro_recall": 0.0,
            "union_exact_page_macro_recall": 0.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
        }
    count = len(rows)
    return {
        "cases": count,
        "search_invoked_cases": sum(bool(row.get("search_invoked")) for row in rows),
        "control_exact_page_hit_rate": round(
            sum(bool(row.get("control_exact_page_hit")) for row in rows) / count,
            6,
        ),
        "union_exact_page_hit_rate": round(
            sum(bool(row.get("union_exact_page_hit")) for row in rows) / count,
            6,
        ),
        "control_exact_page_macro_recall": round(
            sum(float(row.get("control_exact_page_recall") or 0.0) for row in rows)
            / count,
            6,
        ),
        "union_exact_page_macro_recall": round(
            sum(float(row.get("union_exact_page_recall") or 0.0) for row in rows)
            / count,
            6,
        ),
        "wins": sum(row.get("comparison") == "win" for row in rows),
        "losses": sum(row.get("comparison") == "loss" for row in rows),
        "ties": sum(row.get("comparison") == "tie" for row in rows),
        "mean_new_lane_candidates": round(
            sum(int(row.get("new_lane_candidate_count") or 0) for row in rows)
            / count,
            6,
        ),
        "mean_new_lane_in_scope_candidates": round(
            sum(int(row.get("new_lane_in_scope_count") or 0) for row in rows)
            / count,
            6,
        ),
    }


def _dump_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def run(
    *,
    cases_path: Path,
    snapshots_path: Path,
    lane_rows_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    cases = load_jsonl(cases_path, kind="case")
    snapshots = load_snapshots(snapshots_path)
    lane_rows = _load_rows(lane_rows_path)
    case_ids = {str(case["id"]) for case in cases}
    snapshot_by_id = {str(row["case_id"]): row for row in snapshots}
    lane_by_id = {str(row["case_id"]): row for row in lane_rows}
    if set(snapshot_by_id) != case_ids:
        raise ValueError("case and retrieval-snapshot IDs must match exactly")
    if set(lane_by_id) != case_ids:
        raise ValueError("case and lane-row IDs must match exactly")

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        replay = replay_candidate_sets(
            snapshot_by_id[case_id],
            lane_by_id[case_id],
        )
        rows.append(evaluate_replay(case, replay))

    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[str(row.get("language") or "unknown")].append(row)
    lane_modes = sorted(
        {
            str(row.get("lane_mode") or "unknown")
            for row in lane_rows
        }
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "offline_only": True,
        "control": "all_raw_candidates_from_all_frozen_agent_calls",
        "candidate": "control_plus_live_lane_candidates_without_deletion",
        "gold_read_after_both_candidate_sets": True,
        "lane_modes": lane_modes,
        "inputs": {
            "cases": str(cases_path.resolve()),
            "cases_sha256": _sha256(cases_path),
            "snapshots": str(snapshots_path.resolve()),
            "snapshots_sha256": _sha256(snapshots_path),
            "lane_rows": str(lane_rows_path.resolve()),
            "lane_rows_sha256": _sha256(lane_rows_path),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--lane-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run(
        cases_path=args.cases.expanduser().resolve(),
        snapshots_path=args.snapshots.expanduser().resolve(),
        lane_rows_path=args.lane_rows.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
