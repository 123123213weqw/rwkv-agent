#!/usr/bin/env python3
"""Summarize privacy-safe production Web Shadow JSONL telemetry."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "agent-web-shadow-metrics-summary.v1"
ROW_SCHEMA = "agent-web-shadow-metrics.v1"


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * percentile))
    return round(ordered[index], 3)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}") from exc
            if not isinstance(row, Mapping) or row.get("schema_version") != ROW_SCHEMA:
                raise ValueError(f"unexpected schema at line {line_number}")
            rows.append(dict(row))
    return rows


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    elapsed = [float(row.get("elapsed_ms") or 0.0) for row in values]
    statuses = Counter(str(row.get("status") or "unknown") for row in values)
    total = len(values)

    def arm(name: str) -> dict[str, Any]:
        items = [
            dict(row.get(name) or {})
            for row in values
            if isinstance(row.get(name), Mapping)
        ]
        fetches = sum(int(item.get("fetch_count") or 0) for item in items)
        fetch_successes = sum(
            int(item.get("fetch_success_count") or 0) for item in items
        )
        return {
            "mean_candidate_count": round(
                mean(int(item.get("candidate_count") or 0) for item in items),
                6,
            )
            if items
            else 0.0,
            "mean_result_count": round(
                mean(int(item.get("result_count") or 0) for item in items),
                6,
            )
            if items
            else 0.0,
            "fetch_count": fetches,
            "fetch_success_rate": round(fetch_successes / fetches, 6)
            if fetches
            else 0.0,
            "mean_latency_ms": round(
                mean(float(item.get("latency_ms") or 0.0) for item in items),
                3,
            )
            if items
            else 0.0,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "records": total,
        "statuses": dict(sorted(statuses.items())),
        "fallback_rate": round(
            sum(bool(row.get("fallback_used")) for row in values) / total,
            6,
        )
        if total
        else 0.0,
        "mean_legacy_evidence_count": round(
            mean(int(row.get("legacy_evidence_count") or 0) for row in values),
            6,
        )
        if values
        else 0.0,
        "mean_enhanced_evidence_count": round(
            mean(int(row.get("enhanced_evidence_count") or 0) for row in values),
            6,
        )
        if values
        else 0.0,
        "mean_evidence_url_overlap_count": round(
            mean(int(row.get("evidence_url_overlap_count") or 0) for row in values),
            6,
        )
        if values
        else 0.0,
        "shadow_elapsed_ms": {
            "mean": round(mean(elapsed), 3) if elapsed else 0.0,
            "p95": _percentile(elapsed, 0.95),
            "max": round(max(elapsed), 3) if elapsed else 0.0,
        },
        "legacy": arm("legacy"),
        "enhanced": arm("enhanced"),
        "privacy": {
            "contains_queries": False,
            "contains_urls": False,
            "contains_page_content": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    summary = summarize(load_rows(args.path.expanduser().resolve()))
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
