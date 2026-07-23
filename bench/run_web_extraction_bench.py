#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .web_extraction import (
        aggregate_results,
        capture_cases,
        load_snapshot_manifest,
        run_snapshot_benchmark,
    )
    from .web_extraction_schema import load_cases
else:
    from web_extraction import (
        aggregate_results,
        capture_cases,
        load_snapshot_manifest,
        run_snapshot_benchmark,
    )
    from web_extraction_schema import load_cases

DEFAULT_EXTRACTORS = (
    "current,hybrid_fast,trafilatura,justext,readability,resiliparse"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture and benchmark fixed webpage extraction snapshots")
    parser.add_argument("--cases", default="bench/web_extraction_cases.jsonl")
    parser.add_argument("--snapshot-dir", default="data/web-extraction-bench/snapshots")
    parser.add_argument("--manifest", default="data/web-extraction-bench/snapshots.jsonl")
    parser.add_argument("--output", default="bench/runs/web_extraction_v1.jsonl")
    parser.add_argument("--summary", default="bench/runs/web_extraction_v1_summary.json")
    parser.add_argument("--extractors", default=DEFAULT_EXTRACTORS)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--max-bytes", type=int, default=12 * 1024 * 1024)
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="repeat each fixed-snapshot extraction and report its median latency",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    if args.case_id:
        wanted = set(args.case_id)
        available = {str(case["id"]) for case in cases}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(f"unknown case ids: {', '.join(missing)}")
        cases = [case for case in cases if case["id"] in wanted]
    if args.limit > 0:
        cases = cases[: args.limit]

    snapshot_dir = Path(args.snapshot_dir)
    manifest_path = Path(args.manifest)
    if args.capture or args.capture_only:
        captured = asyncio.run(
            capture_cases(
                cases,
                snapshot_dir,
                manifest_path,
                concurrency=args.concurrency,
                timeout_seconds=args.timeout,
                max_bytes=args.max_bytes,
            )
        )
        for row in captured:
            print(
                f"{row['case_id']} fetch={row['fetch_outcome']} status={row['status']} "
                f"bytes={row['body_bytes']} elapsed={row['elapsed_ms']:.0f}ms",
                flush=True,
            )
        if args.capture_only:
            return
    if not manifest_path.exists():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")

    snapshots = load_snapshot_manifest(manifest_path)
    extractors = [item.strip() for item in args.extractors.split(",") if item.strip()]
    records = run_snapshot_benchmark(
        cases,
        snapshots,
        snapshot_dir,
        extractors,
        repetitions=args.repeat,
    )
    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = aggregate_results(cases, snapshots, records)
    summary.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cases": str(cases_path),
            "cases_sha256": sha256(cases_path),
            "manifest": str(manifest_path),
            "extractors_requested": extractors,
            "repetitions": args.repeat,
            "all_outputs_deterministic": all(
                bool(record["deterministic"]) for record in records
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
