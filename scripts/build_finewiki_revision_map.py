#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the latest FineWiki revision per duplicate page.")
    parser.add_argument("--data-root", default="/home/data/wangyue/datasets/finewiki/zhwiki")
    parser.add_argument("--language", default="zh")
    parser.add_argument(
        "--output",
        default="/home/data/wangyue/datasets/finewiki/revisions-v1/duplicate-latest.parquet",
    )
    parser.add_argument(
        "--report",
        default="/home/data/wangyue/search-index/reports/finewiki_revisions_v1_build.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    root = Path(args.data_root)
    paths = sorted(root.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"No Parquet files found under {root}")

    # page_id -> [count, version, date_modified, source_file, row_group, row_index]
    state = {}
    rows = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for group_index in range(parquet.metadata.num_row_groups):
            values = parquet.read_row_group(
                group_index,
                columns=["page_id", "version", "date_modified"],
            ).to_pylist()
            for row_index, row in enumerate(values):
                rows += 1
                page_id = int(row["page_id"])
                version = int(row.get("version") or 0)
                modified = str(row.get("date_modified") or "")
                existing = state.get(page_id)
                candidate = (version, modified)
                if existing is None:
                    state[page_id] = [1, version, modified, path.name, group_index, row_index]
                else:
                    existing[0] += 1
                    if candidate > (existing[1], existing[2]):
                        existing[1:] = [version, modified, path.name, group_index, row_index]

    duplicates = []
    duplicate_rows = 0
    for page_id, value in state.items():
        count, version, modified, source_file, group_index, row_index = value
        if count <= 1:
            continue
        duplicate_rows += count - 1
        duplicates.append({
            "page_id": page_id,
            "revision_count": count,
            "latest_version": version,
            "date_modified": modified,
            "source_file": source_file,
            "row_group": group_index,
            "row_index": row_index,
        })
    duplicates.sort(key=lambda item: item["page_id"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(duplicates)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    if pq.ParquetFile(temporary).metadata.num_rows != len(duplicates):
        raise RuntimeError("Revision map row count mismatch")
    temporary.replace(output)

    report = {
        "data_root": str(root), "language": args.language,
        "output": str(output), "source_rows": rows,
        "unique_pages": len(state), "duplicate_pages": len(duplicates),
        "superseded_rows": duplicate_rows, "output_bytes": output.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
