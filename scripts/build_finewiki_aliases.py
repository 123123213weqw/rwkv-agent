#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import time
from typing import Dict, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from rwkv_search.finewiki import expand_chinese_script_aliases, extract_wikitext_aliases


_PARQUET_CACHE: Dict[str, pq.ParquetFile] = {}


def extract_group_worker(task: Tuple[str, int, str]) -> Tuple[list, list, int]:
    path, group_index, language = task
    parquet = _PARQUET_CACHE.get(path)
    if parquet is None:
        parquet = pq.ParquetFile(path)
        _PARQUET_CACHE[path] = parquet
    rows = parquet.read_row_group(
        group_index,
        columns=["page_id", "title", "wikitext", "text"],
        use_threads=False,
    ).to_pylist()
    page_ids = []
    aliases = []
    alias_count = 0
    for row in rows:
        title = row.get("title") or ""
        values = list(extract_wikitext_aliases(
            title,
            row.get("wikitext") or "",
            rendered_text=row.get("text") or "",
        ))
        if language == "zh":
            values = list(expand_chinese_script_aliases(title, values))
        page_ids.append(int(row["page_id"]))
        aliases.append(values)
        alias_count += len(values)
    return page_ids, aliases, alias_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build row-group-aligned FineWiki alias sidecars.")
    parser.add_argument("--data-root", default="/home/data/wangyue/datasets/finewiki/zhwiki")
    parser.add_argument("--output-root", default="/home/data/wangyue/datasets/finewiki/aliases-v1")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument(
        "--report",
        default="/home/data/wangyue/search-index/reports/finewiki_aliases_v1_build.json",
    )
    return parser.parse_args()


def valid_existing(source: pq.ParquetFile, output: Path) -> bool:
    if not output.exists():
        return False
    try:
        sidecar = pq.ParquetFile(output)
    except Exception:
        return False
    return (
        sidecar.metadata.num_rows == source.metadata.num_rows
        and sidecar.metadata.num_row_groups == source.metadata.num_row_groups
        and sidecar.schema_arrow.names == ["page_id", "aliases"]
    )


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(data_root.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"No Parquet files found under {data_root}")

    schema = pa.schema([("page_id", pa.int64()), ("aliases", pa.list_(pa.string()))])
    files = []
    total_rows = total_aliases = 0
    workers = max(1, int(args.workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for path in paths:
            source = pq.ParquetFile(path)
            output = output_root / f"{path.stem}.aliases.parquet"
            if not args.recreate and valid_existing(source, output):
                sidecar = pq.ParquetFile(output)
                rows = sidecar.metadata.num_rows
                files.append({"source": path.name, "output": output.name, "rows": rows, "reused": True})
                total_rows += rows
                print(json.dumps({"event": "reuse", "file": path.name, "rows": rows}), flush=True)
                continue

            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.unlink(missing_ok=True)
            writer = pq.ParquetWriter(temporary, schema, compression="zstd")
            file_rows = file_aliases = 0
            tasks = [
                (str(path), index, args.language)
                for index in range(source.metadata.num_row_groups)
            ]
            try:
                for group_index, (page_ids, aliases, alias_count) in enumerate(
                    executor.map(extract_group_worker, tasks, chunksize=1)
                ):
                    table = pa.Table.from_arrays(
                        [pa.array(page_ids, type=pa.int64()), pa.array(aliases, type=pa.list_(pa.string()))],
                        schema=schema,
                    )
                    writer.write_table(table, row_group_size=len(page_ids))
                    file_rows += len(page_ids)
                    file_aliases += alias_count
                    if (group_index + 1) % 25 == 0:
                        print(json.dumps({
                            "event": "progress", "file": path.name,
                            "row_groups": group_index + 1, "rows": file_rows,
                            "aliases": file_aliases,
                        }), flush=True)
            finally:
                writer.close()
            built = pq.ParquetFile(temporary)
            if (
                built.metadata.num_rows != source.metadata.num_rows
                or built.metadata.num_row_groups != source.metadata.num_row_groups
            ):
                raise RuntimeError(f"Sidecar alignment failed for {path.name}")
            temporary.replace(output)
            files.append({
                "source": path.name, "output": output.name, "rows": file_rows,
                "aliases": file_aliases, "bytes": output.stat().st_size, "reused": False,
            })
            total_rows += file_rows
            total_aliases += file_aliases
            print(json.dumps({"event": "file_complete", **files[-1]}), flush=True)

    report = {
        "data_root": str(data_root), "output_root": str(output_root),
        "language": args.language, "workers": workers,
        "rows": total_rows, "aliases": total_aliases, "files": files,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **report}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
