#!/usr/bin/env python3
"""Build a compact title-searchable local Wikipedia evidence index."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Sequence
import zlib


SCHEMA_VERSION = "rwkv-agent-wikipedia-title-index.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE pages (
            rowid INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            content_zlib BLOB NOT NULL
        );
        CREATE VIRTUAL TABLE titles USING fts5(
            title,
            content='pages',
            content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE UNIQUE INDEX pages_url ON pages(url);
        """
    )


def build(parquet_dir: Path, output: Path, manifest_path: Path) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    parquet_dir = parquet_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet files found under {parquet_dir}")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite Wikipedia index or manifest")
    output.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(output)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute("PRAGMA cache_size=-524288")
    create_schema(connection)
    started = time.monotonic()
    rowid = 0
    per_file: list[dict[str, Any]] = []
    try:
        for file_index, path in enumerate(files, 1):
            before = rowid
            source = parquet.ParquetFile(path)
            columns = set(source.schema.names)
            required = {"title", "url", "text"}
            if not required <= columns:
                raise ValueError(f"{path} is missing columns {sorted(required - columns)}")
            for batch in source.iter_batches(
                batch_size=2048,
                columns=["title", "url", "text"],
            ):
                values = batch.to_pydict()
                page_rows = []
                title_rows = []
                for title, url, text in zip(
                    values["title"], values["url"], values["text"], strict=True
                ):
                    title_value = " ".join(str(title or "").split())
                    url_value = str(url or "").strip()
                    if not title_value or not url_value:
                        continue
                    rowid += 1
                    compressed = zlib.compress(str(text or "").encode("utf-8"), 1)
                    page_rows.append((rowid, title_value, url_value, compressed))
                    title_rows.append((rowid, title_value))
                connection.executemany(
                    "INSERT OR IGNORE INTO pages(rowid,title,url,content_zlib) VALUES(?,?,?,?)",
                    page_rows,
                )
                connection.executemany(
                    "INSERT INTO titles(rowid,title) VALUES(?,?)",
                    title_rows,
                )
            connection.commit()
            per_file.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "rows": rowid - before,
                    "sha256": sha256(path),
                }
            )
            print(
                json.dumps(
                    {
                        "file": file_index,
                        "files": len(files),
                        "name": path.name,
                        "rows": rowid,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        connection.execute("INSERT INTO titles(titles) VALUES('optimize')")
        connection.commit()
    finally:
        connection.close()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "wikimedia/wikipedia:20231101.en/train parquet conversion",
        "parquet_dir": str(parquet_dir),
        "inputs": per_file,
        "rows": rowid,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256(output),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.parquet_dir, args.output, args.manifest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
