#!/usr/bin/env python3
"""Build a resumable title/evidence index from official Wikipedia XML shards."""

from __future__ import annotations

import argparse
import bz2
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterator, Sequence
from urllib.parse import quote
import xml.etree.ElementTree as ET
import zlib


SCHEMA_VERSION = "rwkv-agent-wikipedia-xml-title-index.v1"
COMMENT = re.compile(r"<!--.*?-->", re.S)
REF = re.compile(r"<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/\s*>", re.I | re.S)
TAG = re.compile(r"<[^>]+>")
TABLE = re.compile(r"\{\|.*?\|\}", re.S)
TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
FILE_LINK = re.compile(r"\[\[(?:File|Image):[^\]]+\]\]", re.I)
WIKI_LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
EXTERNAL_LINK = re.compile(r"\[(?:https?|ftp)://[^\s\]]+(?:\s+([^\]]+))?\]")
HEADING = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$", re.M)
WHITESPACE = re.compile(r"[ \t\f\v]+")
BLANKS = re.compile(r"\n{3,}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_wikitext(value: str, *, max_chars: int) -> str:
    text = COMMENT.sub(" ", str(value or ""))
    text = REF.sub(" ", text)
    text = TABLE.sub(" ", text)
    text = FILE_LINK.sub(" ", text)
    # Bounded repeated removal handles the common shallow nested templates
    # without risking a catastrophic recursive regular expression.
    for _ in range(8):
        cleaned = TEMPLATE.sub(" ", text)
        if cleaned == text:
            break
        text = cleaned
    text = WIKI_LINK.sub(lambda match: match.group(2) or match.group(1), text)
    text = EXTERNAL_LINK.sub(lambda match: match.group(1) or " ", text)
    text = HEADING.sub(lambda match: "\n" + match.group(1) + "\n", text)
    text = TAG.sub(" ", text)
    text = html.unescape(text).replace("'''", "").replace("''", "")
    lines = []
    for line in text.splitlines():
        line = WHITESPACE.sub(" ", line).strip(" *#:;|-\t")
        if line:
            lines.append(line)
    return BLANKS.sub("\n\n", "\n".join(lines)).strip()[:max_chars]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child(element: ET.Element, name: str) -> ET.Element | None:
    return next((item for item in element if local_name(item.tag) == name), None)


def iter_pages(path: Path, *, max_chars: int) -> Iterator[tuple[int, str, str, bytes]]:
    with bz2.open(path, "rb") as stream:
        for _event, page in ET.iterparse(stream, events=("end",)):
            if local_name(page.tag) != "page":
                continue
            try:
                title_node = child(page, "title")
                namespace_node = child(page, "ns")
                page_id_node = child(page, "id")
                revision = child(page, "revision")
                text_node = child(revision, "text") if revision is not None else None
                redirect = child(page, "redirect")
                if (
                    title_node is None
                    or namespace_node is None
                    or page_id_node is None
                    or text_node is None
                    or redirect is not None
                    or str(namespace_node.text or "") != "0"
                ):
                    continue
                title = " ".join(str(title_node.text or "").split())
                page_id = int(str(page_id_node.text or "0"))
                content = clean_wikitext(str(text_node.text or ""), max_chars=max_chars)
                if not title or page_id < 1 or not content:
                    continue
                url = "https://en.wikipedia.org/wiki/" + quote(
                    title.replace(" ", "_"), safe="()_-'"
                )
                yield page_id, title, url, zlib.compress(content.encode("utf-8"), 1)
            finally:
                page.clear()


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO metadata(key,value) VALUES('schema_version',
          'rwkv-agent-wikipedia-xml-title-index.v1');
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
        CREATE TABLE sources (
            name TEXT PRIMARY KEY,
            bytes INTEGER NOT NULL,
            sha1 TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            rows INTEGER NOT NULL,
            completed_at TEXT NOT NULL
        );
        """
    )
    connection.commit()


def expected_sources(dumpstatus_path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(dumpstatus_path.read_text(encoding="utf-8"))
    job = value.get("jobs", {}).get("articlesdump", {})
    if job.get("status") != "done":
        raise ValueError("dumpstatus articlesdump is not complete")
    return {
        str(name): {
            "size": int(item["size"]),
            "sha1": str(item["sha1"]),
            "url": str(item["url"]),
        }
        for name, item in job.get("files", {}).items()
    }


def build(
    xml_dir: Path,
    dumpstatus_path: Path,
    output: Path,
    manifest_path: Path,
    *,
    max_article_chars: int = 750_000,
    finalize: bool = True,
) -> dict[str, Any]:
    xml_dir = xml_dir.expanduser().resolve()
    dumpstatus_path = dumpstatus_path.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    expected = expected_sources(dumpstatus_path)
    expected_files = [xml_dir / name for name in sorted(expected)]
    missing = [path.name for path in expected_files if not path.is_file()]
    if missing and finalize:
        raise FileNotFoundError(f"missing {len(missing)} XML shards: {missing[:5]}")
    files = [path for path in expected_files if path.is_file()]
    if not files:
        raise FileNotFoundError("no complete XML shard is available yet")
    if manifest_path.exists():
        raise FileExistsError("refusing to overwrite completed Wikipedia manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    creating = not output.exists()
    connection = sqlite3.connect(output)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-524288")
    if creating:
        create_schema(connection)
    else:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if row != (SCHEMA_VERSION,):
            raise ValueError("existing index has an incompatible schema")

    started = time.monotonic()
    completed = {
        row[0]: {"bytes": row[1], "sha1": row[2], "sha256": row[3], "rows": row[4]}
        for row in connection.execute(
            "SELECT name,bytes,sha1,sha256,rows FROM sources"
        )
    }
    try:
        for file_index, path in enumerate(files, 1):
            metadata = expected[path.name]
            if path.stat().st_size != metadata["size"]:
                raise ValueError(f"size mismatch: {path.name}")
            previous = completed.get(path.name)
            if previous is not None:
                if previous["bytes"] != metadata["size"] or previous["sha1"] != metadata["sha1"]:
                    raise ValueError(f"resumed source metadata mismatch: {path.name}")
                print(json.dumps({"file": file_index, "name": path.name, "status": "skipped"}), flush=True)
                continue
            sha1 = file_hash(path, "sha1")
            if sha1 != metadata["sha1"]:
                raise ValueError(f"SHA-1 mismatch: {path.name}")
            sha256 = file_hash(path, "sha256")
            inserted = 0
            connection.execute("BEGIN IMMEDIATE")
            try:
                for page_id, title, url, content in iter_pages(
                    path, max_chars=max_article_chars
                ):
                    connection.execute(
                        "INSERT INTO pages(rowid,title,url,content_zlib) VALUES(?,?,?,?)",
                        (page_id, title, url, content),
                    )
                    connection.execute(
                        "INSERT INTO titles(rowid,title) VALUES(?,?)",
                        (page_id, title),
                    )
                    inserted += 1
                    if inserted % 100_000 == 0:
                        print(
                            json.dumps(
                                {
                                    "file": file_index,
                                    "name": path.name,
                                    "rows_in_file": inserted,
                                    "elapsed_seconds": round(time.monotonic() - started, 3),
                                }
                            ),
                            flush=True,
                        )
                connection.execute(
                    "INSERT INTO sources(name,bytes,sha1,sha256,rows,completed_at) VALUES(?,?,?,?,?,?)",
                    (path.name, metadata["size"], sha1, sha256, inserted, utc_now()),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            print(
                json.dumps(
                    {
                        "file": file_index,
                        "files": len(files),
                        "name": path.name,
                        "rows_in_file": inserted,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                ),
                flush=True,
            )
        source_rows = [
            {
                "name": row[0],
                "bytes": row[1],
                "sha1": row[2],
                "sha256": row[3],
                "rows": row[4],
            }
            for row in connection.execute(
                "SELECT name,bytes,sha1,sha256,rows FROM sources ORDER BY name"
            )
        ]
        page_count = int(connection.execute("SELECT count(*) FROM pages").fetchone()[0])
        if finalize:
            connection.execute("INSERT INTO titles(titles) VALUES('optimize')")
            connection.commit()
    finally:
        connection.close()

    if not finalize:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "partial",
            "sources_completed": len(source_rows),
            "sources_expected": len(expected),
            "rows": page_count,
            "output": str(output),
            "output_bytes": output.stat().st_size,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "source": "https://dumps.wikimedia.org/enwiki/20260701/ articlesdump",
        "dumpstatus": {
            "path": str(dumpstatus_path),
            "sha256": file_hash(dumpstatus_path, "sha256"),
        },
        "inputs": source_rows,
        "rows": page_count,
        "max_article_chars": max_article_chars,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": file_hash(output, "sha256"),
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
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--dumpstatus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-article-chars", type=int, default=750_000)
    parser.add_argument(
        "--ingest-available",
        action="store_true",
        help="transactionally ingest complete shards currently present without finalizing",
    )
    args = parser.parse_args(argv)
    manifest = build(
        args.xml_dir,
        args.dumpstatus,
        args.output,
        args.manifest,
        max_article_chars=max(10_000, args.max_article_chars),
        finalize=not args.ingest_available,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
