#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable
from urllib import error, parse, request

from bench.long_knowledge_passage import parse_positive_passage_qrels


API_ROOT = "https://datasets-server.huggingface.co/filter"
SCHEMA_VERSION = "miracl-passage-gold.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch only positive MIRACL passage texts through the Dataset Viewer filter API."
    )
    parser.add_argument(
        "--qrels",
        action="append",
        required=True,
        metavar="LANGUAGE=PATH",
        help="Repeat for each language, for example zh=/path/qrels.tsv.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pause-seconds", type=float, default=0.05)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_language_paths(values: Iterable[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        language, separator, raw_path = value.partition("=")
        if not separator or not language.strip() or not raw_path.strip():
            raise ValueError(f"invalid --qrels value {value!r}")
        language = language.strip()
        if language in output:
            raise ValueError(f"duplicate qrels language {language}")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise ValueError(f"qrels file does not exist: {path}")
        output[language] = path
    return output


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_filter_predicate(docids: Iterable[str]) -> str:
    values = list(dict.fromkeys(str(value) for value in docids if str(value)))
    if not values:
        raise ValueError("docids must not be empty")
    return " OR ".join(f'"docid"={_sql_string(value)}' for value in values)


def fetch_batch(
    language: str,
    docids: list[str],
    *,
    timeout: float,
    retries: int,
) -> list[dict[str, Any]]:
    query = parse.urlencode(
        {
            "dataset": "miracl/miracl-corpus",
            "config": language,
            "split": "train",
            "where": build_filter_predicate(docids),
            "offset": 0,
            "length": min(100, len(docids)),
        }
    )
    url = f"{API_ROOT}?{query}"
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            req = request.Request(
                url,
                headers={"User-Agent": "rwkv-search-passage-benchmark/1.0"},
            )
            with request.urlopen(req, timeout=timeout) as response:
                payload = json.load(response)
            return [dict(item.get("row") or {}) for item in payload.get("rows", ())]
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < max(1, retries):
                time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"Dataset Viewer request failed after retries: {last_error}")


def main() -> int:
    args = parse_args()
    language_paths = parse_language_paths(args.qrels)
    doc_to_qids: dict[str, dict[str, set[str]]] = {}
    for language, path in language_paths.items():
        by_qid = parse_positive_passage_qrels(str(path))
        mapping: dict[str, set[str]] = defaultdict(set)
        for qid, docids in by_qid.items():
            for docid in docids:
                mapping[docid].add(qid)
        doc_to_qids[language] = dict(mapping)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    fetched: dict[str, int] = {}
    missing: dict[str, list[str]] = {}
    with temporary.open("w", encoding="utf-8") as handle:
        for language, mapping in sorted(doc_to_qids.items()):
            docids = sorted(mapping)
            found: set[str] = set()
            fetched_rows: list[dict[str, Any]] = []
            batch_size = max(1, min(80, int(args.batch_size)))
            batches = [
                docids[start : start + batch_size]
                for start in range(0, len(docids), batch_size)
            ]
            completed = 0
            with ThreadPoolExecutor(
                max_workers=max(1, min(16, int(args.workers)))
            ) as executor:
                futures = {
                    executor.submit(
                        fetch_batch,
                        language,
                        batch,
                        timeout=max(1.0, args.timeout),
                        retries=max(1, args.retries),
                    ): batch
                    for batch in batches
                }
                for future in as_completed(futures):
                    batch = futures[future]
                    fetched_rows.extend(future.result())
                    completed += len(batch)
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "language": language,
                                "completed": completed,
                                "total": len(docids),
                            }
                        ),
                        flush=True,
                    )
                    if args.pause_seconds > 0:
                        time.sleep(args.pause_seconds)
            for row in sorted(
                fetched_rows,
                key=lambda item: str(item.get("docid") or ""),
            ):
                    docid = str(row.get("docid") or "")
                    text = str(row.get("text") or "").strip()
                    if docid not in mapping or not text or "#" not in docid:
                        continue
                    page_id, passage_id = docid.split("#", 1)
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "language": language,
                        "docid": docid,
                        "page_id": page_id,
                        "passage_id": passage_id,
                        "title": str(row.get("title") or ""),
                        "text": text,
                        "source_qids": sorted(mapping[docid]),
                    }
                    handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    found.add(docid)
            fetched[language] = len(found)
            missing[language] = sorted(set(docids).difference(found))
    temporary.replace(output_path)

    manifest = {
        "schema_version": "miracl-passage-gold-manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "miracl/miracl-corpus",
        "source_revision": "Dataset Viewer current converted parquet",
        "source_license": "Apache-2.0",
        "api": API_ROOT,
        "qrels": {
            language: {
                "path": str(path),
                "sha256": sha256(path),
                "positive_passages": len(doc_to_qids[language]),
                "fetched_passages": fetched[language],
                "missing_passages": len(missing[language]),
                "missing_docids": missing[language],
            }
            for language, path in sorted(language_paths.items())
        },
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        },
        "publication_policy": "private benchmark input; contains upstream passage text",
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
