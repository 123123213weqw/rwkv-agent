#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Dict, Iterator, List, Sequence, Set

import pyarrow.parquet as pq

from rwkv_search.analysis import DocumentAnalyzer
from rwkv_search.candidate_index import CandidateIndexClient, chunk_to_index_document
from rwkv_search.finewiki import (
    FineWikiArticle,
    FineWikiChunker,
    clean_finewiki_markdown,
    expand_chinese_script_aliases,
    extract_wikitext_aliases,
)


DEFAULT_INDEX = "rwkv-finewiki-zh-candidate-v1"
SOURCE_COLUMNS = [
    "page_id", "url", "title", "text", "date_modified", "wikidata_id",
    "wikiname", "version", "infoboxes", "has_math",
]

_WORKER_ANALYZER = None


def initialize_analysis_worker() -> None:
    global _WORKER_ANALYZER
    _WORKER_ANALYZER = DocumentAnalyzer()


def analyze_chunk_worker(chunk) -> dict:
    global _WORKER_ANALYZER
    if _WORKER_ANALYZER is None:
        initialize_analysis_worker()
    return chunk_to_index_document(chunk, _WORKER_ANALYZER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an isolated FineWiki candidate index.")
    parser.add_argument("--data-root", default="/home/data/wangyue/datasets/finewiki/zhwiki")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--wikiname", default="", help="Defaults to <language>wiki.")
    parser.add_argument("--snapshot-date", default="20250801")
    parser.add_argument(
        "--forced-title", action="append", default=[],
        help="Title that must be included in a sampled build; may be repeated.",
    )
    parser.add_argument(
        "--aliases-root",
        default="/home/data/wangyue/datasets/finewiki/aliases-v1",
        help="Row-group-aligned aliases; pass an empty value to extract aliases inline.",
    )
    parser.add_argument(
        "--revision-map",
        default="/home/data/wangyue/datasets/finewiki/revisions-v1/duplicate-latest.parquet",
        help="Latest-version selection for pages with duplicate revisions.",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:19220")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--limit", type=int, default=50_000, help="Articles; use 0 for the full corpus.")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--batch-docs", type=int, default=400)
    parser.add_argument("--analysis-batch", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--target-chars", type=int, default=0, help="0 selects a language-aware default.")
    parser.add_argument("--max-chars", type=int, default=0, help="0 selects a language-aware default.")
    parser.add_argument("--overlap-chars", type=int, default=-1, help="-1 selects a language-aware default.")
    parser.add_argument("--bench-targets", type=int, default=500)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument(
        "--report",
        default="/home/data/wangyue/search-index/reports/finewiki_candidate_v1_ingest.json",
    )
    parser.add_argument(
        "--targets-output",
        default="/home/data/wangyue/search-index/reports/finewiki_candidate_v1_targets.jsonl",
    )
    return parser.parse_args()


def article_from_row(
    row: dict,
    aliases=None,
    *,
    language: str = "zh",
    wikiname: str = "",
) -> FineWikiArticle:
    def text(value, default: str = "") -> str:
        return default if value is None else str(value)

    title = text(row.get("title"))
    if aliases is None:
        extracted = extract_wikitext_aliases(
            title,
            text(row.get("wikitext")),
            rendered_text=text(row.get("text")),
        )
        aliases = (
            expand_chinese_script_aliases(title, extracted)
            if language == "zh"
            else extracted
        )
    return FineWikiArticle(
        page_id=text(row.get("page_id")),
        url=text(row.get("url")),
        title=title,
        text=text(row.get("text")),
        date_modified=text(row.get("date_modified")),
        wikidata_id=text(row.get("wikidata_id")),
        wikiname=text(row.get("wikiname")) or wikiname or f"{language}wiki",
        source_version=int(row.get("version", 0) or 0),
        infoboxes=text(row.get("infoboxes")),
        has_math=bool(row.get("has_math", False)),
        aliases=tuple(aliases),
    )


def alias_path_for(source: Path, aliases_root: Path | None) -> Path | None:
    return aliases_root / f"{source.stem}.aliases.parquet" if aliases_root else None


def read_rows_with_aliases(
    parquet: pq.ParquetFile,
    group_index: int,
    *,
    alias_parquet: pq.ParquetFile | None,
) -> List[dict]:
    columns = SOURCE_COLUMNS if alias_parquet else [*SOURCE_COLUMNS, "wikitext"]
    rows = parquet.read_row_group(group_index, columns=columns).to_pylist()
    if alias_parquet is None:
        return rows
    alias_rows = alias_parquet.read_row_group(
        group_index, columns=["page_id", "aliases"]
    ).to_pylist()
    if len(rows) != len(alias_rows):
        raise RuntimeError(f"Alias row count mismatch in row group {group_index}")
    for row, alias_row in zip(rows, alias_rows):
        if int(row["page_id"]) != int(alias_row["page_id"]):
            raise RuntimeError(f"Alias page_id mismatch in row group {group_index}")
        row["_aliases"] = alias_row.get("aliases") or []
    return rows


def find_forced_articles(
    paths: Sequence[Path],
    wanted_titles: Set[str],
    aliases_root: Path | None,
    latest_versions: Dict[str, int],
    *,
    language: str,
    wikiname: str,
) -> Dict[str, FineWikiArticle]:
    """Scan only the small title column, then read full rows from matching row groups."""

    found: Dict[str, FineWikiArticle] = {}
    if not wanted_titles:
        return found
    for path in paths:
        parquet = pq.ParquetFile(path)
        alias_path = alias_path_for(path, aliases_root)
        alias_parquet = pq.ParquetFile(alias_path) if alias_path else None
        for group_index in range(parquet.metadata.num_row_groups):
            if len(found) == len(wanted_titles):
                return found
            summaries = parquet.read_row_group(
                group_index, columns=["page_id", "title", "version"]
            ).to_pylist()
            wanted_indices = [
                index for index, row in enumerate(summaries)
                if row["title"] in wanted_titles
                and row["title"] not in found
                and (
                    str(row["page_id"]) not in latest_versions
                    or int(row.get("version") or 0) == latest_versions[str(row["page_id"])]
                )
            ]
            if not wanted_indices:
                continue
            rows = read_rows_with_aliases(parquet, group_index, alias_parquet=alias_parquet)
            for index in wanted_indices:
                row = rows[index]
                article = article_from_row(
                    row,
                    row.get("_aliases"),
                    language=language,
                    wikiname=wikiname,
                )
                found[article.title] = article
    return found


def _selected_groups(parquet: pq.ParquetFile, quota: int, rng: random.Random) -> List[int]:
    groups = parquet.metadata.num_row_groups
    if quota <= 0:
        return []
    mean_rows = max(1, math.ceil(parquet.metadata.num_rows / groups))
    required = min(groups, math.ceil(quota / mean_rows) + 1)
    return sorted(rng.sample(range(groups), required))


def iter_articles(
    paths: Sequence[Path],
    *,
    limit: int,
    seed: int,
    forced: Sequence[FineWikiArticle],
    aliases_root: Path | None,
    latest_versions: Dict[str, int],
    language: str,
    wikiname: str,
) -> Iterator[FineWikiArticle]:
    seen = {article.page_id for article in forced}
    emitted = 0
    for article in forced:
        yield article
        emitted += 1

    if limit > 0 and emitted >= limit:
        return
    remaining_total = max(0, limit - emitted) if limit > 0 else 0
    base, extra = divmod(remaining_total, len(paths)) if limit > 0 else (0, 0)

    for file_index, path in enumerate(paths):
        parquet = pq.ParquetFile(path)
        alias_path = alias_path_for(path, aliases_root)
        alias_parquet = pq.ParquetFile(alias_path) if alias_path else None
        quota = base + (1 if file_index < extra else 0) if limit > 0 else parquet.metadata.num_rows
        if quota <= 0:
            continue
        rng = random.Random(seed + file_index * 1_000_003)
        groups = (
            _selected_groups(parquet, quota + len(forced), rng)
            if limit > 0 else range(parquet.metadata.num_row_groups)
        )
        file_emitted = 0
        for group_index in groups:
            rows = read_rows_with_aliases(parquet, group_index, alias_parquet=alias_parquet)
            if limit > 0:
                rng.shuffle(rows)
            for row in rows:
                article = article_from_row(
                    row,
                    row.get("_aliases"),
                    language=language,
                    wikiname=wikiname,
                )
                latest = latest_versions.get(article.page_id)
                if latest is not None and article.source_version != latest:
                    continue
                if not article.page_id or article.page_id in seen:
                    continue
                seen.add(article.page_id)
                yield article
                emitted += 1
                file_emitted += 1
                if limit > 0 and (emitted >= limit or file_emitted >= quota):
                    break
            if limit > 0 and (emitted >= limit or file_emitted >= quota):
                break
        if limit > 0 and emitted >= limit:
            return


def reservoir_add(items: List[dict], candidate: dict, *, seen: int, limit: int, rng: random.Random) -> None:
    if len(items) < limit:
        items.append(candidate)
        return
    position = rng.randrange(seen)
    if position < limit:
        items[position] = candidate


def main() -> int:
    args = parse_args()
    language = args.language.strip().casefold()
    if not language:
        raise SystemExit("--language must not be empty")
    wikiname = args.wikiname.strip().casefold() or f"{language}wiki"
    started = time.perf_counter()
    root = Path(args.data_root)
    paths = sorted(root.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"No Parquet files found under {root}")

    aliases_root = Path(args.aliases_root) if args.aliases_root else None
    if aliases_root:
        missing_aliases = [str(alias_path_for(path, aliases_root)) for path in paths if not alias_path_for(path, aliases_root).exists()]
        if missing_aliases:
            raise SystemExit(f"Missing alias sidecars: {missing_aliases}")
    revision_map = Path(args.revision_map) if args.revision_map else None
    if revision_map and not revision_map.exists():
        raise SystemExit(f"Missing revision map: {revision_map}")
    latest_versions = {
        str(row["page_id"]): int(row["latest_version"])
        for row in pq.read_table(revision_map, columns=["page_id", "latest_version"]).to_pylist()
    } if revision_map else {}
    forced_titles = set(args.forced_title)
    forced = find_forced_articles(
        paths, forced_titles, aliases_root, latest_versions,
        language=language, wikiname=wikiname,
    )
    print(json.dumps({
        "event": "forced_titles",
        "found": sorted(forced),
        "missing": sorted(forced_titles - set(forced)),
    }, ensure_ascii=False), flush=True)

    client = CandidateIndexClient(args.endpoint, timeout=90.0)
    print(json.dumps({"event": "cluster_health", "health": client.health()}), flush=True)
    client.create_index(args.index, recreate=args.recreate, shards=args.shards)
    client.set_refresh_interval(args.index, "-1")

    target_chars = args.target_chars or (700 if language == "zh" else 1800)
    max_chars = args.max_chars or (900 if language == "zh" else 2400)
    overlap_chars = args.overlap_chars if args.overlap_chars >= 0 else (100 if language == "zh" else 200)
    chunker = FineWikiChunker(
        target_chars=target_chars,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    pending_chunks: List = []
    articles = chunks = skipped = indexed = total_text_chars = 0
    page_types: Counter[str] = Counter()
    chunk_counts: List[int] = []
    targets: List[dict] = []
    target_seen = 0
    target_rng = random.Random(args.seed + 991)
    workers = max(1, int(args.workers))
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_analysis_worker,
    ) if workers > 1 else None

    def index_pending() -> int:
        if not pending_chunks:
            return 0
        documents = (
            map(analyze_chunk_worker, pending_chunks)
            if executor is None else executor.map(analyze_chunk_worker, pending_chunks, chunksize=16)
        )
        written = 0
        bulk_batch: List[dict] = []
        for document in documents:
            bulk_batch.append(document)
            if len(bulk_batch) >= args.batch_docs:
                written += client.bulk(args.index, bulk_batch)
                bulk_batch.clear()
        if bulk_batch:
            written += client.bulk(args.index, bulk_batch)
        pending_chunks.clear()
        return written

    try:
        source = iter_articles(
            paths, limit=args.limit, seed=args.seed,
            forced=list(forced.values()), aliases_root=aliases_root,
            latest_versions=latest_versions, language=language, wikiname=wikiname,
        )
        for article in source:
            articles += 1
            article_chunks = chunker.chunk(article, snapshot_date=args.snapshot_date)
            if not article_chunks:
                skipped += 1
                continue
            page_types[article_chunks[0].page_type] += 1
            chunk_counts.append(len(article_chunks))
            total_text_chars += sum(len(chunk.text) for chunk in article_chunks)

            cleaned_text = clean_finewiki_markdown(article.title, article.text)
            if 2 <= len(article.title) <= 40 and len(cleaned_text) >= 100:
                target_seen += 1
                reservoir_add(
                    targets,
                    {
                        "page_id": article.page_id,
                        "title": article.title,
                        "query_exact": article.title,
                        "query_definition": (
                            f"什么是{article.title}"
                            if language == "zh"
                            else f"What is {article.title}?"
                        ),
                    },
                    seen=target_seen,
                    limit=args.bench_targets,
                    rng=target_rng,
                )

            pending_chunks.extend(article_chunks)
            chunks += len(article_chunks)
            if len(pending_chunks) >= args.analysis_batch:
                indexed += index_pending()
            if articles % 5000 == 0:
                elapsed = time.perf_counter() - started
                print(json.dumps({
                    "event": "progress", "articles": articles, "chunks": chunks,
                    "indexed": indexed, "articles_per_second": round(articles / elapsed, 2),
                    "elapsed_seconds": round(elapsed, 2),
                }), flush=True)
        indexed += index_pending()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        client.set_refresh_interval(args.index, "1s")
        client.refresh(args.index)

    count = client.count(args.index)
    client.flush(args.index)
    stats = client.stats(args.index)
    primary = stats["_all"]["primaries"]
    if count != indexed:
        raise RuntimeError(f"Index count mismatch: bulk indexed={indexed}, index count={count}")

    targets_path = Path(args.targets_output)
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    with targets_path.open("w", encoding="utf-8") as handle:
        for item in targets:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "dataset": str(root), "aliases_root": str(aliases_root) if aliases_root else "inline",
        "revision_map": str(revision_map) if revision_map else "first-seen",
        "snapshot_date": args.snapshot_date, "language": language,
        "wikiname": wikiname, "endpoint": args.endpoint,
        "chunking": {
            "target_chars": target_chars,
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
        },
        "index": args.index, "seed": args.seed, "analysis_workers": workers,
        "article_limit": args.limit or "full", "articles_processed": articles,
        "articles_skipped": skipped, "page_types": dict(page_types), "chunks_indexed": indexed,
        "chunks_per_article": {
            "mean": statistics.fmean(chunk_counts) if chunk_counts else 0.0,
            "median": statistics.median(chunk_counts) if chunk_counts else 0.0,
            "max": max(chunk_counts, default=0),
        },
        "indexed_text_chars": total_text_chars,
        "index_store_bytes": int(primary["store"]["size_in_bytes"]),
        "bench_targets": len(targets), "targets_output": str(targets_path),
        "forced_titles_found": sorted(forced),
        "forced_titles_missing": sorted(forced_titles - set(forced)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **report}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
