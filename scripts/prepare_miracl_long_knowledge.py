#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

from bench.long_knowledge_schema import LongKnowledgeCase, RelevantPage, write_cases


DATASET_REPOSITORY = "miracl/miracl"
DATASET_REVISION = "main"
DATASET_LICENSE = "Apache-2.0"
LANGUAGES = ("zh", "en")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the MIRACL zh/en dev qrels for FineWiki page retrieval.")
    parser.add_argument(
        "--base-url",
        default="https://huggingface.co/datasets/miracl/miracl/resolve/main",
    )
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": "rwkv-search-benchmark/1.0"})
    with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
    temporary.replace(destination)


def paths_for(language: str) -> tuple[str, str]:
    root = f"miracl-v1.0-{language}"
    topics = f"{root}/topics/topics.miracl-v1.0-{language}-dev.tsv"
    qrels = f"{root}/qrels/qrels.miracl-v1.0-{language}-dev.tsv"
    return topics, qrels


def parse_topics(path: Path) -> dict[str, str]:
    topics: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            qid, query = line.split("\t", 1)
        except ValueError as exc:
            raise ValueError(f"invalid topic line {line_number} in {path}") from exc
        if qid in topics:
            raise ValueError(f"duplicate topic qid {qid} in {path}")
        topics[qid] = query.strip()
    return topics


def parse_positive_qrels(path: Path) -> dict[str, dict[str, int]]:
    pages: dict[str, dict[str, int]] = defaultdict(dict)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"invalid qrels line {line_number} in {path}")
        qid, _, docid, raw_relevance = parts
        relevance = int(raw_relevance)
        if relevance <= 0:
            continue
        page_id = docid.split("#", 1)[0]
        pages[qid][page_id] = max(relevance, pages[qid].get(page_id, 0))
    return pages


def main() -> int:
    args = parse_args()
    root = Path(args.output_root)
    raw_root = root / "raw"
    all_cases: list[LongKnowledgeCase] = []
    upstream_files = []
    language_counts = {}

    for language in LANGUAGES:
        relative_topics, relative_qrels = paths_for(language)
        topic_path = raw_root / relative_topics
        qrels_path = raw_root / relative_qrels
        for relative, destination in ((relative_topics, topic_path), (relative_qrels, qrels_path)):
            if not destination.exists():
                download(f"{args.base_url.rstrip('/')}/{relative}", destination)
            upstream_files.append(
                {"path": str(destination.relative_to(root)), "bytes": destination.stat().st_size, "sha256": sha256(destination)}
            )
        topics = parse_topics(topic_path)
        relevant = parse_positive_qrels(qrels_path)
        cases = []
        for qid, query in topics.items():
            pages = relevant.get(qid) or {}
            if not pages:
                continue
            cases.append(
                LongKnowledgeCase(
                    id=f"miracl-{language}-dev-{qid}",
                    query=query,
                    language=language,
                    source_dataset="MIRACL",
                    source_split="dev",
                    source_qid=qid,
                    relevant_pages=tuple(
                        RelevantPage(page_id=page_id, relevance=relevance)
                        for page_id, relevance in sorted(pages.items())
                    ),
                )
            )
        language_counts[language] = len(cases)
        all_cases.extend(cases)

    output = root / "miracl_long_knowledge_dev_v1.jsonl"
    write_cases(output, all_cases)
    manifest = {
        "schema_version": "long-knowledge-dataset-manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_repository": DATASET_REPOSITORY,
        "source_revision": DATASET_REVISION,
        "source_license": DATASET_LICENSE,
        "source_url": "https://github.com/project-miracl/miracl",
        "split": "dev",
        "languages": list(LANGUAGES),
        "language_counts": language_counts,
        "cases": len(all_cases),
        "conversion": "positive passage qrels collapsed to stable Wikipedia page_id",
        "upstream_files": upstream_files,
        "output": {
            "path": output.name,
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
