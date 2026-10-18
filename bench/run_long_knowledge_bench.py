from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from rwkv_search.candidate_index import CandidateIndexClient

from bench.long_knowledge_metrics import aggregate_scores, score_case
from bench.long_knowledge_schema import load_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a FineWiki candidate index with page-level qrels.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19220")
    parser.add_argument("--index", required=True)
    parser.add_argument("--language", default="", help="Optional case-language filter.")
    parser.add_argument("--channel-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases)
    if args.language:
        cases = [case for case in cases if case.language == args.language]
    if not cases:
        raise SystemExit("no cases remain after filtering")
    client = CandidateIndexClient(args.endpoint, timeout=30.0)
    all_relevant_page_ids = {
        page.page_id for case in cases for page in case.relevant_pages
    }
    existing_page_ids = client.existing_page_ids(args.index, sorted(all_relevant_page_ids))
    qrel_cases = [case for case in cases if case.relevant_pages]
    eligible_qrel_cases = sum(
        any(page.page_id in existing_page_ids for page in case.relevant_pages)
        for case in qrel_cases
    )
    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for position, case in enumerate(cases, start=1):
            analysis, hits, latency_ms = client.search(
                case.query,
                index=args.index,
                channel_size=max(10, args.channel_size),
                limit=max(1, args.limit),
            )
            row = score_case(
                case,
                [hit.page_id for hit in hits],
                latency_ms=latency_ms,
            )
            indexed_positives = sorted(
                page.page_id for page in case.relevant_pages
                if page.page_id in existing_page_ids
            )
            row["indexed_relevant_page_ids"] = indexed_positives
            row["index_eligible"] = bool(indexed_positives)
            row["query"] = case.query
            row["analysis"] = analysis.to_dict()
            row["hits"] = [hit.to_dict() for hit in hits]
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if position % 50 == 0:
                print(json.dumps({"event": "progress", "completed": position, "total": len(cases)}), flush=True)
    summary = {
        "schema_version": "long-knowledge-benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": str(Path(args.cases)),
        "endpoint": args.endpoint,
        "index": args.index,
        "language_filter": args.language or None,
        "channel_size": args.channel_size,
        "limit": args.limit,
        "qrel_index_coverage": {
            "relevant_pages": len(all_relevant_page_ids),
            "indexed_relevant_pages": len(existing_page_ids),
            "page_rate": (
                len(existing_page_ids) / len(all_relevant_page_ids)
                if all_relevant_page_ids else None
            ),
            "qrel_cases": len(qrel_cases),
            "eligible_cases": eligible_qrel_cases,
            "case_rate": (
                eligible_qrel_cases / len(qrel_cases) if qrel_cases else None
            ),
        },
        **aggregate_scores(rows),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
