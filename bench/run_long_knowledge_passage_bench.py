from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from bench.long_knowledge_passage import (
    PASSAGE_STRATEGIES,
    PagePassageClient,
    best_gold_overlap,
    best_gold_recall,
    load_gold_passages,
    reconstruct_final_page_order,
    select_passage_variants,
    summarize_passage_rows,
)
from bench.long_knowledge_schema import load_cases
from bench.run_long_knowledge_hybrid_bench import TransformersCrossEncoderScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate page-restricted passage hydration after frozen 5B Hybrid retrieval."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--dense-run", required=True)
    parser.add_argument("--gold-passages", required=True)
    parser.add_argument(
        "--case-ids",
        default="",
        help="Optional newline-delimited case IDs for a bounded pilot.",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:19220")
    parser.add_argument("--index", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--page-limit", type=int, default=8)
    parser.add_argument("--chunks-per-page", type=int, default=6)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def _load_dense_rows(path: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("id") or "")
            if not case_id or case_id in rows:
                raise ValueError(f"missing or duplicate dense row on line {line_number}")
            rows[case_id] = row
    return rows


def _public_passage(passage: dict, *, overlap: float | None = None) -> dict:
    output = {
        "doc_id": str(passage.get("doc_id") or ""),
        "page_id": str(passage.get("page_id") or ""),
        "chunk_id": int(passage.get("chunk_id", -1) or 0),
        "title": str(passage.get("title") or ""),
        "headings": list(passage.get("headings") or ()),
        "text": str(passage.get("text") or ""),
        "url": str(passage.get("url") or ""),
        "char_start": int(passage.get("char_start", 0) or 0),
        "lexical_score": float(passage.get("lexical_score") or 0.0),
        "cross_encoder_score": (
            float(passage["cross_encoder_score"])
            if passage.get("cross_encoder_score") is not None
            else None
        ),
    }
    if overlap is not None:
        output["gold_overlap"] = overlap
    return output


def main() -> int:
    args = parse_args()
    selected_case_ids = (
        {
            line.strip()
            for line in Path(args.case_ids).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if args.case_ids
        else set()
    )
    cases = [
        case
        for case in load_cases(args.cases)
        if case.language == args.language
        and (not selected_case_ids or case.id in selected_case_ids)
    ]
    if not cases:
        raise SystemExit("no matching cases")
    if selected_case_ids and selected_case_ids != {case.id for case in cases}:
        raise SystemExit("case ID filter contains unknown or wrong-language cases")
    dense_rows = _load_dense_rows(args.dense_run)
    missing_dense = {case.id for case in cases}.difference(dense_rows)
    if missing_dense:
        raise SystemExit(f"dense run is missing {len(missing_dense)} selected cases")
    gold = load_gold_passages(args.gold_passages)
    scorer = TransformersCrossEncoderScorer(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        fp16=args.fp16,
    )
    client = PagePassageClient(args.endpoint, timeout=45.0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output_path.open("w", encoding="utf-8") as handle:
        for position, case in enumerate(cases, start=1):
            page_order = reconstruct_final_page_order(dense_rows[case.id])
            selected_pages = [
                str(item.get("page_id") or "")
                for item in page_order[: max(1, args.page_limit)]
                if str(item.get("page_id") or "")
            ]
            relevant_pages = {page.page_id for page in case.relevant_pages}
            started = time.perf_counter()
            hydrated = (
                client.search_pages(
                    case.query,
                    index=args.index,
                    page_ids=selected_pages,
                    chunks_per_page=max(1, args.chunks_per_page),
                )
                if selected_pages
                else {}
            )
            variants_by_page = {
                page_id: select_passage_variants(
                    case.query,
                    hydrated.get(page_id, ()),
                    scorer,
                )
                for page_id in selected_pages
                if hydrated.get(page_id)
            }
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            case_gold = gold.get(case.source_qid, {})
            selected_passages: dict[str, list[dict]] = {
                strategy: [] for strategy in PASSAGE_STRATEGIES
            }
            strategy_metrics: dict[str, dict] = {}
            for strategy in PASSAGE_STRATEGIES:
                relevant_selected = []
                for page_id in selected_pages:
                    variants = variants_by_page.get(page_id, {})
                    passage = variants.get(strategy)
                    if not passage:
                        continue
                    page_gold = case_gold.get(page_id, ())
                    overlap = (
                        best_gold_overlap(passage, page_gold)
                        if page_id in relevant_pages and page_gold
                        else 0.0
                    )
                    recall = (
                        best_gold_recall(passage, page_gold)
                        if page_id in relevant_pages and page_gold
                        else 0.0
                    )
                    selected_passages[strategy].append(
                        _public_passage(passage, overlap=overlap)
                    )
                    if page_id in relevant_pages and page_gold:
                        relevant_selected.append(
                            {
                                "page_id": page_id,
                                "doc_id": str(passage.get("doc_id") or ""),
                                "gold_overlap": overlap,
                                "gold_recall": recall,
                            }
                        )
                strategy_metrics[strategy] = {
                    "case_best_gold_overlap": max(
                        (
                            float(item["gold_overlap"])
                            for item in relevant_selected
                        ),
                        default=0.0,
                    ),
                    "case_best_gold_recall": max(
                        (
                            float(item["gold_recall"])
                            for item in relevant_selected
                        ),
                        default=0.0,
                    ),
                    "relevant_selected_passages": relevant_selected,
                }
            row = {
                "schema_version": "long-knowledge-passage-case.v1",
                "id": case.id,
                "source_qid": case.source_qid,
                "query": case.query,
                "language": case.language,
                "page_limit": args.page_limit,
                "selected_page_ids": selected_pages,
                "relevant_page_ids": sorted(relevant_pages),
                "page_hit_at_limit": float(
                    bool(relevant_pages.intersection(selected_pages))
                ),
                "requested_pages": len(selected_pages),
                "returned_pages": len(variants_by_page),
                "hydration_elapsed_ms": elapsed_ms,
                "selected_passages": selected_passages,
                "strategy_metrics": strategy_metrics,
            }
            rows.append(row)
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            if position % 25 == 0:
                print(
                    json.dumps(
                        {"event": "progress", "completed": position, "total": len(cases)}
                    ),
                    flush=True,
                )

    summary = {
        "schema_version": "long-knowledge-passage-run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": str(Path(args.cases)),
        "dense_run": str(Path(args.dense_run)),
        "gold_passages": str(Path(args.gold_passages)),
        "endpoint": args.endpoint,
        "index": args.index,
        "language": args.language,
        "case_ids": str(Path(args.case_ids)) if args.case_ids else None,
        "page_limit": args.page_limit,
        "chunks_per_page": args.chunks_per_page,
        "model": args.model,
        **summarize_passage_rows(rows),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
