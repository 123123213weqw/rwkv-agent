from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Sequence

from rwkv_search.candidate_index import CandidateIndexClient

from bench.long_knowledge_hybrid import (
    STRATEGIES,
    PairScorer,
    candidate_document,
    evaluate_order,
    rank_candidates,
    summarize_rows,
)
from bench.long_knowledge_schema import load_cases


class TransformersCrossEncoderScorer:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        batch_size: int,
        max_length: int,
        fp16: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "semantic strategies require torch and transformers"
            ) from exc
        self.model_name = model_name
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(64, int(max_length))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(self.device)
        if fp16:
            self.model = self.model.half()
        self.model.eval()

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        if not documents:
            return []
        values: list[float] = []
        pairs = [(query, document) for document in documents]
        with self.torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {
                    name: tensor.to(self.device)
                    for name, tensor in inputs.items()
                }
                logits = self.model(**inputs, return_dict=True).logits
                values.extend(float(value) for value in logits.view(-1).float().cpu())
        return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose and rerank FineWiki page candidates without changing production retrieval."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19220")
    parser.add_argument("--index", required=True)
    parser.add_argument("--language", default="")
    parser.add_argument("--channel-size", type=int, default=100)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--rerank-depth", type=int, default=50)
    parser.add_argument("--rrf-lexical-weight", type=float, default=1.0)
    parser.add_argument("--rrf-semantic-weight", type=float, default=1.0)
    parser.add_argument(
        "--strategies",
        default="lexical",
        help="Comma-separated subset of lexical,semantic,hybrid.",
    )
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def _strategies(value: str) -> tuple[str, ...]:
    selected = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    if not selected:
        raise SystemExit("at least one strategy is required")
    invalid = set(selected).difference(STRATEGIES)
    if invalid:
        raise SystemExit(f"unknown strategies: {sorted(invalid)}")
    if ("semantic" in selected or "hybrid" in selected) and "lexical" not in selected:
        selected = ("lexical", *selected)
    return selected


def main() -> int:
    args = parse_args()
    strategies = _strategies(args.strategies)
    cases = load_cases(args.cases)
    if args.language:
        cases = [case for case in cases if case.language == args.language]
    if not cases:
        raise SystemExit("no cases remain after filtering")

    scorer: PairScorer | None = None
    if "semantic" in strategies or "hybrid" in strategies:
        scorer = TransformersCrossEncoderScorer(
            args.model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            fp16=args.fp16,
        )

    client = CandidateIndexClient(args.endpoint, timeout=45.0)
    all_relevant_page_ids = {
        page.page_id for case in cases for page in case.relevant_pages
    }
    existing_page_ids = client.existing_page_ids(
        args.index,
        sorted(all_relevant_page_ids),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("w", encoding="utf-8") as handle:
        for position, case in enumerate(cases, start=1):
            _, hits, search_ms = client.search(
                case.query,
                index=args.index,
                channel_size=max(10, args.channel_size),
                limit=max(1, args.candidate_limit),
            )
            candidates = [hit.to_dict() for hit in hits]
            scores = [0.0] * len(candidates)
            rerank_ms = 0.0
            if scorer is not None and candidates:
                depth = min(len(candidates), max(1, args.rerank_depth))
                started = time.perf_counter()
                head_scores = scorer.score(
                    case.query,
                    [candidate_document(hit) for hit in candidates[:depth]],
                )
                rerank_ms = (time.perf_counter() - started) * 1000.0
                if len(head_scores) != depth:
                    raise RuntimeError(
                        f"scorer returned {len(head_scores)} scores for {depth} candidates"
                    )
                scores[:depth] = head_scores
            rankings = (
                rank_candidates(
                    candidates,
                    scores,
                    rerank_depth=args.rerank_depth,
                    lexical_weight=args.rrf_lexical_weight,
                    semantic_weight=args.rrf_semantic_weight,
                )
                if scorer is not None
                else {"lexical": candidates}
            )
            index_eligible = any(
                page.page_id in existing_page_ids
                for page in case.relevant_pages
            )
            row = {
                "id": case.id,
                "query": case.query,
                "language": case.language,
                "query_type": case.query_type,
                "expectation": case.expectation,
                "index_eligible": index_eligible,
                "latency_ms": {
                    strategy: search_ms + (rerank_ms if strategy != "lexical" else 0.0)
                    for strategy in strategies
                },
                "search_elapsed_ms": search_ms,
                "rerank_elapsed_ms": rerank_ms,
                "semantic_scores": scores[: min(len(scores), max(1, args.rerank_depth))],
                "strategies": {
                    strategy: evaluate_order(
                        case,
                        rankings[strategy],
                        index_eligible=index_eligible,
                    )
                    for strategy in strategies
                },
                "hits": candidates,
            }
            rows.append(row)
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            if position % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "completed": position,
                            "total": len(cases),
                        }
                    ),
                    flush=True,
                )

    qrel_cases = [case for case in cases if case.relevant_pages]
    eligible_cases = sum(
        any(page.page_id in existing_page_ids for page in case.relevant_pages)
        for case in qrel_cases
    )
    summary = {
        "schema_version": "long-knowledge-hybrid-run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": str(Path(args.cases)),
        "endpoint": args.endpoint,
        "index": args.index,
        "language_filter": args.language or None,
        "channel_size": args.channel_size,
        "candidate_limit": args.candidate_limit,
        "rerank_depth": args.rerank_depth,
        "rrf_lexical_weight": args.rrf_lexical_weight,
        "rrf_semantic_weight": args.rrf_semantic_weight,
        "model": scorer.model_name if scorer is not None else None,
        "qrel_index_coverage": {
            "relevant_pages": len(all_relevant_page_ids),
            "indexed_relevant_pages": len(existing_page_ids),
            "qrel_cases": len(qrel_cases),
            "eligible_cases": eligible_cases,
        },
        **summarize_rows(rows, strategies=strategies),
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
