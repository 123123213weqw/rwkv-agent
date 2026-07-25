from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence
from urllib import error, request

from rwkv_search.candidate_index import CandidateIndexClient

from bench.long_knowledge_hybrid import (
    candidate_document,
    evaluate_order,
    rank_candidates,
    reciprocal_rank_fusion,
    summarize_rows,
)
from bench.long_knowledge_schema import load_cases
from bench.run_long_knowledge_hybrid_bench import TransformersCrossEncoderScorer
from scripts.index_finewiki_page_embeddings import E5Encoder


STRATEGIES = ("lexical", "dense", "lexical_dense", "lexical_dense_rerank")


class DenseIndexClient:
    def __init__(self, endpoint: str, *, timeout: float = 45.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout)

    def search(
        self,
        index: str,
        query_vector: Sequence[float],
        *,
        limit: int,
        num_candidates: int,
    ) -> tuple[list[dict[str, Any]], float]:
        payload = {
            "size": max(1, int(limit)),
            "_source": ["page_id", "title", "text", "headings", "url", "language"],
            "knn": {
                "field": "embedding",
                "query_vector": [float(value) for value in query_vector],
                "k": max(1, int(limit)),
                "num_candidates": max(int(limit), int(num_candidates)),
            },
        }
        started = time.perf_counter()
        req = request.Request(
            f"{self.endpoint}/{index}/_search",
            method="POST",
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"dense search failed with HTTP {exc.code}: {detail}"
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        hits = []
        for hit in result.get("hits", {}).get("hits", ()):
            source = dict(hit.get("_source") or {})
            hits.append(
                {
                    "doc_id": str(source.get("page_id") or ""),
                    "page_id": str(source.get("page_id") or ""),
                    "title": str(source.get("title") or ""),
                    "text": str(source.get("text") or ""),
                    "headings": list(source.get("headings") or ()),
                    "url": str(source.get("url") or ""),
                    "language": str(source.get("language") or ""),
                    "score": float(hit.get("_score") or 0.0),
                    "channels": ["dense"],
                    "ranks": {"dense": len(hits) + 1},
                }
            )
        return hits, elapsed_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare isolated FineWiki lexical, dense, fusion, and fused-rerank retrieval."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19220")
    parser.add_argument("--lexical-index", required=True)
    parser.add_argument("--dense-index", required=True)
    parser.add_argument(
        "--lexical-run",
        default="",
        help="Optional frozen hybrid-run JSONL whose lexical hits are reused.",
    )
    parser.add_argument("--language", default="")
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--channel-size", type=int, default=100)
    parser.add_argument("--dense-num-candidates", type=int, default=1000)
    parser.add_argument("--rrf-lexical-weight", type=float, default=1.0)
    parser.add_argument("--rrf-dense-weight", type=float, default=1.0)
    parser.add_argument("--rerank-depth", type=int, default=50)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--reranker-model", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def _semantic_order(
    candidates: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    depth: int,
) -> list[dict[str, Any]]:
    return rank_candidates(
        candidates,
        scores,
        rerank_depth=depth,
    )["semantic"]


def _load_lexical_cache(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cache = {}
    for row in rows:
        case_id = str(row.get("id") or "")
        if not case_id or case_id in cache:
            raise ValueError("frozen lexical run has missing or duplicate case IDs")
        cache[case_id] = {
            "hits": list(row.get("hits") or ()),
            "search_elapsed_ms": float(row.get("search_elapsed_ms") or 0.0),
        }
    return cache


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases)
    if args.language:
        cases = [case for case in cases if case.language == args.language]
    if not cases:
        raise SystemExit("no cases remain after filtering")

    encoder = E5Encoder(
        args.embedding_model,
        device=args.device,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        fp16=args.fp16,
    )
    reranker = (
        TransformersCrossEncoderScorer(
            args.reranker_model,
            device=args.device,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
            fp16=args.fp16,
        )
        if args.reranker_model
        else None
    )
    strategies = STRATEGIES if reranker is not None else STRATEGIES[:-1]
    lexical_client = CandidateIndexClient(args.endpoint, timeout=45.0)
    dense_client = DenseIndexClient(args.endpoint, timeout=45.0)
    lexical_cache = _load_lexical_cache(args.lexical_run)
    if lexical_cache and set(lexical_cache) != {case.id for case in cases}:
        raise SystemExit(
            "frozen lexical run and filtered cases do not contain identical IDs"
        )
    relevant_page_ids = {
        page.page_id for case in cases for page in case.relevant_pages
    }
    existing_page_ids = lexical_client.existing_page_ids(
        args.lexical_index,
        sorted(relevant_page_ids),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output_path.open("w", encoding="utf-8") as handle:
        for position, case in enumerate(cases, start=1):
            if lexical_cache:
                cached = lexical_cache[case.id]
                lexical = list(cached["hits"])
                lexical_ms = float(cached["search_elapsed_ms"])
            else:
                _, lexical_hits, lexical_ms = lexical_client.search(
                    case.query,
                    index=args.lexical_index,
                    channel_size=max(10, args.channel_size),
                    limit=max(1, args.candidate_limit),
                )
                lexical = [hit.to_dict() for hit in lexical_hits]

            dense_started = time.perf_counter()
            query_vector = encoder.encode_queries([case.query])[0]
            dense, dense_search_ms = dense_client.search(
                args.dense_index,
                query_vector,
                limit=max(1, args.candidate_limit),
                num_candidates=max(1, args.dense_num_candidates),
            )
            dense_ms = (time.perf_counter() - dense_started) * 1000.0
            fused = reciprocal_rank_fusion(
                (lexical, dense),
                weights=(args.rrf_lexical_weight, args.rrf_dense_weight),
                limit=max(1, args.candidate_limit),
            )

            rerank_ms = 0.0
            rerank_scores: list[float] = [0.0] * len(fused)
            if reranker is not None and fused:
                depth = min(len(fused), max(1, args.rerank_depth))
                started = time.perf_counter()
                head_scores = reranker.score(
                    case.query,
                    [candidate_document(hit) for hit in fused[:depth]],
                )
                rerank_ms = (time.perf_counter() - started) * 1000.0
                rerank_scores[:depth] = head_scores
                fused_rerank = _semantic_order(
                    fused,
                    rerank_scores,
                    depth=args.rerank_depth,
                )
            else:
                fused_rerank = fused

            rankings = {
                "lexical": lexical,
                "dense": dense,
                "lexical_dense": fused,
                "lexical_dense_rerank": fused_rerank,
            }
            index_eligible = any(
                page.page_id in existing_page_ids
                for page in case.relevant_pages
            )
            sequential_base_ms = lexical_ms + dense_ms
            row = {
                "id": case.id,
                "query": case.query,
                "language": case.language,
                "query_type": case.query_type,
                "expectation": case.expectation,
                "index_eligible": index_eligible,
                "latency_ms": {
                    "lexical": lexical_ms,
                    "dense": dense_ms,
                    "lexical_dense": sequential_base_ms,
                    "lexical_dense_rerank": sequential_base_ms + rerank_ms,
                },
                "lexical_elapsed_ms": lexical_ms,
                "embedding_and_dense_elapsed_ms": dense_ms,
                "dense_search_elapsed_ms": dense_search_ms,
                "rerank_elapsed_ms": rerank_ms,
                "rerank_scores": rerank_scores[: max(1, args.rerank_depth)],
                "strategies": {
                    strategy: evaluate_order(
                        case,
                        rankings[strategy],
                        index_eligible=index_eligible,
                    )
                    for strategy in strategies
                },
                "lexical_hits": lexical,
                "dense_hits": dense,
                "fused_hits": fused,
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

    qrel_cases = [case for case in cases if case.relevant_pages]
    summary = {
        "schema_version": "long-knowledge-dense-run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": str(Path(args.cases)),
        "endpoint": args.endpoint,
        "lexical_index": args.lexical_index,
        "dense_index": args.dense_index,
        "lexical_run": args.lexical_run or None,
        "latency_composition": (
            "frozen_lexical_plus_current_dense_sequential_estimate"
            if args.lexical_run
            else "same_run_sequential"
        ),
        "language_filter": args.language or None,
        "candidate_limit": args.candidate_limit,
        "channel_size": args.channel_size,
        "dense_num_candidates": args.dense_num_candidates,
        "rrf_lexical_weight": args.rrf_lexical_weight,
        "rrf_dense_weight": args.rrf_dense_weight,
        "rerank_depth": args.rerank_depth,
        "embedding_model": args.embedding_model,
        "reranker_model": args.reranker_model or None,
        "qrel_index_coverage": {
            "relevant_pages": len(relevant_page_ids),
            "indexed_relevant_pages": len(existing_page_ids),
            "qrel_cases": len(qrel_cases),
            "eligible_cases": sum(
                any(page.page_id in existing_page_ids for page in case.relevant_pages)
                for case in qrel_cases
            ),
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
