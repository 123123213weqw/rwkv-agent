#!/usr/bin/env python3
"""Offline cross-encoder rerank benchmark over frozen discovery candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

if __package__:
    from .candidate_rerank import (
        SCHEMA_VERSION,
        STRATEGIES,
        CandidateScorer,
        candidate_document,
        evaluate_ranking,
        public_case_matrix,
        rank_candidates,
        summarize_records,
    )
    from .retrieval_schema import load_cases
else:
    from candidate_rerank import (  # type: ignore
        SCHEMA_VERSION,
        STRATEGIES,
        CandidateScorer,
        candidate_document,
        evaluate_ranking,
        public_case_matrix,
        rank_candidates,
        summarize_records,
    )
    from retrieval_schema import load_cases  # type: ignore


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


class TransformerCrossEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        batch_size: int,
        max_length: int,
        use_fp16: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(64, int(max_length))
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if use_fp16 and self.device.type == "cuda" else None
        kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **kwargs
        )
        self.model.eval().to(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        output: List[float] = []
        for offset in range(0, len(documents), self.batch_size):
            batch = documents[offset : offset + self.batch_size]
            pairs = [[query, document] for document in batch]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self.torch.inference_mode():
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            output.extend(float(value) for value in logits.cpu().tolist())
        return output

    def peak_memory_mb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return round(
            self.torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 3
        )


def run_record(
    case: Mapping[str, Any],
    record: Mapping[str, Any],
    scorer: CandidateScorer,
) -> List[Dict[str, Any]]:
    candidates = [dict(value) for value in record.get("candidates", ())]
    query = str(record.get("request", {}).get("query") or case["query"])
    documents = [candidate_document(value) for value in candidates]
    started = time.perf_counter()
    semantic_scores = list(scorer.score(query, documents))
    model_elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    ranking_started = time.perf_counter()
    strategies = rank_candidates(query, candidates, semantic_scores)
    ranking_elapsed_ms = round((time.perf_counter() - ranking_started) * 1000.0, 3)

    output: List[Dict[str, Any]] = []
    for strategy in STRATEGIES:
        value = strategies[strategy]
        elapsed = 0.0
        if strategy == "admission":
            elapsed = ranking_elapsed_ms
        elif strategy in {"semantic", "hybrid"}:
            elapsed = model_elapsed_ms + ranking_elapsed_ms
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": case["id"],
                "repetition": int(record.get("repetition") or 1),
                "language": case["language"],
                "category": case["category"],
                "strategy": strategy,
                "query": query,
                "model_name": scorer.model_name,
                "model_elapsed_ms": model_elapsed_ms,
                "ranking_elapsed_ms": ranking_elapsed_ms,
                "rerank_elapsed_ms": round(elapsed, 3),
                "metrics": evaluate_ranking(
                    case, value["candidates"], value["rejected"]
                ),
                "candidates": value["candidates"],
                "rejected": value["rejected"],
            }
        )
    return output


def run(args: argparse.Namespace) -> None:
    bench_path = Path(args.bench)
    input_path = Path(args.input)
    cases = {str(case["id"]): case for case in load_cases(bench_path)}
    input_rows = load_jsonl(input_path)
    if args.limit > 0:
        allowed = set(list(cases)[: args.limit])
        input_rows = [row for row in input_rows if str(row.get("id")) in allowed]
    unknown = sorted({str(row.get("id")) for row in input_rows} - set(cases))
    if unknown:
        raise ValueError("input contains unknown cases: " + ", ".join(unknown))
    if any(str(row.get("engine")) != "bing" for row in input_rows):
        raise ValueError("input must contain only frozen Bing records")

    load_started = time.perf_counter()
    scorer = TransformerCrossEncoder(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_fp16=args.fp16,
    )
    model_load_elapsed_ms = round((time.perf_counter() - load_started) * 1000.0, 3)
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(input_rows, 1):
        rows = run_record(cases[str(row["id"])], row, scorer)
        records.extend(rows)
        hybrid = next(value for value in rows if value["strategy"] == "hybrid")
        print(
            f"{index}/{len(input_rows)} id={row['id']} "
            f"repeat={row.get('repetition', 1)} "
            f"hybrid_domain5={int(hybrid['metrics']['candidate_domain_hit_at_5'])} "
            f"model_ms={hybrid['model_elapsed_ms']:.1f}",
            flush=True,
        )

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    public_path = Path(args.public_summary) if args.public_summary else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_records(records)
    summary.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": str(bench_path),
            "benchmark_sha256": sha256(bench_path),
            "input": str(input_path),
            "input_sha256": sha256(input_path),
            "input_record_count": len(input_rows),
            "case_count": len({str(row["id"]) for row in input_rows}),
            "repetitions": sorted(
                {int(row.get("repetition") or 1) for row in input_rows}
            ),
            "model": args.model,
            "device": args.device,
            "fp16": bool(args.fp16),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "model_load_elapsed_ms": model_load_elapsed_ms,
            "peak_gpu_memory_mb": scorer.peak_memory_mb(),
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if public_path:
        public_path.parent.mkdir(parents=True, exist_ok=True)
        public = dict(summary)
        public.pop("input", None)
        public["case_matrix"] = public_case_matrix(records)
        public_path.write_text(
            json.dumps(public, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary["strategies"], ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--bench", default="bench/realtime_web_retrieval.jsonl")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="bench/runs/candidate_rerank_v1.jsonl")
    parser.add_argument("--summary", default="bench/runs/candidate_rerank_v1_summary.json")
    parser.add_argument("--public-summary", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.max_length < 64:
        raise ValueError("max length must be at least 64")
    run(args)


if __name__ == "__main__":
    main()
