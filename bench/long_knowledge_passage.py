from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import re
import statistics
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from rwkv_search.candidate_index import CandidateIndexClient
from rwkv_search.passage_hydration import (
    PagePassageClient as RuntimePagePassageClient,
)
from rwkv_search.passage_hydration import (
    build_page_passage_query as runtime_build_page_passage_query,
)
from rwkv_search.passage_hydration import combine_passages as runtime_combine_passages

from .long_knowledge_hybrid import PairScorer, candidate_document, rank_candidates


PASSAGE_STRATEGIES = ("lead", "lexical", "cross_encoder", "lead_plus_cross")
OVERLAP_THRESHOLDS = (0.3, 0.5)
_KEEP_TEXT = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)


@dataclass(frozen=True)
class GoldPassage:
    language: str
    docid: str
    page_id: str
    passage_id: str
    title: str
    text: str
    source_qids: tuple[str, ...]


def parse_positive_passage_qrels(path: str) -> dict[str, set[str]]:
    """Return positive MIRACL passage IDs grouped by source query ID."""

    output: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"invalid qrels line {line_number} in {path}")
            qid, _, docid, raw_relevance = parts
            if int(raw_relevance) > 0:
                output[qid].add(docid)
    return dict(output)


def load_gold_passages(path: str) -> dict[str, dict[str, list[GoldPassage]]]:
    """Return ``qid -> page_id -> gold passages`` from a private gold JSONL."""

    output: dict[str, dict[str, list[GoldPassage]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "miracl-passage-gold.v1":
                raise ValueError(f"invalid gold schema on line {line_number}")
            docid = str(row.get("docid") or "")
            page_id = str(row.get("page_id") or "")
            passage_id = str(row.get("passage_id") or "")
            text = str(row.get("text") or "").strip()
            qids = tuple(
                dict.fromkeys(str(value) for value in row.get("source_qids", ()) if value)
            )
            if not docid or not page_id or not passage_id or not text or not qids:
                raise ValueError(f"incomplete gold passage on line {line_number}")
            if docid in seen:
                raise ValueError(f"duplicate gold passage {docid}")
            seen.add(docid)
            passage = GoldPassage(
                language=str(row.get("language") or ""),
                docid=docid,
                page_id=page_id,
                passage_id=passage_id,
                title=str(row.get("title") or ""),
                text=text,
                source_qids=qids,
            )
            for qid in qids:
                output[qid][page_id].append(passage)
    return {
        qid: {page_id: list(passages) for page_id, passages in pages.items()}
        for qid, pages in output.items()
    }


def reconstruct_final_page_order(
    dense_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recreate the frozen Dense+RRF+CrossEncoder page order from a 5B row."""

    fused = [dict(value) for value in dense_row.get("fused_hits", ())]
    head_scores = [float(value) for value in dense_row.get("rerank_scores", ())]
    scores = head_scores + [0.0] * max(0, len(fused) - len(head_scores))
    if not fused:
        return []
    if not head_scores:
        return fused
    return rank_candidates(
        fused,
        scores,
        rerank_depth=len(head_scores),
    )["semantic"]


def build_page_passage_query(
    query: str,
    page_ids: Sequence[str],
    *,
    chunks_per_page: int,
) -> dict[str, Any]:
    """Build a generic query-scored passage lookup restricted to known pages."""
    return runtime_build_page_passage_query(
        query,
        page_ids,
        chunks_per_page=chunks_per_page,
    )
class PagePassageClient(CandidateIndexClient):
    """Benchmark-only page hydrator; it does not alter the production search API."""

    def search_pages(
        self,
        query: str,
        *,
        index: str,
        page_ids: Sequence[str],
        chunks_per_page: int = 6,
    ) -> dict[str, list[dict[str, Any]]]:
        return RuntimePagePassageClient.search_pages(
            self,
            query,
            index=index,
            page_ids=page_ids,
            chunks_per_page=chunks_per_page,
        )


def select_passage_variants(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    scorer: PairScorer,
) -> dict[str, dict[str, Any]]:
    if not candidates:
        return {}
    values = [dict(item) for item in candidates if str(item.get("text") or "").strip()]
    if not values:
        return {}
    lead = min(
        values,
        key=lambda item: (
            int(item.get("char_start", 0) or 0),
            int(item.get("chunk_id", 0) or 0),
        ),
    )
    lexical = max(
        values,
        key=lambda item: (
            float(item.get("lexical_score", 0.0) or 0.0),
            -int(item.get("char_start", 0) or 0),
        ),
    )
    semantic_scores = list(
        scorer.score(query, [candidate_document(item) for item in values])
    )
    if len(semantic_scores) != len(values):
        raise ValueError("passage scorer returned an unexpected score count")
    cross_position = max(
        range(len(values)),
        key=lambda position: (
            float(semantic_scores[position]),
            -int(values[position].get("char_start", 0) or 0),
        ),
    )
    cross = dict(values[cross_position])
    for position, score in enumerate(semantic_scores):
        values[position]["cross_encoder_score"] = float(score)
    cross["cross_encoder_score"] = float(semantic_scores[cross_position])
    combined = combine_passages(lead, cross)
    return {
        "lead": dict(lead),
        "lexical": dict(lexical),
        "cross_encoder": cross,
        "lead_plus_cross": combined,
    }


def combine_passages(
    lead: Mapping[str, Any],
    selected: Mapping[str, Any],
    *,
    max_chars: int = 3200,
) -> dict[str, Any]:
    """Keep the page lead and query-specific passage inside one Evidence budget."""
    return runtime_combine_passages(
        lead,
        selected,
        max_chars=max_chars,
    )


def normalized_character_ngrams(text: str, *, size: int = 3) -> set[str]:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = _KEEP_TEXT.sub("", value)
    if not value:
        return set()
    if len(value) < size:
        return {value}
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def character_ngram_f1(left: str, right: str, *, size: int = 3) -> float:
    left_ngrams = normalized_character_ngrams(left, size=size)
    right_ngrams = normalized_character_ngrams(right, size=size)
    if not left_ngrams or not right_ngrams:
        return 0.0
    overlap = len(left_ngrams.intersection(right_ngrams))
    precision = overlap / len(left_ngrams)
    recall = overlap / len(right_ngrams)
    return 2.0 * precision * recall / (precision + recall) if overlap else 0.0


def character_ngram_recall(evidence: str, gold: str, *, size: int = 3) -> float:
    evidence_ngrams = normalized_character_ngrams(evidence, size=size)
    gold_ngrams = normalized_character_ngrams(gold, size=size)
    if not evidence_ngrams or not gold_ngrams:
        return 0.0
    return len(evidence_ngrams.intersection(gold_ngrams)) / len(gold_ngrams)


def best_gold_overlap(
    passage: Mapping[str, Any],
    gold: Sequence[GoldPassage],
) -> float:
    text = str(passage.get("text") or "")
    return max(
        (character_ngram_f1(text, item.text) for item in gold),
        default=0.0,
    )


def best_gold_recall(
    passage: Mapping[str, Any],
    gold: Sequence[GoldPassage],
) -> float:
    text = str(passage.get("text") or "")
    return max(
        (character_ngram_recall(text, item.text) for item in gold),
        default=0.0,
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[position]


def summarize_passage_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("cannot summarize empty passage rows")
    output: dict[str, Any] = {
        "schema_version": "long-knowledge-passage-benchmark.v1",
        "record_count": len(records),
        "page_hit_at_limit": statistics.fmean(
            float(row.get("page_hit_at_limit") or 0.0) for row in records
        ),
        "hydration": {
            "requested_pages": sum(int(row.get("requested_pages") or 0) for row in records),
            "returned_pages": sum(int(row.get("returned_pages") or 0) for row in records),
            "mean_elapsed_ms": statistics.fmean(
                float(row.get("hydration_elapsed_ms") or 0.0) for row in records
            ),
            "p95_elapsed_ms": _percentile(
                [float(row.get("hydration_elapsed_ms") or 0.0) for row in records],
                0.95,
            ),
        },
        "strategies": {},
    }
    for strategy in PASSAGE_STRATEGIES:
        values = [
            float(row["strategy_metrics"][strategy]["case_best_gold_overlap"])
            for row in records
        ]
        conditional = [
            float(item["gold_overlap"])
            for row in records
            for item in row["strategy_metrics"][strategy]["relevant_selected_passages"]
        ]
        nonempty = [
            bool(item.get("text"))
            for row in records
            for item in row["selected_passages"][strategy]
        ]
        strategy_summary: dict[str, Any] = {
            "case_mean_best_gold_overlap": statistics.fmean(values) if values else 0.0,
            "conditional_relevant_passage_mean_overlap": (
                statistics.fmean(conditional) if conditional else 0.0
            ),
            "conditional_relevant_passages": len(conditional),
            "nonempty_rate": (
                statistics.fmean(float(value) for value in nonempty)
                if nonempty
                else 0.0
            ),
        }
        recalls = [
            float(row["strategy_metrics"][strategy]["case_best_gold_recall"])
            for row in records
        ]
        conditional_recalls = [
            float(item["gold_recall"])
            for row in records
            for item in row["strategy_metrics"][strategy]["relevant_selected_passages"]
        ]
        strategy_summary.update(
            {
                "case_mean_best_gold_recall": (
                    statistics.fmean(recalls) if recalls else 0.0
                ),
                "conditional_relevant_passage_mean_recall": (
                    statistics.fmean(conditional_recalls)
                    if conditional_recalls
                    else 0.0
                ),
            }
        )
        for threshold in OVERLAP_THRESHOLDS:
            suffix = str(threshold).replace(".", "_")
            strategy_summary[f"case_hit_at_overlap_{suffix}"] = statistics.fmean(
                float(value >= threshold) for value in values
            )
            strategy_summary[
                f"conditional_passage_hit_at_overlap_{suffix}"
            ] = (
                statistics.fmean(float(value >= threshold) for value in conditional)
                if conditional
                else 0.0
            )
        for threshold in (0.5, 0.8):
            suffix = str(threshold).replace(".", "_")
            strategy_summary[f"case_hit_at_recall_{suffix}"] = statistics.fmean(
                float(value >= threshold) for value in recalls
            )
            strategy_summary[
                f"conditional_passage_hit_at_recall_{suffix}"
            ] = (
                statistics.fmean(
                    float(value >= threshold) for value in conditional_recalls
                )
                if conditional_recalls
                else 0.0
            )
        output["strategies"][strategy] = strategy_summary
    return output
