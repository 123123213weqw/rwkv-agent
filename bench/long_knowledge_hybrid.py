from __future__ import annotations

from collections import Counter, defaultdict
import math
import statistics
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .long_knowledge_schema import LongKnowledgeCase


CUTOFFS = (1, 5, 10, 50, 100)
STRATEGIES = ("lexical", "semantic", "hybrid")


class PairScorer(Protocol):
    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


def candidate_document(hit: Mapping[str, Any], *, max_chars: int = 2400) -> str:
    title = " ".join(str(hit.get("title") or "").split())
    headings = " > ".join(
        " ".join(str(value).split())
        for value in hit.get("headings", ())
        if str(value).strip()
    )
    body = " ".join(str(hit.get("text") or "").split())
    parts = [f"Title: {title}"]
    if headings:
        parts.append(f"Sections: {headings}")
    if body:
        parts.append(f"Passage: {body}")
    return "\n".join(parts)[: max(128, int(max_chars))]


def rank_candidates(
    candidates: Sequence[Mapping[str, Any]],
    semantic_scores: Sequence[float],
    *,
    rerank_depth: int = 50,
    rrf_k: float = 60.0,
    lexical_weight: float = 1.0,
    semantic_weight: float = 1.0,
) -> dict[str, list[dict[str, Any]]]:
    if len(candidates) != len(semantic_scores):
        raise ValueError("semantic score count does not match candidate count")
    raw = [dict(value) for value in candidates]
    depth = min(len(raw), max(1, int(rerank_depth)))
    head = raw[:depth]
    semantic_head = [
        item
        for _, _, item in sorted(
            (
                (-float(semantic_scores[index]), index, dict(item))
                for index, item in enumerate(head)
            ),
            key=lambda value: (value[0], value[1]),
        )
    ]
    semantic_rank = {
        str(item.get("page_id") or ""): rank
        for rank, item in enumerate(semantic_head, start=1)
    }
    lexical_rank = {
        str(item.get("page_id") or ""): rank
        for rank, item in enumerate(head, start=1)
    }
    hybrid_head = sorted(
        (dict(item) for item in head),
        key=lambda item: (
            -(
                max(0.0, float(lexical_weight))
                / (rrf_k + lexical_rank[str(item.get("page_id") or "")])
                + max(0.0, float(semantic_weight))
                / (rrf_k + semantic_rank[str(item.get("page_id") or "")])
            ),
            lexical_rank[str(item.get("page_id") or "")],
        ),
    )
    tail = [dict(value) for value in raw[depth:]]
    return {
        "lexical": raw,
        "semantic": semantic_head + tail,
        "hybrid": hybrid_head + tail,
    }


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Mapping[str, Any]]],
    *,
    weights: Sequence[float] | None = None,
    rrf_k: float = 60.0,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if not rankings:
        return []
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(rankings) != len(weights):
        raise ValueError("ranking and weight counts do not match")
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, tuple[int, int]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for channel, (ranking, weight) in enumerate(zip(rankings, weights)):
        for rank, item in enumerate(ranking, start=1):
            page_id = str(item.get("page_id") or "")
            if not page_id:
                continue
            scores[page_id] += max(0.0, float(weight)) / (rrf_k + rank)
            first_seen.setdefault(page_id, (channel, rank))
            documents.setdefault(page_id, dict(item))
    ordered = sorted(
        documents,
        key=lambda page_id: (
            -scores[page_id],
            first_seen[page_id],
            page_id,
        ),
    )
    output = []
    for page_id in ordered[: max(1, int(limit))]:
        item = documents[page_id]
        item["fusion_score"] = scores[page_id]
        output.append(item)
    return output


def evaluate_order(
    case: LongKnowledgeCase,
    candidates: Sequence[Mapping[str, Any]],
    *,
    index_eligible: bool,
) -> dict[str, Any]:
    page_ids = list(
        dict.fromkeys(
            str(item.get("page_id") or "")
            for item in candidates
            if str(item.get("page_id") or "")
        )
    )
    positives = {page.page_id for page in case.relevant_pages}
    if not positives:
        return {
            "retrieved_pages": len(page_ids),
            "first_relevant_rank": None,
            "failure_stage": "expected_missing",
            "missing_correct_if_empty": not page_ids,
            **{
                f"{kind}_at_{cutoff}": None
                for cutoff in CUTOFFS
                for kind in ("hit", "recall")
            },
        }
    first_rank = next(
        (
            rank
            for rank, page_id in enumerate(page_ids, start=1)
            if page_id in positives
        ),
        None,
    )
    if not index_eligible:
        failure_stage = "corpus_miss"
    elif first_rank is None or first_rank > 100:
        failure_stage = "candidate_recall_miss"
    elif first_rank > 10:
        failure_stage = "ranking_miss"
    else:
        failure_stage = "top10_hit"
    output: dict[str, Any] = {
        "retrieved_pages": len(page_ids),
        "first_relevant_rank": first_rank,
        "failure_stage": failure_stage,
        "missing_correct_if_empty": None,
    }
    for cutoff in CUTOFFS:
        found = positives.intersection(page_ids[:cutoff])
        output[f"hit_at_{cutoff}"] = 1.0 if found else 0.0
        output[f"recall_at_{cutoff}"] = len(found) / len(positives)
    return output


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(
        len(ordered) - 1,
        max(0, math.ceil(fraction * len(ordered)) - 1),
    )
    return float(ordered[position])


def _summarize_strategy(
    rows: Sequence[Mapping[str, Any]],
    strategy: str,
) -> dict[str, Any]:
    positives = [
        row
        for row in rows
        if str(row.get("expectation") or "relevant") != "missing"
    ]
    missing = [
        row
        for row in rows
        if str(row.get("expectation") or "relevant") == "missing"
    ]
    metrics = [row["strategies"][strategy] for row in positives]
    latencies = [
        float(row.get("latency_ms", {}).get(strategy) or 0.0)
        for row in rows
    ]
    output: dict[str, Any] = {
        "cases_total": len(rows),
        "positive_cases": len(positives),
        "expected_missing_cases": len(missing),
        "expected_missing_accuracy_if_empty": (
            statistics.fmean(
                float(row["strategies"][strategy]["missing_correct_if_empty"])
                for row in missing
            )
            if missing
            else None
        ),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "failure_stages": dict(
            sorted(Counter(item["failure_stage"] for item in metrics).items())
        ),
    }
    for cutoff in CUTOFFS:
        for kind in ("hit", "recall"):
            name = f"{kind}_at_{cutoff}"
            output[name] = (
                statistics.fmean(float(item[name]) for item in metrics)
                if metrics
                else None
            )
    return output


def summarize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    strategies: Sequence[str],
) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("cannot summarize an empty experiment")
    selected = tuple(dict.fromkeys(str(value) for value in strategies))
    if not selected:
        raise ValueError("at least one strategy is required")
    missing = [
        strategy
        for strategy in selected
        if any(strategy not in row.get("strategies", {}) for row in records)
    ]
    if missing:
        raise ValueError(f"strategies missing from rows: {sorted(set(missing))}")

    language_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    query_type_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        language_groups[str(row.get("language") or "unknown")].append(row)
        query_type_groups[str(row.get("query_type") or "unspecified")].append(row)
    return {
        "schema_version": "long-knowledge-hybrid-benchmark.v1",
        "record_count": len(records),
        "strategies": {
            strategy: {
                "overall": _summarize_strategy(records, strategy),
                "by_language": {
                    name: _summarize_strategy(items, strategy)
                    for name, items in sorted(language_groups.items())
                },
                "by_query_type": {
                    name: _summarize_strategy(items, strategy)
                    for name, items in sorted(query_type_groups.items())
                },
            }
            for strategy in selected
        },
    }
