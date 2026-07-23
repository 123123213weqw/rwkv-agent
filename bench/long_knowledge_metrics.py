from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Iterable, Mapping, Sequence

from .long_knowledge_schema import LongKnowledgeCase


DEFAULT_CUTOFFS = (1, 5, 10)


def score_case(
    case: LongKnowledgeCase,
    retrieved_page_ids: Sequence[str],
    *,
    latency_ms: float,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict:
    unique = list(dict.fromkeys(str(item) for item in retrieved_page_ids if str(item)))
    relevance = {item.page_id: item.relevance for item in case.relevant_pages}
    positive = set(relevance)
    output = {
        "id": case.id,
        "language": case.language,
        "source_dataset": case.source_dataset,
        "source_split": case.source_split,
        "source_qid": case.source_qid,
        "query_type": case.query_type,
        "relevant_page_count": len(positive),
        "retrieved_page_ids": unique,
        "empty": not unique,
        "latency_ms": float(latency_ms),
        "expectation": case.expectation,
        "retrieval_metrics_eligible": bool(positive),
        "missing_correct": 1.0 if case.expectation == "missing" and not unique else (
            0.0 if case.expectation == "missing" else None
        ),
    }
    if not positive:
        for cutoff in cutoffs:
            output[f"hit_at_{cutoff}"] = None
            output[f"recall_at_{cutoff}"] = None
        output["mrr_at_10"] = None
        output["ndcg_at_10"] = None
        return output
    first_rank = next((rank for rank, page_id in enumerate(unique[:10], start=1) if page_id in positive), None)
    output["mrr_at_10"] = 1.0 / first_rank if first_rank else 0.0
    for cutoff in cutoffs:
        found = positive.intersection(unique[:cutoff])
        output[f"hit_at_{cutoff}"] = 1.0 if found else 0.0
        output[f"recall_at_{cutoff}"] = len(found) / len(positive)
    gains = [relevance.get(page_id, 0) for page_id in unique[:10]]
    dcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sorted(relevance.values(), reverse=True)[:10]
    idcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    output["ndcg_at_10"] = dcg / idcg if idcg else 0.0
    return output


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[position])


def aggregate_scores(rows: Iterable[Mapping[str, object]]) -> dict:
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("cannot aggregate an empty result set")

    def summarize(items: Sequence[Mapping[str, object]]) -> dict:
        metrics = [
            *(f"hit_at_{cutoff}" for cutoff in DEFAULT_CUTOFFS),
            *(f"recall_at_{cutoff}" for cutoff in DEFAULT_CUTOFFS),
            "mrr_at_10",
            "ndcg_at_10",
        ]
        latencies = [float(item.get("latency_ms") or 0.0) for item in items]
        result = {
            "cases": len(items),
            "empty_rate": sum(bool(item.get("empty")) for item in items) / len(items),
            "latency_ms": {
                "mean": statistics.fmean(latencies),
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
        }
        result.update({name: statistics.fmean(float(item.get(name) or 0.0) for item in items) for name in metrics})
        return result

    positives = [row for row in records if bool(row.get("retrieval_metrics_eligible", True))]
    missing = [row for row in records if str(row.get("expectation") or "relevant") == "missing"]
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    query_type_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in positives:
        groups[str(row.get("language") or "unknown")].append(row)
        query_type_groups[str(row.get("query_type") or "unspecified")].append(row)
    result = {
        "cases_total": len(records),
        "positive_cases": len(positives),
        "expected_missing_cases": len(missing),
        "expected_missing_accuracy": (
            statistics.fmean(float(row.get("missing_correct") or 0.0) for row in missing)
            if missing else None
        ),
        "overall": summarize(positives) if positives else None,
        "by_language": {name: summarize(items) for name, items in sorted(groups.items())},
        "by_query_type": {
            name: summarize(items) for name, items in sorted(query_type_groups.items())
        },
    }
    eligible = [
        row for row in positives
        if bool(row.get("index_eligible", True))
    ]
    result["conditional_on_index_coverage"] = summarize(eligible) if eligible else None
    return result
