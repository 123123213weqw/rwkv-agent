from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import fields
from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple
from urllib.parse import unquote, urlsplit

from rwkv_search.realtime.candidate_ranker import admit_candidates
from rwkv_search.realtime.types import DiscoveredURL

if __package__:
    from .retrieval_metrics import (
        classify_garbage_types,
        domain_matches,
        evaluate_candidate_stage,
        is_garbage_result,
        normalized_domain,
        target_page_matches,
    )
else:
    from retrieval_metrics import (  # type: ignore
        classify_garbage_types,
        domain_matches,
        evaluate_candidate_stage,
        is_garbage_result,
        normalized_domain,
        target_page_matches,
    )


SCHEMA_VERSION = "candidate-rerank-bench.v1"
STRATEGIES = ("raw", "admission", "semantic", "hybrid")


class CandidateScorer(Protocol):
    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


_DISCOVERED_FIELDS = {field.name for field in fields(DiscoveredURL)}


def discovered_from_mapping(value: Mapping[str, Any]) -> DiscoveredURL:
    kwargs = {name: value[name] for name in _DISCOVERED_FIELDS if name in value}
    return DiscoveredURL(**kwargs)


def discovered_to_mapping(value: DiscoveredURL) -> Dict[str, Any]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def candidate_document(value: Mapping[str, Any], *, max_chars: int = 1800) -> str:
    """Build a bounded page representation without fetching or benchmark labels."""

    url = str(value.get("url") or "")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    path = unquote(parsed.path or "/")
    source = f"{host}{path}"
    title = " ".join(str(value.get("title") or "").split())
    snippet = " ".join(str(value.get("snippet") or "").split())
    document = f"Title: {title}\nSource: {source}\nSummary: {snippet}".strip()
    return document[: max(64, int(max_chars))]


def _minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _domain_diversify(
    candidates: Sequence[Mapping[str, Any]], *, per_domain_limit: int = 3
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    overflow: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    limit = max(1, int(per_domain_limit))
    for item in candidates:
        value = dict(item)
        host = normalized_domain(str(value.get("url") or ""))
        if counts[host] >= limit:
            overflow.append(value)
            continue
        counts[host] += 1
        selected.append(value)
    selected.extend(overflow)
    return selected


def apply_admission(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    objects = [discovered_from_mapping(value) for value in candidates]
    result = admit_candidates(
        query,
        [query],
        objects,
        max_candidates=len(objects),
        per_domain_limit=3,
    )
    return (
        [discovered_to_mapping(value) for value in result.admitted],
        [discovered_to_mapping(value) for value in result.rejected],
    )


def rank_candidates(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    semantic_scores: Sequence[float],
    *,
    hybrid_semantic_weight: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """Return fixed ablations over exactly the same discovery candidates."""

    if len(candidates) != len(semantic_scores):
        raise ValueError("semantic score count does not match candidate count")
    raw = [dict(value) for value in candidates]
    score_by_url: Dict[str, List[float]] = defaultdict(list)
    for item, score in zip(raw, semantic_scores):
        score_by_url[str(item.get("url") or "")].append(float(score))

    admission, rejected = apply_admission(query, raw)
    semantic = [
        dict(item)
        for _, _, item in sorted(
            (
                (-float(score), index, dict(item))
                for index, (item, score) in enumerate(zip(raw, semantic_scores))
            ),
            key=lambda value: (value[0], value[1]),
        )
    ]

    semantic_for_admitted: List[float] = []
    for item in admission:
        values = score_by_url.get(str(item.get("url") or ""), [])
        semantic_for_admitted.append(values.pop(0) if values else float("-inf"))
    semantic_norm = _minmax(semantic_for_admitted)
    heuristic_norm = _minmax(
        [float(item.get("candidate_score") or 0.0) for item in admission]
    )
    weight = min(1.0, max(0.0, float(hybrid_semantic_weight)))
    hybrid_scored = []
    for index, (item, semantic_value, heuristic_value) in enumerate(
        zip(admission, semantic_norm, heuristic_norm)
    ):
        value = dict(item)
        value["semantic_score"] = semantic_for_admitted[index]
        value["hybrid_score"] = (
            weight * semantic_value + (1.0 - weight) * heuristic_value
        )
        hybrid_scored.append((value["hybrid_score"], -index, value))
    hybrid = [
        item
        for _, _, item in sorted(hybrid_scored, reverse=True)
    ]
    hybrid = _domain_diversify(hybrid, per_domain_limit=3)
    return {
        "raw": {"candidates": raw, "rejected": []},
        "admission": {"candidates": admission, "rejected": rejected},
        "semantic": {"candidates": semantic, "rejected": []},
        "hybrid": {"candidates": hybrid, "rejected": rejected},
    }


def _first_rank(
    candidates: Sequence[Mapping[str, Any]], predicate: Any
) -> int | None:
    for index, item in enumerate(candidates, 1):
        if predicate(item):
            return index
    return None


def evaluate_ranking(
    case: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    expected = [str(value) for value in case.get("expected_domains_any", ())]
    patterns = [str(value) for value in case.get("target_url_patterns_any", ())]
    domain_rank = _first_rank(
        candidates,
        lambda item: any(
            domain_matches(
                normalized_domain(str(item.get("url") or "")), domain
            )
            for domain in expected
        ),
    )
    target_rank = _first_rank(
        candidates,
        lambda item: target_page_matches(
            str(item.get("url") or ""), expected, patterns
        ),
    )
    top8 = list(candidates[:8])
    top8_garbage = sum(is_garbage_result(item, case) for item in top8)
    rejected_domain = sum(
        any(
            domain_matches(normalized_domain(str(item.get("url") or "")), domain)
            for domain in expected
        )
        for item in rejected
    )
    rejected_target = sum(
        target_page_matches(str(item.get("url") or ""), expected, patterns)
        for item in rejected
    )
    forbidden = {str(value) for value in case.get("forbidden_result_types", ())}
    rejected_useful_domain = 0
    for item in rejected:
        matches_expected = any(
            domain_matches(normalized_domain(str(item.get("url") or "")), domain)
            for domain in expected
        )
        rejection_types = set(str(value) for value in item.get("rejection_reasons", ()))
        observed_types = rejection_types | set(classify_garbage_types(item))
        if matches_expected and not forbidden.intersection(observed_types):
            rejected_useful_domain += 1
    return {
        **evaluate_candidate_stage(case, candidates),
        "domain_rank": domain_rank,
        "domain_mrr": round(1.0 / domain_rank, 6) if domain_rank else 0.0,
        "target_rank": target_rank,
        "target_mrr": round(1.0 / target_rank, 6) if target_rank else 0.0,
        "top8_count": len(top8),
        "top8_garbage_count": top8_garbage,
        "top8_garbage_rate": round(top8_garbage / max(1, len(top8)), 6),
        "garbage_type_counts": dict(
            sorted(
                Counter(
                    kind
                    for item in top8
                    for kind in classify_garbage_types(item)
                ).items()
            )
        ),
        "rejected_count": len(rejected),
        "rejected_expected_domain_count": rejected_domain,
        "rejected_useful_expected_domain_count": rejected_useful_domain,
        "rejected_target_page_count": rejected_target,
    }


def _aggregate_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"records": 0}
    metrics = [row["metrics"] for row in rows]

    def rate(name: str) -> float:
        return round(sum(bool(value[name]) for value in metrics) / len(metrics), 4)

    top8_count = sum(int(value["top8_count"]) for value in metrics)
    top8_garbage = sum(int(value["top8_garbage_count"]) for value in metrics)
    latency = [float(row.get("rerank_elapsed_ms") or 0.0) for row in rows]
    ordered_latency = sorted(latency)
    p95_index = max(0, math.ceil(0.95 * len(ordered_latency)) - 1)
    return {
        "records": len(rows),
        "domain_recall_at_5": rate("candidate_domain_hit_at_5"),
        "domain_recall_at_10": rate("candidate_domain_hit_at_10"),
        "target_page_recall_at_10": rate("candidate_target_page_hit_at_10"),
        "target_page_recall_at_20": rate("candidate_target_page_hit_at_20"),
        "domain_mrr": round(statistics.mean(value["domain_mrr"] for value in metrics), 6),
        "target_mrr": round(statistics.mean(value["target_mrr"] for value in metrics), 6),
        "top8_garbage_count": top8_garbage,
        "top8_garbage_rate": round(top8_garbage / max(1, top8_count), 4),
        "rejected_expected_domain_count": sum(
            int(value["rejected_expected_domain_count"]) for value in metrics
        ),
        "rejected_useful_expected_domain_count": sum(
            int(value["rejected_useful_expected_domain_count"]) for value in metrics
        ),
        "rejected_target_page_count": sum(
            int(value["rejected_target_page_count"]) for value in metrics
        ),
        "average_rerank_elapsed_ms": round(statistics.mean(latency), 3),
        "p95_rerank_elapsed_ms": round(ordered_latency[p95_index], 3),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(records),
        "strategies": {},
    }
    for strategy in STRATEGIES:
        selected = [row for row in records if row.get("strategy") == strategy]
        output["strategies"][strategy] = {
            "overall": _aggregate_group(selected),
            "languages": {
                language: _aggregate_group(
                    [row for row in selected if row.get("language") == language]
                )
                for language in ("zh", "en")
            },
        }
    return output


def public_case_matrix(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "repetition": row["repetition"],
            "language": row["language"],
            "category": row["category"],
            "strategy": row["strategy"],
            "rerank_elapsed_ms": row["rerank_elapsed_ms"],
            "metrics": row["metrics"],
        }
        for row in records
    ]
