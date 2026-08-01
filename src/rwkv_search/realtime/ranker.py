from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlsplit

from ..pipeline.reranker import RetrievalReranker
from ..pipeline.query_compiler import normalize_source_preference
from ..semantic_selection import PairScorer
from ..search import SearchResult
from ..text import best_snippet, search_tokens
from .types import RealtimeDocument


def rank_documents(
    query: str,
    documents: Sequence[RealtimeDocument],
    *,
    freshness_mode: str,
    limit: int,
    per_domain_limit: int = 4,
    scorer: PairScorer | None = None,
    query_views: Sequence[str] = (),
    source_preference: str = "any",
) -> List[RealtimeDocument]:
    normalized_preference = normalize_source_preference(source_preference)
    query_tokens = set(search_tokens(query))
    now = datetime.now(timezone.utc)
    for document in documents:
        title_tokens = set(search_tokens(document.title))
        body_tokens = set(search_tokens(document.text[:100000]))
        title_overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
        body_overlap = len(query_tokens & body_tokens) / max(1, len(query_tokens))
        document.relevance = min(1.0, 0.7 * title_overlap + 0.3 * body_overlap)
        document.freshness = freshness_score(
            document.published_at, document.fetched_at, freshness_mode, now
        )
        document.score = (
            4.5 * document.rrf_score
            + 0.24 * document.candidate_score
            + 0.33 * document.relevance
            + 0.22 * document.authority
            + 0.16 * document.freshness
            + 0.14 * document.extraction_quality
            + source_bonus(document.source_type)
        )
    rows = [
        {
            "title": item.title,
            "content": item.text[:6000],
            "uri": item.url,
            "score": item.score,
            "_best_position": index,
            "_upstream_score": item.score,
            "_preference_score": (
                source_preference_score(item.source_type, item.authority)
                if normalized_preference != "any"
                else 0.0
            ),
        }
        for index, item in enumerate(
            sorted(documents, key=lambda value: value.score, reverse=True),
            1,
        )
    ]
    semantic_order = RetrievalReranker(scorer=scorer).rank(
        query,
        query_views,
        rows,
        limit=len(rows),
        preference_weight=0.12 if normalized_preference != "any" else 0.0,
    )
    by_url = {item.url: item for item in documents}
    ordered = [
        by_url[str(row.get("uri") or "")]
        for row in semantic_order.items
        if str(row.get("uri") or "") in by_url
    ]
    selected: List[RealtimeDocument] = []
    domains: Dict[str, int] = {}
    hashes: List[int] = []
    for item in ordered:
        domain = (urlsplit(item.url).hostname or "").casefold()
        if domains.get(domain, 0) >= per_domain_limit:
            continue
        current_hash = int(item.simhash or "0", 16)
        if any(bin(current_hash ^ previous).count("1") <= 3 for previous in hashes):
            continue
        domains[domain] = domains.get(domain, 0) + 1
        hashes.append(current_hash)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def to_search_results(query: str, documents: Iterable[RealtimeDocument]) -> List[SearchResult]:
    output: List[SearchResult] = []
    for item in documents:
        identifier = int.from_bytes(
            hashlib.blake2b(item.url.encode("utf-8"), digest_size=7).digest(), "big"
        )
        output.append(
            SearchResult(
                document_id=-identifier,
                url=item.url,
                title=item.title,
                snippet=best_snippet(item.text, query),
                content=item.text,
                published_at=item.published_at,
                fetched_at=item.fetched_at,
                source_type=item.source_type,
                authority=item.authority,
                score=item.score,
                score_components={
                    "rrf": item.rrf_score,
                    "candidate_score": item.candidate_score,
                    "relevance": item.relevance,
                    "authority": item.authority,
                    "freshness": item.freshness,
                    "extraction_quality": item.extraction_quality,
                    "source_bonus": source_bonus(item.source_type),
                    "snippet_fallback": float(
                        item.retrieval_mode == "search_snippet_fallback"
                    ),
                },
            )
        )
    return output


def source_bonus(source_type: str) -> float:
    return {
        "regulator": 0.12,
        "company_filing": 0.11,
        "official_docs": 0.10,
        "paper": 0.09,
        "github_release": 0.08,
        "official_repository": 0.075,
        "academic": 0.07,
        "news": 0.04,
        "forum": -0.02,
    }.get(source_type, 0.0)


def source_preference_score(source_type: str, authority: float) -> float:
    """Score declared source metadata; never infer a topic from the query."""

    preferred = {
        "academic",
        "company_filing",
        "github_release",
        "official_docs",
        "official_repository",
        "paper",
        "regulator",
    }
    return max(0.0, min(1.0, float(authority))) if source_type in preferred else 0.0


def freshness_score(
    published_at: Optional[str], fetched_at: float, mode: str, now: datetime
) -> float:
    timestamp: Optional[datetime] = None
    if published_at:
        value = published_at.strip()
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                timestamp = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                timestamp = None
    if timestamp is None:
        # Fetch time is not publication time. Treat missing dates as neutral,
        # otherwise any old undated page fetched today incorrectly looks fresh.
        return {"realtime": 0.25, "latest": 0.45, "stable": 0.78}.get(mode, 0.6)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - timestamp.astimezone(timezone.utc)).total_seconds() / 86400.0)
    half_life = {"realtime": 1.5, "latest": 30.0, "stable": 3650.0}.get(mode, 3650.0)
    return math.exp(-math.log(2.0) * age_days / half_life)
