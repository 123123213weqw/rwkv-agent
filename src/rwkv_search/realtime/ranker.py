from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlsplit

from ..search import SearchResult
from ..text import best_snippet, search_tokens
from .types import RealtimeDocument


_ENTITY_STOPWORDS = {
    "what", "who", "where", "when", "why", "how", "is", "are", "the",
    "a", "an", "of", "for", "and", "or", "latest", "recent", "current",
    "official", "documentation", "docs", "news", "release", "notes",
}

_FINANCE_QUERY_RE = re.compile(
    r"(股票|股市|A股|港股|美股|基金|ETF|买入|卖出|行情|市盈率|财报|"
    r"stock|equity|NASDAQ|NYSE|filing)",
    re.I,
)
_FINANCE_TITLE_RE = re.compile(
    r"(股票|股市|A股|港股|美股|基金|ETF|行情|指数|上市公司|公告|财报|业绩|"
    r"stock|equity|market|NASDAQ|NYSE|filing|earnings)",
    re.I,
)
_FINANCE_PRIMARY_TYPES = {"regulator", "company_filing"}


def rank_documents(
    query: str,
    documents: Sequence[RealtimeDocument],
    *,
    freshness_mode: str,
    limit: int,
    per_domain_limit: int = 2,
) -> List[RealtimeDocument]:
    query_tokens = set(search_tokens(query))
    entity_tokens = {
        token.casefold()
        for token in query_tokens
        if len(token) >= 2
        and token.isascii()
        and token.isalnum()
        and token.casefold() not in _ENTITY_STOPWORDS
    }
    now = datetime.now(timezone.utc)
    finance_mode = bool(_FINANCE_QUERY_RE.search(query))
    entity_matches: Dict[int, bool] = {}
    finance_matches: Dict[int, bool] = {}
    for document in documents:
        title_tokens = set(search_tokens(document.title))
        body_tokens = set(search_tokens(document.text[:100000]))
        title_overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
        body_overlap = len(query_tokens & body_tokens) / max(1, len(query_tokens))
        document.relevance = min(1.0, 0.7 * title_overlap + 0.3 * body_overlap)
        host = (urlsplit(document.url).hostname or "").casefold()
        domain_parts = set(host.replace("www.", "").split("."))
        entity_haystack = title_tokens | body_tokens | set(
            search_tokens(document.url)
        )
        entity_matches[id(document)] = not entity_tokens or bool(
            entity_tokens & entity_haystack
        )
        finance_haystack = f"{document.title} {document.text[:3000]}"
        finance_matches[id(document)] = (
            not finance_mode
            or (
                document.source_type in _FINANCE_PRIMARY_TYPES
                and bool(_FINANCE_TITLE_RE.search(finance_haystack))
            )
            or (
                document.source_type == "news"
                and bool(_FINANCE_TITLE_RE.search(document.title))
            )
        )
        if entity_tokens & domain_parts:
            document.authority = max(document.authority, 0.92)
            if document.source_type == "web":
                document.source_type = "official_docs"
        document.freshness = freshness_score(
            document.published_at, document.fetched_at, freshness_mode, now
        )
        document.score = (
            4.5 * document.rrf_score
            + 0.33 * document.relevance
            + 0.22 * document.authority
            + 0.16 * document.freshness
            + 0.14 * document.extraction_quality
            + source_bonus(document.source_type)
        )
    # A named Latin entity such as Python/RWKV is a hard constraint. Without
    # it, common Chinese question words can rank a completely unrelated page.
    ordered = sorted(
        (
            item
            for item in documents
            if entity_matches.get(id(item), True)
            and finance_matches.get(id(item), True)
        ),
        key=lambda item: item.score,
        reverse=True,
    )
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
                    "relevance": item.relevance,
                    "authority": item.authority,
                    "freshness": item.freshness,
                    "extraction_quality": item.extraction_quality,
                    "source_bonus": source_bonus(item.source_type),
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
