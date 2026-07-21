from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlsplit

from .config import SearchConfig
from .db import SearchDatabase
from .text import best_snippet, search_tokens


class DenseRetriever(Protocol):
    def search(self, query: str, limit: int) -> Sequence[Tuple[int, float]]:
        ...


@dataclass
class SearchResult:
    document_id: int
    url: str
    title: str
    snippet: str
    content: str
    published_at: Optional[str]
    fetched_at: float
    source_type: str
    authority: float
    score: float
    score_components: Dict[str, float]
    source_id: Optional[str] = None
    updated_at: Optional[str] = None
    matched_channels: Tuple[str, ...] = ()

    def to_dict(self, include_content: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_content:
            data.pop("content", None)
        return data


class HybridSearcher:
    def __init__(
        self,
        database: SearchDatabase,
        config: Optional[SearchConfig] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        reranker: Optional[Callable[[str, Sequence[SearchResult]], Sequence[float]]] = None,
    ) -> None:
        self.database = database
        self.config = config or SearchConfig()
        self.dense_retriever = dense_retriever
        self.reranker = reranker

    def search(self, query: str, *, freshness: str = "stable", limit: Optional[int] = None) -> List[SearchResult]:
        tokens = search_tokens(query)
        lexical_rows = self.database.search_fts(tokens, self.config.candidate_limit)
        if not lexical_rows and not self.dense_retriever:
            return []

        lexical_rank = {int(row["id"]): rank for rank, row in enumerate(lexical_rows, start=1)}
        dense_rows = list(self.dense_retriever.search(query, self.config.candidate_limit)) if self.dense_retriever else []
        dense_rank = {int(doc_id): rank for rank, (doc_id, _) in enumerate(dense_rows, start=1)}
        all_ids = list(dict.fromkeys(list(lexical_rank) + [int(doc_id) for doc_id, _ in dense_rows]))
        rows_by_id = {int(row["id"]): row for row in lexical_rows}
        for row in self.database.get_documents(doc_id for doc_id in all_ids if doc_id not in rows_by_id):
            rows_by_id[int(row["id"])] = row

        now = datetime.now(timezone.utc)
        query_set = set(tokens)
        anchor_tokens = [
            token for token in tokens
            if len(token) >= 4 and any("a" <= ch <= "z" for ch in token.casefold())
        ]
        candidates: List[SearchResult] = []
        for doc_id in all_ids:
            row = rows_by_id.get(doc_id)
            if not row:
                continue
            searchable = str(row.get("search_text") or "").casefold()
            if anchor_tokens and not any(token.casefold() in searchable for token in anchor_tokens):
                continue
            rrf = 0.0
            if doc_id in lexical_rank:
                rrf += 1.0 / (60.0 + lexical_rank[doc_id])
            if doc_id in dense_rank:
                rrf += 1.0 / (60.0 + dense_rank[doc_id])
            title_tokens = set(search_tokens(str(row["title"])))
            title_overlap = len(query_set & title_tokens) / max(1, len(query_set))
            authority = float(row.get("authority") or 0.5)
            freshness_score = self._freshness(row, freshness, now)
            source_bonus = self._source_bonus(str(row.get("source_type") or "web"))
            total = 5.0 * rrf + 0.22 * title_overlap + 0.18 * authority + 0.14 * freshness_score + source_bonus
            candidates.append(
                SearchResult(
                    document_id=doc_id,
                    url=str(row["canonical_url"]),
                    title=str(row["title"]),
                    snippet=best_snippet(str(row["content"]), query),
                    content=str(row["content"]),
                    published_at=str(row["published_at"]) if row.get("published_at") else None,
                    fetched_at=float(row["fetched_at"]),
                    source_type=str(row.get("source_type") or "web"),
                    authority=authority,
                    score=total,
                    score_components={
                        "rrf": rrf,
                        "title_overlap": title_overlap,
                        "authority": authority,
                        "freshness": freshness_score,
                        "source_bonus": source_bonus,
                    },
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        if self.reranker and candidates:
            top = candidates[:30]
            scores = list(self.reranker(query, top))
            for item, score in zip(top, scores):
                item.score = 0.45 * item.score + 0.55 * float(score)
                item.score_components["reranker"] = float(score)
            candidates.sort(key=lambda item: item.score, reverse=True)
        return self._diversify(candidates, limit or self.config.result_limit)

    def _diversify(self, candidates: Sequence[SearchResult], limit: int) -> List[SearchResult]:
        domain_counts: Dict[str, int] = {}
        selected: List[SearchResult] = []
        for candidate in candidates:
            domain = (urlsplit(candidate.url).hostname or "").casefold()
            if domain_counts.get(domain, 0) >= self.config.per_domain_limit:
                continue
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            selected.append(candidate)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _source_bonus(source_type: str) -> float:
        return {
            "regulator": 0.12,
            "company_filing": 0.11,
            "official_docs": 0.10,
            "paper": 0.09,
            "github_release": 0.08,
            "local_document": 0.07,
            "news": 0.04,
            "forum": -0.02,
        }.get(source_type, 0.0)

    @staticmethod
    def _freshness(row: Dict[str, Any], freshness: str, now: datetime) -> float:
        timestamp = None
        if row.get("published_at"):
            value = str(row["published_at"])
            try:
                timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                try:
                    timestamp = parsedate_to_datetime(value)
                except (TypeError, ValueError):
                    timestamp = None
        if timestamp is None:
            timestamp = datetime.fromtimestamp(float(row["fetched_at"]), tz=timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - timestamp.astimezone(timezone.utc)).total_seconds() / 86400.0)
        half_life = {"realtime": 1.5, "latest": 30.0, "stable": 3650.0}.get(freshness, 3650.0)
        return math.exp(-math.log(2.0) * age_days / half_life)
