from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from ..semantic_selection import PairScorer, select_diverse_items
from ..text import canonicalize_url, search_tokens
from .query_compiler import SourcePreference, normalize_source_preference


_SOURCE_TERM_NOISE = {
    "a", "an", "and", "current", "find", "for", "from", "in", "is", "its",
    "latest", "materials", "new", "of", "official", "on", "original", "please",
    "primary", "quarterly", "recent", "release", "repository", "search", "site",
    "source", "the", "to", "update", "what", "which",
    "一下", "一手", "以官", "公式", "公告", "原始", "发布", "官方", "官网",
    "当前", "最新", "最近", "目前", "请以", "请找", "资料", "来源", "页面",
}
_INSTITUTIONAL_LABELS = {"ac", "edu", "gov", "int", "mil"}
_ORIGINAL_PLATFORM_HOSTS = {
    "arxiv.org",
    "doi.org",
    "github.com",
    "gitlab.com",
}
_ORIGINAL_PATH_MARKERS = (
    "/abs/",
    "/commit/",
    "/commits/",
    "/disclosure/",
    "/filing/",
    "/paper/",
    "/pdf/",
    "/publication/",
    "/releases/",
)


@dataclass(frozen=True)
class RerankResult:
    items: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


class RetrievalReranker:
    """One generic reranker shared by candidates, pages, and Evidence."""

    def __init__(self, *, scorer: PairScorer | None = None) -> None:
        self.scorer = scorer

    def rank(
        self,
        question: str,
        query_views: Sequence[str],
        items: Sequence[Mapping[str, Any]],
        *,
        limit: int,
        preference_weight: float = 0.0,
    ) -> RerankResult:
        selection = select_diverse_items(
            question,
            query_views,
            items,
            limit=limit,
            scorer=self.scorer,
            preference_weight=preference_weight,
        )
        return RerankResult(selection.items, selection.metadata())

    def rank_candidates(
        self,
        question: str,
        query_views: Sequence[str],
        candidates: Sequence[Any],
        *,
        limit: int,
        source_preference: SourcePreference | str = "any",
    ) -> tuple[list[Any], dict[str, Any]]:
        preference = normalize_source_preference(str(source_preference))
        source_terms = self._source_terms(question, query_views)
        rows: list[dict[str, Any]] = []
        by_url: dict[str, Any] = {}
        alignments: dict[str, float] = {}
        positions: dict[str, int] = {}
        for position, candidate in enumerate(candidates, 1):
            url = str(getattr(candidate, "url", "") or "")
            canonical = canonicalize_url(url) or url
            if not canonical:
                continue
            by_url.setdefault(canonical, candidate)
            positions.setdefault(canonical, position)
            alignment = (
                self._source_alignment(candidate, source_terms, preference)
                if preference != "any"
                else 0.0
            )
            alignments[canonical] = max(alignments.get(canonical, 0.0), alignment)
            rows.append(
                {
                    "title": str(getattr(candidate, "title", "") or ""),
                    "content": str(getattr(candidate, "snippet", "") or ""),
                    "uri": canonical,
                    "_best_position": position,
                    "_upstream_score": float(
                        getattr(candidate, "candidate_score", 0.0)
                        or getattr(candidate, "rrf_score", 0.0)
                        or getattr(candidate, "engine_score", 0.0)
                        or 0.0
                    ),
                    "_preference_score": alignment,
                }
            )
        result = self.rank(
            question,
            query_views,
            rows,
            limit=limit,
            preference_weight=0.18 if preference != "any" else 0.0,
        )
        ordered = [by_url[str(item.get("uri") or "")] for item in result.items]
        if preference != "any":
            def preference_key(item: Any) -> tuple[float, int]:
                url = str(getattr(item, "url", "") or "")
                canonical = canonicalize_url(url) or url
                alignment = alignments.get(canonical, 0.0)
                # Keep semantic/MMR order inside a supported source tier. When
                # there is no source evidence, retain the upstream engine order
                # rather than inventing authority from ordinary relevance.
                fallback_position = (
                    positions.get(canonical, len(positions) + 1)
                    if alignment <= 0.0
                    else 0
                )
                return -alignment, fallback_position

            ordered.sort(key=preference_key)
        for item in ordered:
            url = str(getattr(item, "url", "") or "")
            canonical = canonicalize_url(url) or url
            components = getattr(item, "score_components", None)
            if isinstance(components, dict):
                components["source_preference_alignment"] = round(
                    alignments.get(canonical, 0.0), 6
                )
        result.metadata["source_preference"] = preference
        result.metadata["source_preference_reordered"] = preference != "any"
        return ordered, result.metadata

    @staticmethod
    def _source_terms(question: str, query_views: Sequence[str]) -> tuple[str, ...]:
        output: list[str] = []
        seen: set[str] = set()
        for value in (question, *query_views):
            for token in search_tokens(str(value or "")):
                folded = token.casefold()
                compact = re.sub(r"[^a-z0-9]+", "", folded) if folded.isascii() else folded
                if (
                    not compact
                    or compact in _SOURCE_TERM_NOISE
                    or (compact.isascii() and len(compact) < 2)
                    or compact in seen
                ):
                    continue
                seen.add(compact)
                output.append(compact)
        return tuple(output[:24])

    @staticmethod
    def _source_alignment(
        candidate: Any,
        terms: Sequence[str],
        preference: SourcePreference,
    ) -> float:
        url = str(getattr(candidate, "url", "") or "")
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold().removeprefix("www.")
        if not host:
            return 0.0
        if str(getattr(candidate, "engine", "") or "").casefold() == "direct":
            return 1.0
        labels = tuple(part for part in host.split(".") if part)
        host_compact = re.sub(r"[^a-z0-9]+", "", host)
        path = unquote(parsed.path or "").casefold()
        path_compact = re.sub(r"[^a-z0-9]+", "", path)
        entity_alignment = 0.0
        for term in terms:
            if not term.isascii():
                continue
            if term in labels or (len(term) >= 3 and term in host_compact):
                entity_alignment = max(entity_alignment, 1.0)
            elif len(term) >= 3 and term in path_compact:
                entity_alignment = max(entity_alignment, 0.82)
        brand = RetrievalReranker._registrable_brand(labels)
        title_terms = {
            re.sub(r"[^a-z0-9]+", "", token.casefold())
            for token in search_tokens(str(getattr(candidate, "title", "") or ""))
            if token.isascii()
        }
        brand_matches_query = bool(
            brand
            and any(
                term.isascii()
                and len(term) >= 3
                and (term == brand or term in brand or brand in term)
                for term in terms
            )
        )
        first_party = (
            0.94 if brand_matches_query and brand in title_terms else 0.0
        )
        channels = {
            str(value).casefold()
            for value in getattr(candidate, "source_channels", ()) or ()
        }
        original_artifact = 0.0
        if channels - {"", "general", "web"}:
            original_artifact = max(original_artifact, 0.86)
        if host in _ORIGINAL_PLATFORM_HOSTS or any(
            host.endswith("." + value) for value in _ORIGINAL_PLATFORM_HOSTS
        ):
            original_artifact = max(original_artifact, 0.92)
        if any(marker in path for marker in _ORIGINAL_PATH_MARKERS):
            original_artifact = max(original_artifact, 0.9)
        institutional = 0.0
        if _INSTITUTIONAL_LABELS.intersection(labels):
            institutional = 0.78
        if preference == "original":
            return max(original_artifact, 0.62 * first_party, 0.58 * entity_alignment)
        if preference == "primary":
            return max(first_party, institutional, original_artifact, entity_alignment)
        return max(first_party, institutional, 0.9 * original_artifact, entity_alignment)

    @staticmethod
    def _registrable_brand(labels: Sequence[str]) -> str:
        if len(labels) < 2:
            return labels[0] if labels else ""
        public_second_level = {"ac", "co", "com", "edu", "gov", "net", "org"}
        index = -3 if len(labels) >= 3 and labels[-2] in public_second_level else -2
        return re.sub(r"[^a-z0-9]+", "", labels[index])
