from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import unquote, urlsplit

from ..pipeline.reranker import RetrievalReranker
from ..pipeline.query_compiler import SourcePreference
from ..semantic_selection import PairScorer
from ..text import search_tokens
from .types import DiscoveredURL


# Candidate admission deliberately uses page features rather than topic/domain
# routing.  The small lists below describe page shapes (search, dictionary,
# login, error), not preferred sources for any business domain.
_QUERY_NOISE = {
    "a",
    "an",
    "and",
    "are",
    "current",
    "find",
    "for",
    "how",
    "is",
    "latest",
    "new",
    "official",
    "of",
    "please",
    "recent",
    "search",
    "the",
    "to",
    "what",
    "which",
    "官网",
    "官方",
    "当前",
    "最新",
    "最近",
    "目前",
    "什么",
    "信息",
    "一下",
    "搜索",
    "查询",
    "请找",
    "请以",
}
_SEARCH_HOST_RE = re.compile(
    r"(^|\.)(?:bing|google|baidu|sogou|so|duckduckgo)\.[a-z.]+$", re.I
)
_SEARCH_PATH_RE = re.compile(r"^/(?:search|s|web)(?:/|$)", re.I)
_DICTIONARY_RE = re.compile(
    r"字典|词典|汉典|百科|释义|是什么意思|拼音|部首|笔顺|在线翻译|翻译|音标|读音|例句|"
    r"dictionary|definition|pronunciation|thesaurus|/(?:dict|dictionary)(?:/|\.)|"
    r"/word(?:\?|/)|(?:^|[./_-])baike(?:[./_-]|$)|"
    r"(?:^|[./_-])(?:cidian|zidian)(?:[./_-]|$)",
    re.I,
)
_LOGIN_RE = re.compile(
    r"(?:^|[/_.-])(?:login|signin|sign-in|captcha|verify)(?:[/_.-]|$)|"
    r"\bcaptcha\b|登录|验证码|安全验证|verify you are human|sign in to continue",
    re.I,
)
_ERROR_RE = re.compile(
    r"(?:^|[/_.-])(?:404|403|error)(?:[/_.-]|$)|"
    r"page not found|access denied|forbidden|页面不存在|访问被拒绝",
    re.I,
)
_SITE_RE = re.compile(r"(?:^|\s)site:([^\s]+)", re.I)


@dataclass
class CandidateAdmission:
    admitted: List[DiscoveredURL]
    rejected: List[DiscoveredURL] = field(default_factory=list)
    rejection_counts: Dict[str, int] = field(default_factory=dict)


def _normalized_host(url: str) -> str:
    host = (urlsplit(url).hostname or "").casefold().strip(".")
    return host[4:] if host.startswith("www.") else host


def _site_matches(host: str, site: str) -> bool:
    expected = site.casefold().strip().strip(".")
    if expected.startswith("www."):
        expected = expected[4:]
    return bool(host and expected and (host == expected or host.endswith("." + expected)))


def _extract_sites(values: Iterable[str]) -> Tuple[str, ...]:
    output: List[str] = []
    seen = set()
    for value in values:
        for match in _SITE_RE.finditer(value):
            site = match.group(1).rstrip(".,;，。；").casefold()
            if site and site not in seen:
                seen.add(site)
                output.append(site)
    return tuple(output)


def candidate_rejection_reasons(
    query: str,
    candidate: DiscoveredURL,
    *,
    explicit_sites: Sequence[str] = (),
) -> List[str]:
    """Return high-precision pre-fetch rejection reasons.

    Ambiguous quality signals only affect ranking.  Rejection is intentionally
    limited to page shapes that are almost never useful evidence, so candidate
    recall is not traded away for a cosmetically clean result list.
    """

    parsed = urlsplit(candidate.url)
    host = _normalized_host(candidate.url)
    path = unquote(parsed.path or "/").casefold()
    page_shape = f"{candidate.url} {candidate.title}"
    reasons: List[str] = []

    if not host or parsed.scheme not in {"http", "https"}:
        reasons.append("invalid_url")
        return reasons
    if explicit_sites and candidate.engine != "direct" and not any(
        _site_matches(host, site) for site in explicit_sites
    ):
        reasons.append("outside_explicit_site")
    if _SEARCH_HOST_RE.search(host) and (
        path in {"", "/"} or _SEARCH_PATH_RE.search(path)
    ):
        reasons.append("search_homepage")
    if _DICTIONARY_RE.search(page_shape):
        reasons.append("dictionary")
    if _LOGIN_RE.search(page_shape):
        reasons.append("login_or_captcha")
    if _ERROR_RE.search(page_shape):
        reasons.append("error_page")
    if not candidate.title.strip() and not candidate.snippet.strip():
        reasons.append("empty_metadata")
    return sorted(set(reasons))


def _content_tokens(values: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        for token in search_tokens(value):
            for folded in _token_aliases(token):
                if folded in _QUERY_NOISE or len(folded) < 2:
                    continue
                if folded not in seen:
                    seen.add(folded)
                    output.append(folded)
    return output


def _field_tokens(value: str) -> set[str]:
    return {
        alias
        for token in search_tokens(unquote(value))
        for alias in _token_aliases(token)
    }


def _token_aliases(token: str) -> set[str]:
    folded = token.casefold()
    output = {folded}
    if folded.isascii():
        compact = re.sub(r"[^a-z0-9]+", "", folded)
        if len(compact) >= 2:
            output.add(compact)
        output.update(
            part for part in re.split(r"[^a-z0-9]+", folded) if len(part) >= 2
        )
    return output


def _weighted_coverage(
    query_tokens: Sequence[str], field_tokens: set[str], weights: Dict[str, float]
) -> float:
    total = sum(weights.get(token, 1.0) for token in query_tokens)
    if total <= 0:
        return 0.0
    matched = sum(weights.get(token, 1.0) for token in query_tokens if token in field_tokens)
    return matched / total


def _entity_tokens(query_tokens: Sequence[str]) -> List[str]:
    latin = [
        token
        for token in query_tokens
        if token.isascii()
        and any(character.isalpha() for character in token)
        and token not in _QUERY_NOISE
    ]
    # The first content expression is normally the named subject in concise
    # P4 queries.  Keep the signal bounded instead of trying to classify a
    # business domain.
    return latin[:3] or list(query_tokens[:1])


def _score_candidates(
    query: str,
    queries: Sequence[str],
    candidates: Sequence[DiscoveredURL],
) -> None:
    query_tokens = _content_tokens([*queries, query])
    candidate_fields: List[Tuple[set[str], set[str], set[str]]] = []
    document_frequency: Counter[str] = Counter()
    for item in candidates:
        parsed = urlsplit(item.url)
        title_tokens = _field_tokens(item.title)
        url_tokens = _field_tokens(
            f"{_normalized_host(item.url)} {parsed.path} {parsed.query}"
        )
        snippet_tokens = _field_tokens(item.snippet)
        candidate_fields.append((title_tokens, url_tokens, snippet_tokens))
        for token in set().union(title_tokens, url_tokens, snippet_tokens):
            if token in query_tokens:
                document_frequency[token] += 1

    count = max(1, len(candidates))
    weights = {
        token: math.log(1.0 + (count + 0.5) / (document_frequency[token] + 0.5))
        for token in query_tokens
    }
    entity_tokens = _entity_tokens(query_tokens)
    maximum_rrf = max((item.rrf_score for item in candidates), default=0.0)
    maximum_engine_score = max((item.engine_score for item in candidates), default=0.0)

    for original_position, (item, fields) in enumerate(
        zip(candidates, candidate_fields), start=1
    ):
        title_tokens, url_tokens, snippet_tokens = fields
        title_coverage = _weighted_coverage(query_tokens, title_tokens, weights)
        url_coverage = _weighted_coverage(query_tokens, url_tokens, weights)
        snippet_coverage = _weighted_coverage(query_tokens, snippet_tokens, weights)
        combined = title_tokens | url_tokens | snippet_tokens
        entity_coverage = (
            sum(token in combined for token in entity_tokens) / max(1, len(entity_tokens))
        )
        rank_prior = 1.0 / math.log2(max(2, int(item.rank or original_position) + 1))
        rrf_prior = item.rrf_score / maximum_rrf if maximum_rrf > 0 else 0.0
        engine_prior = (
            item.engine_score / maximum_engine_score if maximum_engine_score > 0 else 0.0
        )
        score = (
            0.25 * title_coverage
            + 0.22 * url_coverage
            + 0.05 * snippet_coverage
            + 0.08 * entity_coverage
            + 0.25 * rank_prior
            + 0.08 * rrf_prior
            + 0.06 * engine_prior
        )
        if item.engine == "direct":
            score += 1.0
        item.candidate_score = round(score, 8)
        item.score_components = {
            "title_coverage": round(title_coverage, 6),
            "url_coverage": round(url_coverage, 6),
            "snippet_coverage": round(snippet_coverage, 6),
            "entity_coverage": round(entity_coverage, 6),
            "rank_prior": round(rank_prior, 6),
            "rrf_prior": round(rrf_prior, 6),
            "engine_prior": round(engine_prior, 6),
            "original_position": float(original_position),
        }


def admit_candidates(
    query: str,
    queries: Sequence[str],
    candidates: Sequence[DiscoveredURL],
    *,
    max_candidates: int,
    per_domain_limit: int = 3,
    scorer: PairScorer | None = None,
    source_preference: SourcePreference | str = "any",
) -> CandidateAdmission:
    """Filter obvious garbage, score candidates, and diversify the fetch prefix."""

    explicit_sites = _extract_sites([query, *queries])
    admitted: List[DiscoveredURL] = []
    rejected: List[DiscoveredURL] = []
    rejection_counts: Counter[str] = Counter()
    for item in candidates:
        reasons = candidate_rejection_reasons(
            query, item, explicit_sites=explicit_sites
        )
        item.rejection_reasons = reasons
        if reasons:
            rejected.append(item)
            rejection_counts.update(reasons)
        else:
            admitted.append(item)

    _score_candidates(query, queries, admitted)

    def score_key(item: DiscoveredURL) -> tuple[float, float, int]:
        return (
            item.candidate_score,
            item.rrf_score,
            -int(item.rank or 10**6),
        )

    # Preserve the set of the search engine's admitted top 10 while allowing
    # metadata reranking and diversity inside that set. This is a standard
    # recall-protected rerank boundary: a noisy metadata score may change
    # fetch order, but cannot evict a first-page result from Recall@10.
    protected = [
        item
        for item in admitted
        if item.score_components.get("original_position", 10**6) <= 10
    ]
    protected_ids = {id(item) for item in protected}
    tail = [item for item in admitted if id(item) not in protected_ids]
    protected.sort(key=score_key, reverse=True)
    tail.sort(key=score_key, reverse=True)
    reranker = RetrievalReranker(scorer=scorer)
    if protected:
        protected, _ = reranker.rank_candidates(
            query,
            queries,
            protected,
            limit=len(protected),
            source_preference=source_preference,
        )
    if tail:
        tail, _ = reranker.rank_candidates(
            query,
            queries,
            tail,
            limit=len(tail),
            source_preference=source_preference,
        )

    # SearXNG applies result-container ranking and duplicate merging; this
    # second bounded pass adds fetch-budget diversity.  Overflow candidates are
    # retained after the diverse prefix, so Recall@20 is not needlessly lost.
    def diversify(values: Sequence[DiscoveredURL]) -> List[DiscoveredURL]:
        selected: List[DiscoveredURL] = []
        overflow: List[DiscoveredURL] = []
        domains: Counter[str] = Counter()
        limit = max(1, int(per_domain_limit))
        for item in values:
            host = _normalized_host(item.url)
            if domains[host] >= limit:
                overflow.append(item)
                continue
            domains[host] += 1
            selected.append(item)
        selected.extend(overflow)
        return selected

    selected = [*diversify(protected), *diversify(tail)]
    return CandidateAdmission(
        admitted=selected[: max(0, max_candidates)],
        rejected=rejected,
        rejection_counts=dict(sorted(rejection_counts.items())),
    )
