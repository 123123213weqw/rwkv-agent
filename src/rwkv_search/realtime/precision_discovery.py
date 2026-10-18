from __future__ import annotations

import copy
import re
from typing import Iterable, List, Sequence, Tuple

from ..analysis.query import QueryAnalyzer
from ..pipeline.discovery import DiscoveryLayer
from ..pipeline.source_selector import SourceCapability, SourceSelector
from ..semantic_selection import PairScorer
from ..text import canonicalize_url
from .types import DiscoveredURL, RealtimeDocument


_CJK_SEQUENCE = re.compile(r"[\u3400-\u9fff]{2,}")
_ENGLISH_SHELL = re.compile(
    r"^(?:(?:please|could you|can you)\s+)?(?:find|search(?:\s+for)?|show|"
    r"look\s+up|tell\s+me|what\s+is|what\s+are)\s+",
    re.I,
)
_ENGLISH_TOKEN = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9_+#'-]*\.)+[A-Za-z]{2,}|"
    r"[A-Za-z][A-Za-z0-9_+#'.-]*|\d+(?:\.\d+)+"
)
_ANCHOR_NUMBER = re.compile(r"^(?:19|20)\d{2}$|^\d+(?:[./-]\d+)+$")
_QUERY_FUNCTION_WORDS = frozenset(
    {
        "a", "according", "an", "and", "are", "at", "find", "for", "from",
        "in", "is", "its", "look", "of", "on", "please", "search", "show",
        "site", "source", "the", "to", "use", "website", "what", "with",
    }
)
_INTERMEDIARY_HOSTS = frozenset(
    {
        "baidu.com", "bing.com", "facebook.com", "google.com", "linkedin.com",
        "medium.com", "reddit.com", "so.com", "sogou.com", "wikipedia.org",
        "x.com", "youtube.com", "zhihu.com",
    }
)
_CHANNELS = {
    "general": SourceCapability(
        "general",
        "General public web pages, official websites, documentation, release, news, "
        "policy and company pages. 通用公开网页、官网、文档、版本、新闻和政策。",
        always=True,
    ),
    "repos": SourceCapability(
        "repos",
        "GitHub GitLab source-code hosting, repository repositories, repo, commit, "
        "release and issue. 代码仓库、提交、版本发布和问题记录。",
    ),
    "science": SourceCapability(
        "science",
        "Scholarly paper papers, preprint, arXiv, DOI and academic publication. "
        "论文、预印本、DOI 和学术出版物。",
    ),
}
_PRIMARY = {
    "primary": SourceCapability(
        "primary",
        "The user requires an official, original, first-party, regulatory or "
        "primary source instead of commentary. 用户要求官方、原始或一手来源。",
    )
}


def compact_general_query(query: str) -> str:
    """Remove an English conversation shell without classifying its topic."""

    value = " ".join(str(query or "").split()).strip()
    if not value or _CJK_SEQUENCE.search(value):
        return value
    stripped = _ENGLISH_SHELL.sub("", value).strip(" ?.!,:;")
    tokens = _ENGLISH_TOKEN.findall(stripped)
    if len(tokens) < 5:
        return stripped or value
    priority: List[str] = []
    content: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        folded = token.casefold().strip("'")
        if not folded or folded in seen or folded in _QUERY_FUNCTION_WORDS:
            continue
        seen.add(folded)
        is_domain = "." in token and token.rsplit(".", 1)[-1].isalpha()
        is_named = token.isupper() or token[:1].isupper()
        (priority if is_domain or is_named else content).append(token)
    return " ".join([*priority, *content][:14]) or stripped or value


def build_original_query_lane(
    original_query: str,
    model_query: str,
    *,
    site: str = "",
    analyzer: QueryAnalyzer | None = None,
) -> str:
    """Build one deterministic complementary lane from the user query.

    The original wording is trusted input, unlike facts introduced by a model
    rewrite.  QueryAnalyzer removes conversational surface syntax and, for a
    long question, retains high-value exact terms, names and numbers.  A
    caller-supplied site remains a hard constraint.  Returning an empty string
    means that the lane would duplicate the already compiled model query.
    """

    original = " ".join(str(original_query or "").split()).strip()
    if not original:
        return ""
    analysis = (analyzer or QueryAnalyzer(max_queries=1)).analyze(original)
    values = tuple(analysis.search_queries) or (analysis.resolved_query,)
    lane = " ".join(str(values[0] or "").split()).strip()
    if not lane:
        return ""
    lane = re.sub(r"(?:^|\s)site:[^\s]+", " ", lane, flags=re.I)
    lane = " ".join(lane.split()).strip()
    site_value = normalized_host(site)
    if site_value:
        lane = f"{lane} site:{site_value}"
    def normalize(value: str) -> str:
        return " ".join(str(value or "").casefold().split())

    if normalize(lane) == normalize(model_query):
        return ""
    return lane


def build_host_token_query_lane(
    original_query: str,
    model_query: str,
    *,
    site: str,
    analyzer: QueryAnalyzer | None = None,
) -> str:
    """Build a scoped discovery view that does not depend on ``site:``.

    Search backends implement the ``site:`` operator inconsistently.  This
    lane uses the trusted original wording plus the caller-provided host as an
    ordinary token.  The caller must still enforce that host as a hard result
    boundary, so relaxing query syntax never relaxes scope.
    """

    host = normalized_host(site)
    if not host:
        return ""
    base = build_original_query_lane(
        original_query,
        model_query,
        analyzer=analyzer,
    )
    if not base:
        original = " ".join(str(original_query or "").split()).strip()
        base = re.sub(r"(?:^|\s)site:[^\s]+", " ", original, flags=re.I)
        base = " ".join(base.split()).strip()
    if not base:
        return ""
    lane = f"{base} {host}"

    def normalize(value: str) -> str:
        return " ".join(str(value or "").casefold().split())

    if normalize(lane) == normalize(model_query):
        return ""
    return lane


def build_anchor_phrase_query_lane(
    original_query: str,
    model_query: str,
    *,
    site: str,
    analyzer: QueryAnalyzer | None = None,
) -> str:
    """Compile a short exact-anchor query from user-authored text.

    This is the lexical complement to a natural-language model rewrite.  It
    prioritizes explicitly quoted/structured spans, dates, and a small number
    of long content tokens while preserving their original order.  No target
    URL, topic vocabulary, answer, or site-specific path rule is used.
    """

    host = normalized_host(site)
    original = " ".join(str(original_query or "").split()).strip()
    if not host or not original:
        return ""
    analysis = (analyzer or QueryAnalyzer(max_queries=1)).analyze(original)
    anchors: List[str] = []
    seen: set[str] = set()

    def add(value: str, *, quoted: bool = False) -> None:
        clean = " ".join(str(value or "").split()).strip(' "“”‘’\'')
        folded = clean.casefold()
        if not clean or folded in seen:
            return
        seen.add(folded)
        anchors.append(f'"{clean}"' if quoted else clean)

    for value in analysis.exact_terms[:2]:
        add(value, quoted=True)

    numeric = [
        token
        for token in analysis.tokens
        if token.kind in {"number", "exact"}
        and _ANCHOR_NUMBER.fullmatch(token.normalized)
    ]
    for token in numeric[:3]:
        add(token.surface)

    content = [
        token
        for token in analysis.tokens
        if token.kind == "word"
        and token.weight >= 0.5
        and (
            (token.script == "han" and len(token.normalized) >= 3)
            or (token.script != "han" and len(token.normalized) >= 4)
        )
    ]
    content.sort(key=lambda token: (-len(token.normalized), token.start))
    for token in content[: 2 if analysis.exact_terms else 5]:
        add(token.surface)

    if not anchors:
        return ""
    lane = " ".join([*anchors[:8], f"site:{host}"])

    def normalize(value: str) -> str:
        return " ".join(str(value or "").casefold().split())

    if normalize(lane) == normalize(model_query):
        return ""
    return lane


def select_source_channels(
    query: str,
    queries: Sequence[str] = (),
    *,
    scorer: PairScorer | None = None,
) -> Tuple[str, ...]:
    visible = " ".join([query, *queries])
    return SourceSelector(scorer=scorer).select(
        visible,
        tuple(_CHANNELS),
        _CHANNELS,
        max_optional=1,
    )


def source_channel_query(query: str, channel: str) -> str:
    """Use one compiler for all channels; adapters may encode API syntax."""

    if channel not in _CHANNELS:
        raise ValueError(f"unsupported source channel: {channel}")
    return compact_general_query(query)


def primary_source_requested(
    query: str,
    queries: Sequence[str] = (),
    *,
    scorer: PairScorer | None = None,
) -> bool:
    visible = " ".join([query, *queries])
    return bool(
        SourceSelector(scorer=scorer).select(
            visible,
            ("primary",),
            _PRIMARY,
            max_optional=1,
        )
    )


def normalized_host(url: str) -> str:
    return DiscoveryLayer.normalized_host(url)


def organization_domain(value: str) -> str:
    return DiscoveryLayer.organization_domain(value)


def same_organization(url: str, domain: str) -> bool:
    return organization_domain(url) == organization_domain(domain)


def select_pivot_domains(
    query: str,
    queries: Sequence[str],
    candidates: Sequence[DiscoveredURL],
    *,
    max_domains: int = 2,
    scorer: PairScorer | None = None,
) -> List[str]:
    layer = DiscoveryLayer(
        scorer=scorer,
        intermediary_hosts=_INTERMEDIARY_HOSTS,
    )
    rows = [
        {
            "url": item.url,
            "title": item.title,
            "snippet": item.snippet,
            "candidate_score": item.candidate_score,
            "rrf_score": item.rrf_score,
        }
        for item in candidates
    ]
    return layer.select_pivot_domains(
        query,
        queries,
        rows,
        max_domains=max_domains,
    )


def build_pivot_queries(query: str, domains: Sequence[str]) -> List[str]:
    base = re.sub(r"(?:^|\s)site:[^\s]+", " ", query, flags=re.I)
    base = " ".join(base.split()).strip()
    return [f"site:{domain} {base}" for domain in domains if domain and base]


def merge_candidate_groups(
    initial: Sequence[DiscoveredURL],
    pivot: Sequence[DiscoveredURL],
    *,
    max_candidates: int,
) -> List[DiscoveredURL]:
    """Merge discovery stages while retaining stage and engine provenance."""

    merged: dict[str, DiscoveredURL] = {}
    for stage, items in (("initial", initial), ("domain_pivot", pivot)):
        for source in items:
            item = copy.deepcopy(source)
            url = canonicalize_url(item.url)
            if not url:
                continue
            item.url = url
            stages = list(dict.fromkeys([*item.discovery_stages, item.discovery_stage, stage]))
            item.discovery_stage = stages[0]
            item.discovery_stages = stages
            if stage == "domain_pivot":
                item.rrf_score += 0.03
            existing = merged.get(url)
            if existing is None:
                merged[url] = item
                continue
            existing.rrf_score = max(existing.rrf_score, item.rrf_score)
            existing.engine_score = max(existing.engine_score, item.engine_score)
            existing.engines = list(dict.fromkeys([*existing.engines, *item.engines, item.engine]))
            existing.positions = list(dict.fromkeys([*existing.positions, *item.positions]))
            existing.matched_queries = list(
                dict.fromkeys([*existing.matched_queries, *item.matched_queries])
            )
            existing.source_channels = list(
                dict.fromkeys([*existing.source_channels, *item.source_channels])
            )
            existing.discovery_stages = list(
                dict.fromkeys([*existing.discovery_stages, *item.discovery_stages])
            )
            for matched_query, position in item.query_positions.items():
                previous = existing.query_positions.get(matched_query)
                existing.query_positions[matched_query] = min(previous or position, position)
            if len(item.snippet) > len(existing.snippet):
                existing.snippet = item.snippet
            if len(item.title) > len(existing.title):
                existing.title = item.title
            if len(item.cached_text) > len(existing.cached_text):
                existing.cached_text = item.cached_text
                existing.cached_text_mode = item.cached_text_mode
    ordered = list(merged.values())
    ordered.sort(key=lambda item: (item.rrf_score, -int(item.rank or 10**6)), reverse=True)
    return ordered[: max(0, max_candidates)]


def merge_query_candidate_groups(
    groups: Sequence[Tuple[str, Sequence[DiscoveredURL]]],
    *,
    max_candidates: int,
) -> List[DiscoveredURL]:
    """Fuse independently executed model queries using canonical URL RRF."""

    merged: dict[str, DiscoveredURL] = {}
    for query_index, (query, candidates) in enumerate(groups):
        stage = "initial" if query_index == 0 else "model_feedback"
        for position, source in enumerate(candidates, 1):
            canonical = canonicalize_url(source.url)
            if not canonical:
                continue
            item = copy.deepcopy(source)
            contribution = 1.0 / (60.0 + position) + 0.002 / (query_index + 1)
            existing = merged.get(canonical)
            if existing is None:
                item.url = canonical
                item.rank = position
                item.rrf_score = contribution
                item.matched_queries = list(dict.fromkeys([*item.matched_queries, query]))
                item.query_positions = {**item.query_positions, query: position}
                item.discovery_stage = stage
                item.discovery_stages = list(dict.fromkeys([*item.discovery_stages, stage]))
                item.engines = list(dict.fromkeys([*item.engines, item.engine]))
                merged[canonical] = item
                continue
            existing.rrf_score += contribution
            existing.rank = min(existing.rank or position, position)
            existing.engine_score = max(existing.engine_score, item.engine_score)
            existing.engines = list(
                dict.fromkeys([*existing.engines, *item.engines, item.engine])
            )
            existing.matched_queries = list(
                dict.fromkeys([*existing.matched_queries, query])
            )
            existing.query_positions[query] = min(
                existing.query_positions.get(query, position), position
            )
            existing.discovery_stages = list(
                dict.fromkeys([*existing.discovery_stages, stage])
            )
            if len(item.title) > len(existing.title):
                existing.title = item.title
            if len(item.snippet) > len(existing.snippet):
                existing.snippet = item.snippet
            if len(item.cached_text) > len(existing.cached_text):
                existing.cached_text = item.cached_text
                existing.cached_text_mode = item.cached_text_mode
    ordered = sorted(
        merged.values(),
        key=lambda item: (item.rrf_score, -int(item.rank or 10**6), item.url),
        reverse=True,
    )
    return ordered[: max(0, max_candidates)]


def discover_one_hop_links(
    query: str,
    queries: Sequence[str],
    documents: Sequence[RealtimeDocument],
    *,
    allowed_domains: Sequence[str],
    seen_urls: Iterable[str] = (),
    max_links: int = 8,
    scorer: PairScorer | None = None,
) -> List[DiscoveredURL]:
    layer = DiscoveryLayer(scorer=scorer)
    pages = [
        {
            "url": document.url,
            "title": document.title,
            "content": document.text[:3000],
            "links": list(document.links),
        }
        for document in documents
    ]
    rows = layer.select_one_hop_links(
        query,
        queries,
        pages,
        allowed_domains=allowed_domains,
        seen_urls=seen_urls,
        max_links=max_links,
    )
    output: List[DiscoveredURL] = []
    for position, row in enumerate(rows, 1):
        url = str(row.get("uri") or "")
        if not url:
            continue
        output.append(
            DiscoveredURL(
                url=url,
                title=str(row.get("title") or url),
                snippet=str(row.get("content") or "")[:1800],
                engine="page_link",
                rank=position,
                rrf_score=0.04,
                engines=["page_link"],
                matched_queries=[queries[0] if queries else query],
                query_positions={queries[0] if queries else query: position},
                discovery_stage="one_hop_link",
                discovery_stages=["one_hop_link"],
                parent_url=str(row.get("parent_url") or ""),
            )
        )
    return output
