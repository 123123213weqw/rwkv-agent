from __future__ import annotations

import copy
import re
from typing import Iterable, List, Sequence, Tuple
from urllib.parse import unquote, urlsplit

from ..text import canonicalize_url, search_tokens
from .types import DiscoveredURL, RealtimeDocument


# These are source-shape intents, not topic/domain routing.  General search is
# always retained; a specialist SearXNG category is added only when the user
# explicitly asks for that kind of primary source.
_REPOSITORY_SOURCE_RE = re.compile(
    r"github|gitlab|codeberg|sourceforge|代码仓库|官方仓库|源码仓库|"
    r"\brepositor(?:y|ies)\b|\brepo\b|\bcommits?\b",
    re.I,
)
_SCIENCE_SOURCE_RE = re.compile(
    r"arxiv|doi\b|原始论文|官方论文|学术论文|预印本|"
    r"\b(?:original\s+)?papers?\b|\bpreprints?\b",
    re.I,
)
_REPOSITORY_QUERY_NOISE_RE = re.compile(
    r"github|gitlab|codeberg|sourceforge|代码仓库|官方仓库|源码仓库|仓库|"
    r"最近|最新|当前|更新记录|更新|发布页|发布|正式版本|版本|"
    r"\brepositor(?:y|ies)\b|\brepo\b|\bcommits?\b|\bofficial\b|"
    r"\blatest\b|\brecent\b|\breleases?\b|\bversions?\b|\bchanges?\b",
    re.I,
)
_SCIENCE_QUERY_NOISE_RE = re.compile(
    r"原始论文|官方论文|学术论文|论文页面|论文|预印本|查找|搜索|官方|"
    r"arxiv|doi\b|\b(?:original\s+)?papers?\b|\bpreprints?\b|"
    r"\bofficial\b|\bfind\b|\bsearch\b",
    re.I,
)
_PRIMARY_SOURCE_RE = re.compile(
    r"官方|官网|原文|一手来源|原始来源|只看|为准|公布|发布|发布说明|公告|财报|"
    r"投资者关系|政策文件|论文|仓库|新闻室|"
    r"\bofficial\b|\bprimary source\b|\boriginal source\b|"
    r"\baccording to\b|\bfrom\b|\binvestor relations?\b|\bnewsroom\b|"
    r"\brelease notes?\b|\bfilings?\b|\bform 10-[qk]\b",
    re.I,
)

_PLATFORM_HOSTS = {
    "github.com",
    "gitlab.com",
    "codeberg.org",
    "arxiv.org",
}
_INTERMEDIARY_HOSTS = {
    "baidu.com",
    "bing.com",
    "facebook.com",
    "google.com",
    "linkedin.com",
    "medium.com",
    "reddit.com",
    "so.com",
    "sogou.com",
    "wikipedia.org",
    "x.com",
    "youtube.com",
    "zhihu.com",
}
_COMMON_SECOND_LEVEL = {"ac", "co", "com", "edu", "gov", "net", "org"}
_TRUSTED_SUFFIXES = (
    ".gov",
    ".gov.cn",
    ".gov.uk",
    ".edu",
    ".edu.cn",
    ".ac.uk",
    ".europa.eu",
    ".int",
)
_OFFICIAL_VISIBLE_RE = re.compile(
    r"官方|官网|政府|委员会|统计局|研究院|official|government|"
    r"investor relations?|newsroom|documentation",
    re.I,
)
_CJK_SEQUENCE_RE = re.compile(r"[\u3400-\u9fff]{2,}")
_ENGLISH_QUERY_SHELL_RE = re.compile(
    r"^(?:(?:please|could you|can you)\s+)?(?:"
    r"find|search(?:\s+for)?|show|look\s+up|tell\s+me|"
    r"what\s+is|what\s+are"
    r")\s+",
    re.I,
)
_ENGLISH_QUERY_TOKEN_RE = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9_+#'-]*\.)+[A-Za-z]{2,}|"
    r"[A-Za-z][A-Za-z0-9_+#'.-]*|\d+(?:\.\d+)+"
)
_ENGLISH_QUERY_NOISE = {
    "a",
    "according",
    "an",
    "and",
    "are",
    "at",
    "current",
    "find",
    "for",
    "from",
    "in",
    "is",
    "its",
    "latest",
    "look",
    "most",
    "newest",
    "of",
    "official",
    "on",
    "please",
    "recent",
    "search",
    "show",
    "site",
    "source",
    "the",
    "to",
    "use",
    "website",
    "what",
    "with",
}
_ENGLISH_QUERY_MODIFIERS = {
    "current",
    "latest",
    "newest",
    "official",
    "recent",
}
_CJK_SCOPE_NOISE = {
    "当前",
    "最新",
    "最近",
    "目前",
    "官方",
    "官网",
    "发布",
    "公告",
    "数据",
    "报告",
    "原文",
    "版本",
    "政策",
    "文件",
    "季度",
    "年度",
    "什么",
    "查找",
    "搜索",
    "信息",
    "情况",
    "概况",
    "正式",
    "一期",
    "一个",
}


def compact_general_query(query: str) -> str:
    """Put a long English search request's subject before its chat shell.

    HTML search fallbacks and several metasearch engines are markedly less
    reliable when a natural-language request begins with ``Find`` or
    ``What is``. This bounded lexical compaction keeps names, domains, version
    strings and answer-bearing terms, then appends freshness/source modifiers.
    It does not classify the topic or inject a preferred domain.
    """

    value = " ".join(str(query or "").split()).strip()
    if not value or _CJK_SEQUENCE_RE.search(value):
        return value
    stripped = _ENGLISH_QUERY_SHELL_RE.sub("", value).strip(" ?.!,:;")
    tokens = _ENGLISH_QUERY_TOKEN_RE.findall(stripped)
    if len(tokens) < 5:
        return stripped or value

    priority: List[str] = []
    content: List[str] = []
    modifiers: List[str] = []
    seen = set()
    for token in tokens:
        folded = token.casefold().strip("'")
        if not folded or folded in seen:
            continue
        seen.add(folded)
        if folded in _ENGLISH_QUERY_MODIFIERS:
            modifiers.append(token)
            continue
        if folded in _ENGLISH_QUERY_NOISE:
            continue
        is_domain = "." in token and token.rsplit(".", 1)[-1].isalpha()
        is_acronym = token.isupper() and 1 < len(token) <= 12
        is_named = token[:1].isupper()
        target = priority if is_domain or is_acronym or is_named else content
        target.append(token)

    compacted = [*priority, *content, *modifiers]
    return " ".join(compacted[:14]) or stripped or value
_LATIN_SCOPE_NOISE = {
    "current",
    "latest",
    "recent",
    "official",
    "release",
    "releases",
    "report",
    "reports",
    "find",
    "search",
    "from",
    "according",
    "stable",
    "newest",
    "new",
}
_LINK_REJECT_RE = re.compile(
    r"(?:^|[/_.-])(?:login|signin|sign-in|captcha|search|tag|category|author|"
    r"privacy|terms|cookies?|account)(?:[/_.?&=-]|$)",
    re.I,
)
_ASSET_RE = re.compile(
    r"\.(?:avif|css|gif|ico|jpe?g|js|json|mp3|mp4|png|svg|webm|webp|woff2?)(?:$|\?)",
    re.I,
)

_PAGE_SHAPE_GROUPS: Tuple[Tuple[re.Pattern[str], frozenset[str]], ...] = (
    (
        re.compile(r"发布|版本|更新|release|version|changelog|lts", re.I),
        frozenset(
            {
                "release",
                "releases",
                "download",
                "downloads",
                "changelog",
                "version",
                "versions",
                "announcement",
            }
        ),
    ),
    (
        re.compile(r"公告|新闻|动态|announcement|news|press", re.I),
        frozenset(
            {
                "news",
                "newsroom",
                "press",
                "announcement",
                "announcements",
                "article",
                "articles",
                "blog",
            }
        ),
    ),
    (
        re.compile(
            r"财报|业绩|季度|年报|filing|earnings|financial|quarterly|annual report",
            re.I,
        ),
        frozenset(
            {
                "investor",
                "investors",
                "financial",
                "finance",
                "earnings",
                "results",
                "quarterly",
                "annual",
                "filing",
                "filings",
                "reports",
            }
        ),
    ),
    (
        re.compile(
            r"政策|法规|法案|声明|policy|regulation|act|statement|decision", re.I
        ),
        frozenset(
            {
                "policy",
                "policies",
                "regulation",
                "regulations",
                "law",
                "legal",
                "statement",
                "statements",
                "decision",
                "decisions",
                "documents",
            }
        ),
    ),
    (
        re.compile(r"数据|统计|报告|data|statistics|report", re.I),
        frozenset(
            {
                "data",
                "statistics",
                "report",
                "reports",
                "release",
                "releases",
                "publication",
                "publications",
            }
        ),
    ),
    (
        re.compile(r"论文|paper|preprint|arxiv|doi", re.I),
        frozenset(
            {
                "paper",
                "papers",
                "publication",
                "publications",
                "article",
                "articles",
                "abs",
                "pdf",
                "doi",
            }
        ),
    ),
    (
        re.compile(r"预警|通报|advisory|warning|alert|bulletin", re.I),
        frozenset(
            {
                "advisory",
                "advisories",
                "warning",
                "warnings",
                "alert",
                "alerts",
                "bulletin",
                "bulletins",
            }
        ),
    ),
)


def select_source_channels(query: str, queries: Sequence[str] = ()) -> Tuple[str, ...]:
    """Select at most one explicit specialist source in addition to general."""
    visible = " ".join([query, *queries])
    if _REPOSITORY_SOURCE_RE.search(visible):
        return ("general", "repos")
    if _SCIENCE_SOURCE_RE.search(visible):
        return ("general", "science")
    return ("general",)


def source_channel_query(query: str, channel: str) -> str:
    """Build the smallest source-native lookup without an LLM schema.

    Repository and paper engines work best when task/source words are removed
    and the subject expression is retained.  Latin project names are preferred
    when present; otherwise the cleaned multilingual text is used.
    """
    if channel == "repos":
        cleaned = _REPOSITORY_QUERY_NOISE_RE.sub(" ", query)
    elif channel == "science":
        cleaned = _SCIENCE_QUERY_NOISE_RE.sub(" ", query)
    else:
        return " ".join(query.split()).strip()
    cleaned = " ".join(cleaned.split()).strip(" -_:：,，。")
    latin = [
        token
        for token in search_tokens(cleaned)
        if token.isascii() and any(character.isalpha() for character in token)
    ]
    if latin:
        return " ".join(latin[:4])
    return cleaned or " ".join(query.split()).strip()


def primary_source_requested(query: str, queries: Sequence[str] = ()) -> bool:
    return bool(_PRIMARY_SOURCE_RE.search(" ".join([query, *queries])))


def normalized_host(url: str) -> str:
    host = (urlsplit(url).hostname or "").casefold().strip(".")
    return host[4:] if host.startswith("www.") else host


def organization_domain(value: str) -> str:
    """Return a conservative registrable-style parent without network I/O."""
    host = normalized_host(value) if "://" in value else value.casefold().strip(".")
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    keep = 3 if len(labels[-1]) == 2 and labels[-2] in _COMMON_SECOND_LEVEL else 2
    if keep == 3 and labels[-3] == "www":
        keep = 2
    return ".".join(labels[-keep:])


def same_organization(url: str, domain: str) -> bool:
    return organization_domain(url) == organization_domain(domain)


def _host_aliases(host: str) -> set[str]:
    aliases = set(search_tokens(host))
    aliases.update(label.casefold() for label in host.split(".") if len(label) >= 2)
    aliases.add(re.sub(r"[^a-z0-9]+", "", host.casefold()))
    return {value for value in aliases if len(value) >= 2}


def _query_aliases(values: Iterable[str]) -> set[str]:
    aliases = set()
    latin_tokens: List[str] = []
    for token in search_tokens(" ".join(values)):
        folded = token.casefold()
        aliases.add(folded)
        compact = re.sub(r"[^a-z0-9]+", "", folded)
        if len(compact) >= 2:
            aliases.add(compact)
        if (
            folded.isascii()
            and any(character.isalpha() for character in folded)
            and folded not in _LATIN_SCOPE_NOISE
        ):
            latin_tokens.append(compact or folded)
    aliases.update(
        latin_tokens[index] + latin_tokens[index + 1]
        for index in range(len(latin_tokens) - 1)
        if len(latin_tokens[index] + latin_tokens[index + 1]) >= 4
    )
    return aliases


def _cjk_scope_alignment(query: str, title: str) -> bool:
    query_tokens = {
        token
        for sequence in _CJK_SEQUENCE_RE.findall(query)
        for token in search_tokens(sequence)
        if len(token) == 2 and token not in _CJK_SCOPE_NOISE
    }
    title_tokens = {
        token
        for sequence in _CJK_SEQUENCE_RE.findall(title)
        for token in search_tokens(sequence)
        if len(token) == 2 and token not in _CJK_SCOPE_NOISE
    }
    if not query_tokens:
        return False
    overlap = query_tokens.intersection(title_tokens)
    required = 1 if len(query_tokens) == 1 else 2
    return len(overlap) >= required


def select_pivot_domains(
    query: str,
    queries: Sequence[str],
    candidates: Sequence[DiscoveredURL],
    *,
    max_domains: int = 2,
) -> List[str]:
    """Infer likely first-party organization domains from actual result pages.

    A domain must carry a generic trust signal: entity/domain alignment, an
    explicitly requested source platform, or an institutional suffix together
    with a matching Chinese organization title. Benchmark labels are never inputs.
    """
    channels = select_source_channels(query, queries)
    if max_domains <= 0 or not (
        primary_source_requested(query, queries) or len(channels) > 1
    ):
        return []
    visible_query = " ".join([query, *queries])
    query_aliases = _query_aliases([query, *queries])
    scored: List[Tuple[float, int, str, bool]] = []
    seen = set()
    for position, item in enumerate(candidates, 1):
        host = normalized_host(item.url)
        domain = organization_domain(host)
        if not host or not domain or domain in seen:
            continue
        if any(
            host == value or host.endswith("." + value) for value in _INTERMEDIARY_HOSTS
        ):
            continue
        host_aliases = _host_aliases(domain)
        aligned = bool(query_aliases.intersection(host_aliases))
        institutional = host.endswith(_TRUSTED_SUFFIXES) or host.endswith(".gov.cn")
        visible_official = bool(
            _OFFICIAL_VISIBLE_RE.search(f"{item.title} {item.snippet}")
        )
        cjk_aligned = _cjk_scope_alignment(visible_query, item.title)
        requested_platform = (
            "repos" in channels
            and domain in {"github.com", "gitlab.com", "codeberg.org"}
        ) or ("science" in channels and domain == "arxiv.org")
        if domain in _PLATFORM_HOSTS and not requested_platform:
            continue
        # A page merely saying "official" is not evidence that its host is the
        # first party.  It can increase confidence only after a domain/entity,
        # institutional-suffix, or explicitly requested platform signal has
        # admitted the organization scope.
        if not (aligned or requested_platform or (institutional and cjk_aligned)):
            continue
        score = (
            1.4 * float(aligned)
            + 1.0 * float(institutional)
            + 0.7 * float(requested_platform)
            + 1.1 * float(cjk_aligned)
            + 0.35 * float(visible_official)
            + 0.25 / max(1, position)
            + 0.2 * float(item.candidate_score)
        )
        seen.add(domain)
        scored.append((score, position, domain, institutional))
    scored.sort(key=lambda value: (-value[0], value[1], value[2]))
    institutional_labels = {
        domain.split(".", 1)[0] for _, _, domain, trusted in scored if trusted
    }
    filtered = [
        value
        for value in scored
        if value[3] or value[2].split(".", 1)[0] not in institutional_labels
    ]
    return [domain for _, _, domain, _ in filtered[: max(0, max_domains)]]


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
    """Merge discovery stages while retaining per-stage and engine provenance."""
    merged: dict[str, DiscoveredURL] = {}
    for stage, items in (("initial", initial), ("domain_pivot", pivot)):
        for source in items:
            item = copy.deepcopy(source)
            url = canonicalize_url(item.url)
            if not url:
                continue
            item.url = url
            stages = list(
                dict.fromkeys([*item.discovery_stages, item.discovery_stage, stage])
            )
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
            existing.engines = list(
                dict.fromkeys([*existing.engines, *item.engines, item.engine])
            )
            existing.positions = list(
                dict.fromkeys([*existing.positions, *item.positions])
            )
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
                existing.query_positions[matched_query] = min(
                    previous or position, position
                )
            if len(item.snippet) > len(existing.snippet):
                existing.snippet = item.snippet
            if len(item.title) > len(existing.title):
                existing.title = item.title
    ordered = list(merged.values())
    ordered.sort(
        key=lambda item: (item.rrf_score, -int(item.rank or 10**6)), reverse=True
    )
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
                item.matched_queries = list(
                    dict.fromkeys([*item.matched_queries, query])
                )
                item.query_positions = {**item.query_positions, query: position}
                item.discovery_stage = stage
                item.discovery_stages = list(
                    dict.fromkeys([*item.discovery_stages, stage])
                )
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
    ordered = sorted(
        merged.values(),
        key=lambda item: (item.rrf_score, -int(item.rank or 10**6), item.url),
        reverse=True,
    )
    return ordered[: max(0, max_candidates)]


def _requested_page_shapes(query: str) -> set[str]:
    output: set[str] = set()
    for pattern, shapes in _PAGE_SHAPE_GROUPS:
        if pattern.search(query):
            output.update(shapes)
    return output


def discover_one_hop_links(
    query: str,
    queries: Sequence[str],
    documents: Sequence[RealtimeDocument],
    *,
    allowed_domains: Sequence[str],
    seen_urls: Iterable[str] = (),
    max_links: int = 8,
) -> List[DiscoveredURL]:
    """Score a bounded set of same-organization links from fetched pages."""
    if max_links <= 0 or not allowed_domains:
        return []
    allowed = {organization_domain(value) for value in allowed_domains if value}
    seen = {canonicalize_url(value) for value in seen_urls}
    query_tokens = _query_aliases([query, *queries])
    page_shapes = _requested_page_shapes(" ".join([query, *queries]))
    versions = set(re.findall(r"\b\d+(?:\.\d+){1,3}\b", " ".join([query, *queries])))
    scored: List[Tuple[float, int, DiscoveredURL]] = []
    dedup = set()
    serial = 0
    for document in documents:
        parent_domain = organization_domain(document.url)
        if parent_domain not in allowed:
            continue
        for link in document.links:
            canonical = canonicalize_url(link)
            if (
                not canonical
                or canonical in seen
                or canonical in dedup
                or canonical == canonicalize_url(document.url)
            ):
                continue
            if organization_domain(canonical) != parent_domain:
                continue
            parsed = urlsplit(canonical)
            path_query = unquote(f"{parsed.path} {parsed.query}").casefold()
            if (
                parsed.path in {"", "/"}
                or _ASSET_RE.search(path_query)
                or _LINK_REJECT_RE.search(path_query)
            ):
                continue
            link_tokens = _query_aliases([path_query])
            token_overlap = len(query_tokens.intersection(link_tokens)) / max(
                1, min(6, len(query_tokens))
            )
            shape_overlap = len(page_shapes.intersection(link_tokens)) / max(
                1, min(3, len(page_shapes))
            )
            version_match = any(
                version in path_query
                or version.replace(".", "-") in path_query
                or version.replace(".", "") in path_query
                for version in versions
            )
            depth = len([part for part in parsed.path.split("/") if part])
            detail_score = 1.0 if 1 <= depth <= 6 else 0.35
            score = (
                0.52 * token_overlap
                + 0.38 * shape_overlap
                + 0.35 * float(version_match)
                + 0.10 * detail_score
            )
            if score < 0.12:
                continue
            serial += 1
            dedup.add(canonical)
            scored.append(
                (
                    score,
                    serial,
                    DiscoveredURL(
                        url=canonical,
                        title=canonical,
                        engine="page_link",
                        rank=serial,
                        rrf_score=0.04 + min(0.04, score * 0.04),
                        engines=["page_link"],
                        matched_queries=[queries[0] if queries else query],
                        query_positions={queries[0] if queries else query: serial},
                        discovery_stage="one_hop_link",
                        discovery_stages=["one_hop_link"],
                        parent_url=document.url,
                    ),
                )
            )
    scored.sort(key=lambda value: (-value[0], value[1], value[2].url))
    return [item for _, _, item in scored[: max(0, max_links)]]
