from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from ..semantic_selection import PairScorer, select_diverse_items
from ..text import canonicalize_url, search_tokens
from .source_selector import SourceCapability, SourceSelector


_ASSET = re.compile(
    r"\.(?:avif|css|gif|ico|jpe?g|js|json|mp3|mp4|png|svg|webm|webp|woff2?)(?:$|\?)",
    re.I,
)
_NON_CONTENT_PATH = re.compile(
    r"(?:^|[/_.-])(?:login|signin|sign-in|captcha|search|tag|category|author|"
    r"privacy|terms|cookies?|account)(?:[/_.?&=\s-]|$)",
    re.I,
)
_COMMON_SECOND_LEVEL = frozenset({"ac", "co", "com", "edu", "gov", "net", "org"})
_INSTITUTIONAL_SUFFIXES = (
    ".gov",
    ".gov.cn",
    ".gov.uk",
    ".edu",
    ".edu.cn",
    ".ac.uk",
    ".europa.eu",
    ".int",
)
_ALIGNMENT_NOISE = frozenset(
    {
        "a", "an", "and", "are", "current", "find", "for", "from", "in",
        "is", "latest", "new", "news", "official", "of", "on", "recent",
        "release", "report", "search", "site", "source", "the", "to", "what",
        "官网", "官方", "当前", "最新", "最近", "发布", "报告", "数据", "信息",
        "国家", "政府", "网站",
    }
)


class DiscoveryLayer:
    """Bounded source, domain-pivot, and one-hop discovery primitives."""

    def __init__(
        self,
        *,
        scorer: PairScorer | None = None,
        intermediary_hosts: Iterable[str] = (),
    ) -> None:
        self.scorer = scorer
        self.source_selector = SourceSelector(scorer=scorer)
        self.intermediary_hosts = frozenset(
            self.normalized_host(value) for value in intermediary_hosts if value
        )

    def select_sources(
        self,
        query: str,
        configured: Sequence[str],
        capabilities: Mapping[str, SourceCapability],
        *,
        max_optional: int = 1,
    ) -> tuple[str, ...]:
        return self.source_selector.select(
            query,
            configured,
            capabilities,
            max_optional=max_optional,
        )

    def select_pivot_domains(
        self,
        query: str,
        query_views: Sequence[str],
        candidates: Sequence[Mapping[str, object]],
        *,
        max_domains: int = 2,
    ) -> list[str]:
        if max_domains <= 0:
            return []
        query_aliases = self._aliases([query, *query_views])
        domain_rows: dict[str, dict[str, object]] = {}
        for position, candidate in enumerate(candidates, 1):
            url = str(candidate.get("url") or candidate.get("uri") or "")
            host = self.normalized_host(url)
            domain = self.organization_domain(host)
            if not domain or self._intermediary(host):
                continue
            host_aliases = self._aliases([domain])
            title_aliases = self._aliases(
                [
                    str(candidate.get("title") or ""),
                    str(candidate.get("snippet") or candidate.get("content") or ""),
                ]
            )
            aligned = bool(query_aliases.intersection(host_aliases))
            institutional = host.endswith(_INSTITUTIONAL_SUFFIXES)
            overlap = query_aliases.intersection(title_aliases)
            institutional_alignment = institutional and (
                len(overlap) >= 2
                or any(value.isascii() and len(value) >= 3 for value in overlap)
            )
            if not (aligned or institutional_alignment):
                continue
            row = {
                "title": str(candidate.get("title") or domain),
                "content": str(candidate.get("snippet") or candidate.get("content") or ""),
                "uri": url,
                "domain": domain,
                "institutional": institutional,
                "_best_position": position,
                "_upstream_score": float(
                    candidate.get("candidate_score")
                    or candidate.get("score")
                    or candidate.get("rrf_score")
                    or 0.0
                ),
            }
            existing = domain_rows.get(domain)
            if existing is None or float(row["_upstream_score"]) > float(existing["_upstream_score"]):
                domain_rows[domain] = row
        institutional_labels = {
            domain.split(".", 1)[0]
            for domain, row in domain_rows.items()
            if bool(row.get("institutional"))
        }
        rows = [
            row
            for domain, row in domain_rows.items()
            if bool(row.get("institutional"))
            or domain.split(".", 1)[0] not in institutional_labels
        ]
        selection = select_diverse_items(
            query,
            query_views,
            rows,
            limit=max_domains,
            scorer=self.scorer,
            domain_weight=0.0,
        )
        return [str(item.get("domain") or "") for item in selection.items if item.get("domain")]

    def select_one_hop_links(
        self,
        query: str,
        query_views: Sequence[str],
        pages: Sequence[Mapping[str, object]],
        *,
        allowed_domains: Sequence[str],
        seen_urls: Iterable[str] = (),
        max_links: int = 8,
    ) -> list[dict[str, object]]:
        allowed = {self.organization_domain(value) for value in allowed_domains if value}
        seen = {canonicalize_url(str(value)) for value in seen_urls}
        pool: list[dict[str, object]] = []
        dedup: set[str] = set()
        for page in pages:
            parent = str(page.get("url") or page.get("uri") or "")
            parent_domain = self.organization_domain(parent)
            if parent_domain not in allowed:
                continue
            for position, raw in enumerate(page.get("links") or (), 1):
                url = canonicalize_url(str(raw or ""))
                if not url or url in seen or url in dedup or url == canonicalize_url(parent):
                    continue
                if self.organization_domain(url) != parent_domain:
                    continue
                parsed = urlsplit(url)
                path_query = unquote(f"{parsed.path} {parsed.query}").casefold()
                if parsed.path in {"", "/"} or _ASSET.search(path_query) or _NON_CONTENT_PATH.search(path_query):
                    continue
                dedup.add(url)
                pool.append(
                    {
                        "title": unquote(parsed.path).replace("/", " ").strip() or url,
                        "content": path_query,
                        "uri": url,
                        "parent_url": parent,
                        "_best_position": position,
                        "_upstream_score": 1.0 / max(1, position),
                    }
                )
        selection = select_diverse_items(
            query,
            query_views,
            pool,
            limit=max_links,
            scorer=self.scorer,
        )
        return [dict(item) for item in selection.items]

    def _intermediary(self, host: str) -> bool:
        return any(host == value or host.endswith("." + value) for value in self.intermediary_hosts)

    @staticmethod
    def normalized_host(value: str) -> str:
        host = (urlsplit(value).hostname or value).casefold().strip(".")
        return host[4:] if host.startswith("www.") else host

    @classmethod
    def organization_domain(cls, value: str) -> str:
        host = cls.normalized_host(value)
        labels = host.split(".")
        if len(labels) <= 2:
            return host
        keep = 3 if len(labels[-1]) == 2 and labels[-2] in _COMMON_SECOND_LEVEL else 2
        selected = labels[-keep:]
        if selected and selected[0] == "www":
            selected = selected[1:]
        return ".".join(selected)

    @staticmethod
    def _aliases(values: Iterable[str]) -> set[str]:
        aliases: set[str] = set()
        latin: list[str] = []
        for token in search_tokens(" ".join(str(value) for value in values)):
            folded = token.casefold()
            if folded in _ALIGNMENT_NOISE:
                continue
            if len(folded) >= 2:
                aliases.add(folded)
            compact = re.sub(r"[^a-z0-9]+", "", folded)
            if len(compact) >= 2:
                aliases.add(compact)
            if folded.isascii():
                parts = [
                    part
                    for part in re.split(r"[^a-z0-9]+", folded)
                    if len(part) >= 2
                ]
                aliases.update(parts)
                latin.extend(parts)
        aliases.update(
            latin[index] + latin[index + 1]
            for index in range(len(latin) - 1)
            if len(latin[index] + latin[index + 1]) >= 4
        )
        return aliases
