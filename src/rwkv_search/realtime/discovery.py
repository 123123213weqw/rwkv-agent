from __future__ import annotations

import asyncio
import base64
import copy
import html
import json
import os
from pathlib import Path
import re
import sqlite3
import zlib
from html.parser import HTMLParser
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import parse_qs, quote, urlsplit

from ..config import RealtimeSearchConfig
from ..text import canonicalize_url
from .cache import TTLByteCache
from .local_discovery import LocalIndexDiscovery
from .precision_discovery import source_channel_query
from .source_api import SourceAPIDiscovery
from ..semantic_selection import PairScorer
from .types import DiscoveredURL


_URL_RE = re.compile(r"https?://[^\s<>\]\[\"']+", re.I)
_VERSION_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff]")
_KOREAN_RE = re.compile(r"[\uac00-\ud7af]")
_WIKIPEDIA_TERM = re.compile(r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)*")
_WIKIPEDIA_STOP = frozenset(
    {
        "about", "after", "also", "and", "answer", "are", "as", "at",
        "before", "by", "did", "do", "does", "for", "from", "had", "has",
        "have", "how", "in", "into", "is", "it", "of", "on", "or", "question",
        "scope", "search", "site", "that", "the", "their", "this", "to", "was",
        "website", "were", "what", "when", "where", "which", "who", "why", "with",
    }
)


def _local_only_candidate(item: DiscoveredURL) -> bool:
    engines = {item.engine, *item.engines}
    engines.discard("")
    return engines == {"local_index"}


def searxng_search_params(
    query: str,
    freshness: str,
    source_channels: Sequence[str] = (),
    engines: Sequence[str] = (),
) -> Dict[str, str]:
    """Build the small, deterministic SearXNG API contract used at runtime.

    Do not apply a global ``time_range`` to general web discovery.  SearXNG
    excludes every engine that does not advertise time-range support when this
    parameter is present, which removes useful primary-source engines such as
    GitHub from otherwise fresh searches.  The planner query and later
    publication-time ranking carry freshness without reducing engine coverage.
    """
    if _JAPANESE_RE.search(query):
        language = "ja"
    elif _KOREAN_RE.search(query):
        language = "ko"
    elif _CJK_RE.search(query):
        language = "zh-CN"
    else:
        language = "en"
    params = {
        "q": query,
        "format": "json",
        "language": language,
        "safesearch": "1",
    }
    # Internal source channels describe query formation, not SearXNG category
    # names. Leaking values such as ``repos`` into ``categories`` silently
    # disables otherwise useful engines on a stock SearXNG installation.
    # Runtime engine selection is explicit and bounded so one suspended or
    # rate-limited engine cannot make the whole metasearch request unstable.
    selected_engines = list(
        dict.fromkeys(value.strip() for value in engines if value.strip())
    )
    if selected_engines:
        params["engines"] = ",".join(selected_engines)
    return params


def searxng_engines_for_query(
    config: RealtimeSearchConfig,
    query: str,
) -> tuple[str, ...]:
    """Select base lanes plus a bounded writing-system-specific lane set."""

    if _JAPANESE_RE.search(query):
        language = "ja"
    elif _KOREAN_RE.search(query):
        language = "ko"
    elif _CJK_RE.search(query):
        language = "zh"
    else:
        language = "default"
    additions = config.searxng_language_engines.get(language, ())
    if not additions and language != "default":
        additions = config.searxng_language_engines.get("default", ())
    return tuple(
        dict.fromkeys(
            value.strip()
            for value in (*config.searxng_engines, *additions)
            if value.strip()
        )
    )


def bing_search_params(query: str) -> Dict[str, str]:
    """Build the Bing Web HTML parameters supported by SearXNG's engine.

    Bing Web HTML does not expose a dependable generic time-range parameter.
    Freshness therefore stays explicit in the planner query instead of being
    represented by a no-op URL option.
    """
    if _JAPANESE_RE.search(query):
        market = "ja-JP"
    elif _KOREAN_RE.search(query):
        market = "ko-KR"
    elif _CJK_RE.search(query):
        market = "zh-CN"
    else:
        market = "en-US"
    return {"q": query, "count": "30", "adlt": "moderate", "mkt": market}


def bing_search_headers(query: str) -> Dict[str, str]:
    market = bing_search_params(query)["mkt"]
    language = market.split("-", 1)[0]
    return {"Accept-Language": f"{market},{language};q=0.9"}


class _SearchHTMLParser(HTMLParser):
    def __init__(self, engine: str) -> None:
        super().__init__(convert_charrefs=True)
        self.engine = engine
        self.results: List[DiscoveredURL] = []
        self._capture = False
        self._href = ""
        self._parts: List[str] = []
        self._depth = 0
        self._in_bing_result = False
        self._snippet_capture = False
        self._snippet_parts: List[str] = []

    def handle_starttag(
        self, tag: str, attrs: Sequence[tuple[str, Optional[str]]]
    ) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        lowered = tag.lower()
        if (
            self.engine == "bing"
            and lowered == "li"
            and "b_algo" in values.get("class", "")
        ):
            self._in_bing_result = True
        if (
            self.engine == "so360"
            and lowered == "li"
            and "res-list" in values.get("class", "")
        ):
            self._in_bing_result = True
        if (
            self.engine in {"bing", "so360"}
            and self._in_bing_result
            and lowered == "p"
        ):
            self._snippet_capture = True
            self._snippet_parts = []
            return
        if self._capture:
            self._depth += 1
            return
        if lowered != "a" or not values.get("href"):
            return
        classes = values.get("class", "")
        href = values.get("data-mdurl") or values["href"]
        if (
            self.engine == "bing"
            and self._in_bing_result
            and ("tilk" in classes or href.startswith("http"))
        ):
            self._capture, self._href, self._parts, self._depth = True, href, [], 0
        elif (
            self.engine == "so360"
            and self._in_bing_result
            and href.startswith("http")
        ):
            self._capture, self._href, self._parts, self._depth = True, href, [], 0
        elif self.engine == "baidu" and href.startswith("http"):
            self._capture, self._href, self._parts, self._depth = True, href, [], 0

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if self._capture:
            if self._depth:
                self._depth -= 1
                return
            if lowered == "a":
                title = " ".join(self._parts).strip()
                url = _result_url(html.unescape(self._href), self.engine)
                if url and title and not _is_search_internal(url):
                    self.results.append(
                        DiscoveredURL(url=url, title=title[:500], engine=self.engine)
                    )
                self._capture, self._href, self._parts = False, "", []
            return
        if self._snippet_capture and lowered == "p":
            if self.results:
                self.results[-1].snippet = " ".join(self._snippet_parts)[:1500]
            self._snippet_capture = False
            self._snippet_parts = []
            return
        if self.engine in {"bing", "so360"} and lowered == "li":
            self._in_bing_result = False

    def handle_data(self, data: str) -> None:
        if self._capture and data.strip():
            self._parts.append(data.strip())
        elif self._snippet_capture and data.strip():
            self._snippet_parts.append(data.strip())


def _is_search_internal(url: str) -> bool:
    host = (urlsplit(url).hostname or "").casefold()
    return host.endswith(
        ("bing.com", "bing.net", "baidu.com", "so.com", "searx.space")
    )


def _result_url(href: str, engine: str) -> Optional[str]:
    """Unwrap Bing's base64 click URL without spending a redirect request."""
    value = html.unescape(href)
    parsed = urlsplit(value)
    host = (parsed.hostname or "").casefold()
    if engine == "bing" and host.endswith("bing.com"):
        encoded = (parse_qs(parsed.query).get("u") or [""])[0]
        if encoded.startswith("a1"):
            payload = encoded[2:]
            try:
                payload += "=" * (-len(payload) % 4)
                value = base64.urlsafe_b64decode(payload).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                return None
    return canonicalize_url(value)


def parse_search_html(raw_html: str, engine: str) -> List[DiscoveredURL]:
    """Extract result links with selectolax, with a stdlib-only fallback."""
    output: List[DiscoveredURL] = []
    try:
        from selectolax.parser import HTMLParser as SelectolaxParser  # type: ignore

        tree = SelectolaxParser(raw_html)
        seen = set()
        if engine == "bing":
            containers = tree.css("li.b_algo")
            for container in containers:
                node = container.css_first("h2 a") or container.css_first("a.tilk")
                if node is None:
                    continue
                href = node.attributes.get("href", "")
                url = _result_url(href, engine)
                title = " ".join(node.text(separator=" ", strip=True).split())
                if url and title and url not in seen and not _is_search_internal(url):
                    seen.add(url)
                    paragraph = container.css_first("p")
                    snippet = (
                        " ".join(paragraph.text(separator=" ", strip=True).split())
                        if paragraph is not None
                        else ""
                    )
                    output.append(
                        DiscoveredURL(
                            url=url,
                            title=title[:500],
                            snippet=snippet[:1500],
                            engine=engine,
                        )
                    )
        elif engine == "so360":
            containers = tree.css("li.res-list")
            for container in containers:
                node = container.css_first("h3 a")
                if node is None:
                    continue
                href = (
                    node.attributes.get("data-mdurl", "")
                    or node.attributes.get("href", "")
                )
                url = _result_url(href, engine)
                title = " ".join(node.text(separator=" ", strip=True).split())
                if url and title and url not in seen and not _is_search_internal(url):
                    seen.add(url)
                    paragraph = container.css_first("p.res-desc")
                    snippet = (
                        " ".join(paragraph.text(separator=" ", strip=True).split())
                        if paragraph is not None
                        else ""
                    )
                    output.append(
                        DiscoveredURL(
                            url=url,
                            title=title[:500],
                            snippet=snippet[:1500],
                            engine=engine,
                        )
                    )
        else:
            for selector in ("div.result h3 a", "div.c-container h3 a"):
                for node in tree.css(selector):
                    href = node.attributes.get("href", "")
                    url = _result_url(href, engine)
                    title = " ".join(node.text(separator=" ", strip=True).split())
                    if (
                        url
                        and title
                        and url not in seen
                        and not _is_search_internal(url)
                    ):
                        seen.add(url)
                        output.append(
                            DiscoveredURL(url=url, title=title[:500], engine=engine)
                        )
        if output:
            return output
    except ImportError:
        pass
    parser = _SearchHTMLParser(engine)
    parser.feed(raw_html)
    return parser.results


def parse_wikipedia_results(value: Mapping[str, Any]) -> List[DiscoveredURL]:
    """Convert the public MediaWiki search API into discovery candidates."""

    output: List[DiscoveredURL] = []
    query = value.get("query")
    search = query.get("search", []) if isinstance(query, Mapping) else []
    if not isinstance(search, list):
        return output
    for item in search:
        if not isinstance(item, Mapping):
            continue
        title = " ".join(str(item.get("title") or "").split())
        if not title:
            continue
        raw_snippet = re.sub(r"<[^>]+>", " ", str(item.get("snippet") or ""))
        snippet = " ".join(html.unescape(raw_snippet).split())
        output.append(
            DiscoveredURL(
                url="https://en.wikipedia.org/wiki/"
                + quote(title.replace(" ", "_")),
                title=title[:500],
                snippet=snippet[:1500],
                engine="wikipedia",
            )
        )
    return output


def _wikipedia_snippet(text: str, query_terms: Sequence[str]) -> str:
    passages = [" ".join(value.split()) for value in str(text or "").splitlines()]
    passages = [value for value in passages if value]
    if not passages:
        return ""
    wanted = set(query_terms)
    ranked = sorted(
        (
            (
                sum(1 for term in wanted if term in value.casefold()),
                -index,
                value,
            )
            for index, value in enumerate(passages)
        ),
        reverse=True,
    )
    selected = [value for score, _index, value in ranked if score > 0][:3]
    if not selected:
        selected = passages[:2]
    return " ".join(selected)[:1800]


def search_local_wikipedia(
    database: str | Path,
    query: str,
    *,
    limit: int = 20,
) -> List[DiscoveredURL]:
    """Search the immutable local title index and return grounded snippets."""

    path = Path(database).expanduser().resolve()
    if not path.is_file():
        return []
    terms = []
    for match in _WIKIPEDIA_TERM.finditer(str(query or "")):
        term = match.group(0).casefold().strip(".-'")
        if len(term) < 3 or term in _WIKIPEDIA_STOP or term in terms:
            continue
        terms.append(term)
        if len(terms) >= 14:
            break
    if not terms:
        return []
    expression = " OR ".join('"' + value.replace('"', '""') + '"' for value in terms)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
    connection.execute("PRAGMA query_only=ON")
    try:
        rows = connection.execute(
            """
            SELECT pages.title, pages.url, pages.content_zlib, bm25(titles, 8.0)
            FROM titles JOIN pages ON pages.rowid=titles.rowid
            WHERE titles MATCH ?
            ORDER BY bm25(titles, 8.0)
            LIMIT ?
            """,
            (expression, max(1, int(limit))),
        ).fetchall()
    finally:
        connection.close()
    output: List[DiscoveredURL] = []
    for rank, (title, url, content, score) in enumerate(rows, 1):
        try:
            text = zlib.decompress(content).decode("utf-8", "replace")
        except (TypeError, zlib.error):
            text = ""
        output.append(
            DiscoveredURL(
                url=str(url),
                title=str(title)[:500],
                snippet=_wikipedia_snippet(text, terms),
                engine="wikipedia_local",
                rank=rank,
                rrf_score=1.0 / (40.0 + rank) + max(0.0, -float(score)) * 1e-4,
            )
        )
    return output


def parse_searxng_results(data: Mapping[str, Any]) -> List[DiscoveredURL]:
    """Preserve SearXNG's merged engines, positions, and result-container score."""
    output: List[DiscoveredURL] = []
    for value in data.get("results", [])[:50]:
        if not isinstance(value, dict):
            continue
        url = canonicalize_url(str(value.get("url") or ""))
        if not url:
            continue
        output.append(
            DiscoveredURL(
                url=url,
                title=str(value.get("title") or "")[:500],
                snippet=str(value.get("content") or "")[:1500],
                engine=str(value.get("engine") or "searxng"),
                published_hint=str(value.get("publishedDate") or "") or None,
                engine_score=float(value.get("score") or 0.0),
                engines=[str(item) for item in value.get("engines", ()) if item],
                positions=[
                    int(item)
                    for item in value.get("positions", ())
                    if isinstance(item, (int, float)) or str(item).isdigit()
                ],
            )
        )
    return output


class URLDiscovery:
    def __init__(
        self,
        config: RealtimeSearchConfig,
        session: object,
        cache: Optional[TTLByteCache[List[DiscoveredURL]]] = None,
        semantic_scorer: PairScorer | None = None,
        local_discovery: LocalIndexDiscovery | None = None,
    ) -> None:
        self.config = config
        self.session = session
        self.cache = (
            cache
            if cache is not None
            else TTLByteCache[List[DiscoveredURL]](
                max(1024 * 1024, config.cache_max_bytes // 8)
            )
        )
        self.api_sources = SourceAPIDiscovery(
            config,
            session,
            semantic_scorer=semantic_scorer,
        )
        self.local_discovery = local_discovery or LocalIndexDiscovery(config)

    async def discover(
        self,
        queries: Sequence[str],
        *,
        freshness: str,
        max_candidates: int,
        diagnostics: Optional[List[Dict[str, str]]] = None,
        source_channels: Sequence[str] = (),
    ) -> List[DiscoveredURL]:
        clean_queries = [query.strip() for query in queries if query.strip()]
        channels = tuple(
            dict.fromkeys(value.strip() for value in source_channels if value.strip())
        )[:2]
        tasks = [
            self._discover_one(query, freshness, diagnostics, channels)
            for query in clean_queries
        ]
        groups = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        merged: Dict[str, DiscoveredURL] = {}
        for query_index, group in enumerate(groups):
            if isinstance(group, Exception):
                continue
            matched_query = clean_queries[query_index]
            for rank, item in enumerate(group, start=1):
                url = canonicalize_url(item.url)
                if not url:
                    continue
                bonus = 1.0 / (60.0 + rank) + 0.002 / (query_index + 1) + item.rrf_score
                existing = merged.get(url)
                if existing:
                    # Raw and rewritten query lanes are correlated views of
                    # one local index. Repeated local hits are not independent
                    # votes; summing them creates false consensus for generic
                    # pages such as years and numbers. Live engines retain the
                    # original multi-query RRF behaviour.
                    if _local_only_candidate(existing) and _local_only_candidate(item):
                        existing.rrf_score = max(existing.rrf_score, bonus)
                    else:
                        existing.rrf_score += bonus
                    existing.rank = min(existing.rank or rank, rank)
                    existing.engine_score = max(
                        existing.engine_score, item.engine_score
                    )
                    existing.engines = list(
                        dict.fromkeys(
                            [
                                *existing.engines,
                                existing.engine,
                                *item.engines,
                                item.engine,
                            ]
                        )
                    )
                    existing.positions = list(
                        dict.fromkeys([*existing.positions, *item.positions])
                    )
                    existing.matched_queries = list(
                        dict.fromkeys([*existing.matched_queries, matched_query])
                    )
                    previous = existing.query_positions.get(matched_query)
                    existing.query_positions[matched_query] = min(
                        previous or rank, rank
                    )
                    existing.source_channels = list(
                        dict.fromkeys(
                            [*existing.source_channels, *item.source_channels]
                        )
                    )
                    if len(item.snippet) > len(existing.snippet):
                        existing.snippet = item.snippet
                    if len(item.cached_text) > len(existing.cached_text):
                        existing.cached_text = item.cached_text
                        existing.cached_text_mode = item.cached_text_mode
                else:
                    item.url = url
                    item.rank = rank
                    item.rrf_score = bonus
                    item.engines = list(dict.fromkeys([*item.engines, item.engine]))
                    item.matched_queries = [matched_query]
                    item.query_positions = {matched_query: rank}
                    if not item.source_channels:
                        item.source_channels = list(channels)
                    item.discovery_stages = list(
                        dict.fromkeys([*item.discovery_stages, item.discovery_stage])
                    )
                    merged[url] = item
        focus_versions = set(_VERSION_RE.findall(" ".join(queries)))
        if focus_versions:
            for item in merged.values():
                haystack = f"{item.title} {item.url}".casefold()
                if any(version.casefold() in haystack for version in focus_versions):
                    item.rrf_score += 0.03
        return sorted(merged.values(), key=lambda item: item.rrf_score, reverse=True)[
            :max_candidates
        ]

    async def _discover_one(
        self,
        query: str,
        freshness: str,
        diagnostics: Optional[List[Dict[str, str]]] = None,
        source_channels: Sequence[str] = (),
    ) -> List[DiscoveredURL]:
        direct = self._direct_urls(query)
        # ``general`` is the same external request as the no-channel legacy
        # path. Normalizing it allows a Shadow arm to replay the exact cached
        # discovery response instead of doubling search-engine traffic.
        channel_key = ",".join(
            value for value in source_channels if value and value != "general"
        )
        engine_key = ",".join(searxng_engines_for_query(self.config, query))
        fallback_key = ",".join(self.config.fallback_engines)
        provider_key = ",".join(self.config.api_discovery_providers)
        local_key = (
            f"{int(self.config.local_discovery_enabled)}:"
            f"{self.config.local_discovery_endpoint}:"
            f"{','.join(f'{name}={index}' for name, index in sorted(self.config.local_discovery_indexes.items()))}"
        )
        key = (
            f"{self.config.searxng_url.rstrip('/')}\0{engine_key}\0"
            f"{fallback_key}\0{provider_key}\0{local_key}\0"
            f"{self.config.bing_base_url.rstrip('/')}\0"
            f"{freshness}\0{channel_key}\0{query.casefold()}"
        )
        cached = self.cache.get(key)
        if cached is not None:
            return direct + copy.deepcopy(cached)

        api_task = asyncio.create_task(
            self.api_sources.discover(query, diagnostics=diagnostics)
        )
        local_task = asyncio.create_task(
            self.local_discovery.discover(
                query,
                freshness=freshness,
                diagnostics=diagnostics,
            )
        )
        results: List[DiscoveredURL] = []
        if self.config.searxng_url.rstrip("/") and len(source_channels) > 1:
            channel_tasks = [
                self._discover_one(
                    source_channel_query(query, channel),
                    freshness,
                    diagnostics,
                    source_channels=(channel,),
                )
                for channel in source_channels
            ]
            channel_groups = await asyncio.gather(
                *channel_tasks, return_exceptions=True
            )
            merged_channels: Dict[str, DiscoveredURL] = {}
            for channel_index, group in enumerate(channel_groups):
                if isinstance(group, Exception):
                    continue
                channel = source_channels[channel_index]
                for rank, item in enumerate(group, 1):
                    item.source_channels = list(
                        dict.fromkeys([*item.source_channels, channel])
                    )
                    item.rrf_score += 1.0 / (60.0 + rank)
                    if channel != "general":
                        item.rrf_score += 0.006
                    existing = merged_channels.get(item.url)
                    if existing is None:
                        merged_channels[item.url] = item
                    elif item.rrf_score > existing.rrf_score:
                        item.source_channels = list(
                            dict.fromkeys(
                                [*existing.source_channels, *item.source_channels]
                            )
                        )
                        item.engines = list(
                            dict.fromkeys(
                                [
                                    *existing.engines,
                                    existing.engine,
                                    *item.engines,
                                    item.engine,
                                ]
                            )
                        )
                        item.discovery_stages = list(
                            dict.fromkeys(
                                [
                                    *existing.discovery_stages,
                                    existing.discovery_stage,
                                    *item.discovery_stages,
                                    item.discovery_stage,
                                ]
                            )
                        )
                        if len(existing.cached_text) > len(item.cached_text):
                            item.cached_text = existing.cached_text
                            item.cached_text_mode = existing.cached_text_mode
                        merged_channels[item.url] = item
                    else:
                        existing.source_channels = list(
                            dict.fromkeys(
                                [*existing.source_channels, *item.source_channels]
                            )
                        )
                        existing.engines = list(
                            dict.fromkeys(
                                [*existing.engines, *item.engines, item.engine]
                            )
                        )
                        existing.discovery_stages = list(
                            dict.fromkeys(
                                [
                                    *existing.discovery_stages,
                                    existing.discovery_stage,
                                    *item.discovery_stages,
                                    item.discovery_stage,
                                ]
                            )
                        )
                        if len(item.cached_text) > len(existing.cached_text):
                            existing.cached_text = item.cached_text
                            existing.cached_text_mode = item.cached_text_mode
            results = sorted(
                merged_channels.values(),
                key=lambda item: item.rrf_score,
                reverse=True,
            )
        else:
            results = await self._searxng(
                query, freshness, diagnostics, source_channels=source_channels
            )
        if not results:
            engines = [
                engine
                for engine in self.config.fallback_engines
                if engine in {"bing", "baidu", "so360", "wikipedia"}
            ]

            async def fallback(engine: str) -> tuple[str, List[DiscoveredURL]]:
                try:
                    return engine, await self._html_engine(query, engine)
                except Exception as exc:
                    if diagnostics is not None:
                        diagnostics.append(
                            {
                                "query": query,
                                "engine": engine,
                                "error_type": type(exc).__name__,
                                "message": str(exc)[:300],
                            }
                        )
                    return engine, []

            groups = await asyncio.gather(*(fallback(engine) for engine in engines))
            merged_fallbacks: Dict[str, DiscoveredURL] = {}
            for _engine, group in groups:
                for rank, item in enumerate(group, start=1):
                    item.rrf_score += 1.0 / (60.0 + rank)
                    item.engines = list(
                        dict.fromkeys([*item.engines, item.engine])
                    )
                    existing = merged_fallbacks.get(item.url)
                    if existing is None:
                        merged_fallbacks[item.url] = item
                    else:
                        existing.rrf_score += item.rrf_score
                        existing.engines = list(
                            dict.fromkeys([*existing.engines, *item.engines])
                        )
                        if len(item.snippet) > len(existing.snippet):
                            existing.snippet = item.snippet
                        if len(item.cached_text) > len(existing.cached_text):
                            existing.cached_text = item.cached_text
                            existing.cached_text_mode = item.cached_text_mode
            results = sorted(
                merged_fallbacks.values(),
                key=lambda item: item.rrf_score,
                reverse=True,
            )
        api_results, local_results = await asyncio.gather(api_task, local_task)
        if api_results or local_results:
            merged_sources: Dict[str, DiscoveredURL] = {}
            for group in (results, api_results, local_results):
                for rank, item in enumerate(group, start=1):
                    url = canonicalize_url(item.url)
                    if not url:
                        continue
                    item.url = url
                    item.rrf_score += 1.0 / (60.0 + rank)
                    item.engines = list(
                        dict.fromkeys([*item.engines, item.engine])
                    )
                    existing = merged_sources.get(url)
                    if existing is None:
                        merged_sources[url] = item
                        continue
                    existing.rrf_score += item.rrf_score
                    existing.engine_score = max(
                        existing.engine_score,
                        item.engine_score,
                    )
                    existing.engines = list(
                        dict.fromkeys([*existing.engines, *item.engines])
                    )
                    existing.discovery_stages = list(
                        dict.fromkeys(
                            [
                                *existing.discovery_stages,
                                existing.discovery_stage,
                                *item.discovery_stages,
                                item.discovery_stage,
                            ]
                        )
                    )
                    if len(item.snippet) > len(existing.snippet):
                        existing.snippet = item.snippet
                    if len(item.cached_text) > len(existing.cached_text):
                        existing.cached_text = item.cached_text
                        existing.cached_text_mode = item.cached_text_mode
                    if not existing.published_hint and item.published_hint:
                        existing.published_hint = item.published_hint
            results = sorted(
                merged_sources.values(),
                key=lambda item: item.rrf_score,
                reverse=True,
            )
        unique_results: List[DiscoveredURL] = []
        seen_urls = set()
        for item in results:
            url = canonicalize_url(item.url)
            if not url or url in seen_urls:
                continue
            item.url = url
            seen_urls.add(url)
            unique_results.append(item)
        results = unique_results
        for index, item in enumerate(results, start=1):
            item.rank = index
            if not item.source_channels:
                item.source_channels = list(source_channels)
        self.cache.put(
            key,
            copy.deepcopy(results),
            self.config.search_cache_ttl_seconds,
            size=sum(
                len(item.url) + len(item.title) + len(item.snippet) for item in results
            ),
        )
        return direct + results

    async def _searxng(
        self,
        query: str,
        freshness: str,
        diagnostics: Optional[List[Dict[str, str]]] = None,
        source_channels: Sequence[str] = (),
    ) -> List[DiscoveredURL]:
        engines = searxng_engines_for_query(self.config, query)
        if len(engines) <= 1:
            return await self._searxng_request(
                query,
                freshness,
                diagnostics,
                source_channels,
                engines=engines,
                diagnostic_engine="searxng",
            )

        # Fan out one bounded request per engine and fuse locally.  A combined
        # SearXNG request can lose every engine when one slow backend stalls
        # the response; independent lanes preserve the healthy engines and
        # also protect each engine's first page before RRF fusion.
        groups = await asyncio.gather(
            *(
                self._searxng_request(
                    query,
                    freshness,
                    diagnostics,
                    source_channels,
                    engines=(engine,),
                    diagnostic_engine=f"searxng:{engine}",
                )
                for engine in engines
            ),
            return_exceptions=True,
        )
        merged: Dict[str, DiscoveredURL] = {}
        for group in groups:
            if isinstance(group, Exception):
                continue
            for rank, item in enumerate(group, 1):
                url = canonicalize_url(item.url)
                if not url:
                    continue
                item.url = url
                item.rrf_score += 1.0 / (60.0 + rank)
                item.positions = list(dict.fromkeys([*item.positions, rank]))
                item.engines = list(
                    dict.fromkeys([*item.engines, item.engine])
                )
                existing = merged.get(item.url)
                if existing is None:
                    merged[item.url] = item
                    continue
                existing.rrf_score += item.rrf_score
                existing.engine_score = max(
                    existing.engine_score, item.engine_score
                )
                existing.engines = list(
                    dict.fromkeys([*existing.engines, *item.engines, item.engine])
                )
                existing.positions = list(
                    dict.fromkeys([*existing.positions, *item.positions])
                )
                if len(item.title) > len(existing.title):
                    existing.title = item.title
                if len(item.snippet) > len(existing.snippet):
                    existing.snippet = item.snippet
        return sorted(
            merged.values(),
            key=lambda item: (item.rrf_score, item.engine_score, item.url),
            reverse=True,
        )[:50]

    async def _searxng_request(
        self,
        query: str,
        freshness: str,
        diagnostics: Optional[List[Dict[str, str]]],
        source_channels: Sequence[str],
        *,
        engines: Sequence[str],
        diagnostic_engine: str,
    ) -> List[DiscoveredURL]:
        base = self.config.searxng_url.rstrip("/")
        if not base:
            return []
        lane_key = (
            "searxng_lane\0"
            f"{base}\0{','.join(engines)}\0{freshness}\0"
            f"{','.join(source_channels)}\0{query.casefold()}"
        )
        cached = self.cache.get(lane_key)
        if cached is not None:
            return copy.deepcopy(cached)
        params = searxng_search_params(
            query,
            freshness,
            source_channels,
            engines,
        )
        headers = {}
        if (urlsplit(base).hostname or "").casefold() in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            headers["X-Real-IP"] = "127.0.0.1"
        data: Mapping[str, Any] | None = None
        final_error: Exception | None = None
        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    self.session.get(  # type: ignore[attr-defined]
                        f"{base}/search",
                        params=params,
                        headers=headers,
                        allow_redirects=True,
                    ),
                    timeout=self.config.discovery_timeout_seconds,
                )
                async with response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}")
                    raw = await _read_limited(response.content, 2 * 1024 * 1024)
                    raw = _decode_http_body(
                        raw, response.headers.get("Content-Encoding", "")
                    )
                data = json.loads(raw.decode("utf-8", "replace"))
                break
            except Exception as exc:
                final_error = exc
                if attempt == 0:
                    # This is a retry against the local metasearch process,
                    # not an unbounded retry loop in the crawler.  It repairs
                    # transient keep-alive/read failures while keeping each
                    # engine lane capped at two requests.
                    await asyncio.sleep(0.05)
                    continue
        if data is None:
            if diagnostics is not None and final_error is not None:
                error_type = type(final_error).__name__
                message = str(final_error)[:300]
                if isinstance(final_error, RuntimeError) and message.startswith(
                    "HTTP "
                ):
                    error_type = "HTTPStatusError"
                diagnostics.append(
                    {
                        "query": query,
                        "engine": diagnostic_engine,
                        "source_channels": ",".join(source_channels),
                        "error_type": error_type,
                        "message": message,
                        "attempts": "2",
                    }
                )
            return []
        results = parse_searxng_results(data)
        if results:
            self.cache.put(
                lane_key,
                copy.deepcopy(results),
                self.config.search_cache_ttl_seconds,
                size=sum(
                    len(item.url) + len(item.title) + len(item.snippet)
                    for item in results
                ),
            )
        return results

    async def _html_engine(self, query: str, engine: str) -> List[DiscoveredURL]:
        if engine == "wikipedia":
            local_database = os.getenv("RWKV_AGENT_WIKIPEDIA_DB", "").strip()
            if local_database:
                return await asyncio.to_thread(
                    search_local_wikipedia,
                    local_database,
                    query,
                )
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": "20",
                "format": "json",
                "formatversion": "2",
                "utf8": "1",
            }
            headers: Dict[str, str] = {}
        elif engine == "baidu":
            url = "https://www.baidu.com/s"
            params = {"wd": query}
            headers: Dict[str, str] = {}
        elif engine == "so360":
            url = "https://www.so.com/s"
            params = {"q": query}
            headers = {}
        else:
            url = f"{self.config.bing_base_url.rstrip('/')}/search"
            params = bing_search_params(query)
            headers = bing_search_headers(query)
        response = await asyncio.wait_for(
            self.session.get(  # type: ignore[attr-defined]
                url,
                params=params,
                headers=headers,
                allow_redirects=True,
            ),
            timeout=self.config.discovery_timeout_seconds,
        )
        async with response:
            if response.status != 200:
                return []
            raw = await _read_limited(response.content, 2 * 1024 * 1024)
            raw = _decode_http_body(raw, response.headers.get("Content-Encoding", ""))
            charset = response.charset or "utf-8"
        decoded = raw.decode(charset, "replace")
        if engine == "wikipedia":
            try:
                value = json.loads(decoded)
            except json.JSONDecodeError:
                return []
            return parse_wikipedia_results(value)
        return parse_search_html(decoded, engine)

    @staticmethod
    def _direct_urls(query: str) -> List[DiscoveredURL]:
        output: List[DiscoveredURL] = []
        for value in _URL_RE.findall(query):
            url = canonicalize_url(value.rstrip(".,，。！？!?;；)）"))
            if url:
                output.append(DiscoveredURL(url=url, title=url, engine="direct"))
        return output


def _decode_http_body(raw: bytes, encoding: str) -> bytes:
    """Decode discovery responses because the shared session keeps raw bytes."""
    normalized = encoding.casefold().strip()
    try:
        if normalized in {"gzip", "x-gzip"}:
            value = zlib.decompress(raw, 16 + zlib.MAX_WBITS, 4 * 1024 * 1024)
        elif normalized == "deflate":
            value = zlib.decompress(raw, zlib.MAX_WBITS, 4 * 1024 * 1024)
        elif normalized in {"", "identity"}:
            value = raw
        else:
            return b""
    except zlib.error:
        return b""
    return value if len(value) <= 4 * 1024 * 1024 else b""


async def _read_limited(stream: object, limit: int) -> bytes:
    output = bytearray()
    async for chunk in stream.iter_chunked(64 * 1024):  # type: ignore[attr-defined]
        output.extend(chunk)
        if len(output) > limit:
            return b""
    return bytes(output)
