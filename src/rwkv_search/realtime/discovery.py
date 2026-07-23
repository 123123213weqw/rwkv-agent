from __future__ import annotations

import asyncio
import base64
import copy
import html
import json
import re
import zlib
from html.parser import HTMLParser
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlsplit

from ..config import RealtimeSearchConfig
from ..text import canonicalize_url
from .cache import TTLByteCache
from .precision_discovery import source_channel_query
from .types import DiscoveredURL


_URL_RE = re.compile(r"https?://[^\s<>\]\[\"']+", re.I)
_VERSION_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff]")
_KOREAN_RE = re.compile(r"[\uac00-\ud7af]")


def searxng_search_params(
    query: str,
    freshness: str,
    source_channels: Sequence[str] = (),
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
    channels = list(
        dict.fromkeys(value.strip() for value in source_channels if value.strip())
    )
    if channels:
        params["categories"] = ",".join(channels[:2])
    return params


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
        if self.engine == "bing" and self._in_bing_result and lowered == "p":
            self._snippet_capture = True
            self._snippet_parts = []
            return
        if self._capture:
            self._depth += 1
            return
        if lowered != "a" or not values.get("href"):
            return
        classes = values.get("class", "")
        href = values["href"]
        if (
            self.engine == "bing"
            and self._in_bing_result
            and ("tilk" in classes or href.startswith("http"))
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
        if self.engine == "bing" and lowered == "li":
            self._in_bing_result = False

    def handle_data(self, data: str) -> None:
        if self._capture and data.strip():
            self._parts.append(data.strip())
        elif self._snippet_capture and data.strip():
            self._snippet_parts.append(data.strip())


def _is_search_internal(url: str) -> bool:
    host = (urlsplit(url).hostname or "").casefold()
    return host.endswith(("bing.com", "bing.net", "baidu.com", "searx.space"))


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
    def __init__(self, config: RealtimeSearchConfig, session: object) -> None:
        self.config = config
        self.session = session
        self.cache: TTLByteCache[List[DiscoveredURL]] = TTLByteCache(
            max(1024 * 1024, config.cache_max_bytes // 8)
        )

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
        channel_key = ",".join(source_channels)
        key = f"{freshness}\0{channel_key}\0{query.casefold()}"
        cached = self.cache.get(key)
        if cached is not None:
            return direct + copy.deepcopy(cached)

        results: List[DiscoveredURL] = []
        if self.config.searxng_url.rstrip("/") and len(source_channels) > 1:
            channel_tasks = [
                self._searxng(
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
            for engine in self.config.fallback_engines:
                try:
                    if engine == "bing":
                        results = await self._html_engine(query, "bing")
                    elif engine == "baidu":
                        results = await self._html_engine(query, "baidu")
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
                    results = []
                if results:
                    break
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
        base = self.config.searxng_url.rstrip("/")
        if not base:
            return []
        params = searxng_search_params(query, freshness, source_channels)
        headers = {}
        if (urlsplit(base).hostname or "").casefold() in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            headers["X-Real-IP"] = "127.0.0.1"
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
                    if diagnostics is not None:
                        diagnostics.append(
                            {
                                "query": query,
                                "engine": "searxng",
                                "source_channels": ",".join(source_channels),
                                "error_type": "HTTPStatusError",
                                "message": f"HTTP {response.status}",
                            }
                        )
                    return []
                raw = await _read_limited(response.content, 2 * 1024 * 1024)
                raw = _decode_http_body(
                    raw, response.headers.get("Content-Encoding", "")
                )
            data = json.loads(raw.decode("utf-8", "replace"))
        except Exception as exc:
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "query": query,
                        "engine": "searxng",
                        "source_channels": ",".join(source_channels),
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:300],
                    }
                )
            return []
        return parse_searxng_results(data)

    async def _html_engine(self, query: str, engine: str) -> List[DiscoveredURL]:
        if engine == "baidu":
            url = "https://www.baidu.com/s"
            params = {"wd": query}
            headers: Dict[str, str] = {}
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
        return parse_search_html(raw.decode(charset, "replace"), engine)

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
