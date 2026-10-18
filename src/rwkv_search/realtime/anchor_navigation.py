"""Bounded, anchor-aware navigation inside a first-party website.

The production crawler currently stores only outgoing URLs, which discards the
anchor label and nearby page context before link selection.  This experimental
module preserves those signals and uses the existing generic query-view BM25
selector to choose a small best-first frontier.  It contains no site, topic, or
Gold URL rules and is not wired into the default runtime.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import heapq
import re
import time
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urljoin, urlsplit

from selectolax.parser import HTMLParser

from ..semantic_selection import select_diverse_items
from ..text import canonicalize_url


_SCRIPT_SCHEME = re.compile(r"^(?:javascript|mailto|tel|data):", re.I)
_ASSET = re.compile(
    r"\.(?:7z|avi|bmp|css|docx?|eot|exe|gif|gz|ico|jpe?g|js|m4a|mov|mp3|"
    r"mp4|mpeg|ogg|otf|pdf|png|pptx?|rar|rss|svg|tar|tiff?|ttf|wav|webm|webp|"
    r"woff2?|xlsx?|xml|zip)(?:$|[?#])",
    re.I,
)
_NON_CONTENT = re.compile(
    r"(?:^|/)(?:account|captcha|feedback|login|logout|register|search|signin|"
    r"signup)(?:/|$)",
    re.I,
)
_PAGINATION_TEXT = re.compile(
    r"^(?:next|next\s+page|older|more|下一页|下页|后一页|后页|更多|\d{1,4})$",
    re.I,
)
_CONTEXT_TAGS = frozenset({"article", "dd", "div", "li", "section", "td", "tr"})


@dataclass(frozen=True)
class AnchorLink:
    url: str
    title: str
    context: str
    parent_url: str
    position: int
    pagination: bool = False


@dataclass(frozen=True)
class AnchorNavigationResult:
    candidates: tuple[AnchorLink, ...]
    fetched_urls: tuple[str, ...]
    requests: tuple[Mapping[str, Any], ...]
    error: str = ""


@dataclass(frozen=True)
class AnchorPageFetch:
    body: bytes
    final_url: str
    status: int
    content_type: str
    elapsed_ms: float
    error: str = ""


class AnchorPageCache:
    """Run-scoped URL singleflight for duplicate first-party page reads."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[AnchorPageFetch]] = {}

    async def fetch(
        self,
        session: Any,
        url: str,
        *,
        scope_host: str,
        timeout_seconds: float,
        max_body_bytes: int,
    ) -> tuple[AnchorPageFetch, bool]:
        key = canonicalize_url(url) or str(url)
        task = self._tasks.get(key)
        cache_hit = task is not None
        if task is None:
            task = asyncio.create_task(
                _fetch_anchor_page(
                    session,
                    key,
                    scope_host=scope_host,
                    timeout_seconds=timeout_seconds,
                    max_body_bytes=max_body_bytes,
                )
            )
            self._tasks[key] = task
        return await asyncio.shield(task), cache_hit


def _host(value: str) -> str:
    return (urlsplit(value).hostname or "").casefold().removeprefix("www.")


def _same_host_or_subdomain(url: str, scope_host: str) -> bool:
    actual = _host(url)
    return bool(
        actual
        and scope_host
        and (actual == scope_host or actual.endswith("." + scope_host))
    )


def _clean_text(value: str, *, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _anchor_context(node: Any) -> str:
    parent = node.parent
    fallback = ""
    for _ in range(4):
        if parent is None:
            break
        text = _clean_text(parent.text(separator=" "), limit=700)
        if text and not fallback:
            fallback = text
        if str(parent.tag or "").casefold() in _CONTEXT_TAGS and len(text) <= 700:
            return text
        parent = parent.parent
    return fallback


def _is_pagination(node: Any, title: str) -> bool:
    rel = str(node.attributes.get("rel") or "").casefold().split()
    if "next" in rel:
        return True
    visible = " ".join(
        (
            title,
            str(node.attributes.get("title") or ""),
            str(node.attributes.get("aria-label") or ""),
        )
    ).strip()
    return bool(_PAGINATION_TEXT.fullmatch(" ".join(visible.split())))


def extract_anchor_links(
    html: bytes | str,
    *,
    base_url: str,
    scope_host: str = "",
    limit: int = 500,
) -> tuple[AnchorLink, ...]:
    """Extract same-site links while preserving label and local context."""

    parser = HTMLParser(html)
    effective_base = base_url
    base = parser.css_first("base[href]")
    if base is not None:
        effective_base = urljoin(
            base_url,
            str(base.attributes.get("href") or ""),
        )
    expected = scope_host or _host(base_url)
    current = canonicalize_url(base_url)
    output: list[AnchorLink] = []
    indexes: dict[str, int] = {}
    for position, node in enumerate(parser.css("a[href]"), 1):
        raw = str(node.attributes.get("href") or "").strip()
        if not raw or raw.startswith("#") or _SCRIPT_SCHEME.match(raw):
            continue
        url = canonicalize_url(urljoin(effective_base, raw))
        if not url or url == current or not _same_host_or_subdomain(url, expected):
            continue
        decoded = unquote(urlsplit(url).path)
        if decoded in {"", "/"} or _ASSET.search(url) or _NON_CONTENT.search(decoded):
            continue
        title = _clean_text(node.text(separator=" "), limit=500)
        if not title:
            title = _clean_text(
                str(node.attributes.get("title") or "")
                or str(node.attributes.get("aria-label") or ""),
                limit=500,
            )
        context = _anchor_context(node)
        if not title and not context:
            continue
        candidate = AnchorLink(
            url=url,
            title=title or decoded.replace("/", " ").strip(),
            context=context,
            parent_url=current or base_url,
            position=position,
            pagination=_is_pagination(node, title),
        )
        existing_index = indexes.get(url)
        if existing_index is not None:
            existing = output[existing_index]
            if len(candidate.title) + len(candidate.context) > len(existing.title) + len(existing.context):
                output[existing_index] = candidate
            continue
        indexes[url] = len(output)
        output.append(candidate)
        if len(output) >= max(0, int(limit)):
            break
    return tuple(output)


def select_navigation_links(
    question: str,
    query_views: Sequence[str],
    links: Sequence[AnchorLink],
    *,
    max_links: int = 24,
    max_pagination_links: int = 2,
) -> tuple[AnchorLink, ...]:
    """Select a bounded frontier using anchor/context rather than URL alone."""

    cap = min(max(0, int(max_links)), len(links))
    if cap == 0:
        return ()
    pagination = [link for link in links if link.pagination]
    pagination_cap = min(max(0, int(max_pagination_links)), cap, len(pagination))
    semantic_cap = cap - pagination_cap
    items = [
        {
            "url": link.url,
            "title": link.title,
            "content": f"{link.context} {unquote(urlsplit(link.url).path)}",
            "parent_url": link.parent_url,
            "position": link.position,
            "pagination": link.pagination,
            "_best_position": link.position,
            "_upstream_score": 1.0 / max(1, link.position),
            "_order": index,
        }
        for index, link in enumerate(links)
        if not link.pagination
    ]
    selected = select_diverse_items(
        question,
        query_views,
        items,
        limit=semantic_cap,
        scorer=None,
        redundancy_weight=0.10,
        domain_weight=0.0,
    )
    by_url = {link.url: link for link in links}
    output = [
        by_url[str(item.get("url") or item.get("uri") or "")]
        for item in selected.items
        if str(item.get("url") or item.get("uri") or "") in by_url
    ]
    seen = {link.url for link in output}
    for link in pagination:
        if link.url in seen:
            continue
        output.append(link)
        seen.add(link.url)
        if len(output) >= cap:
            break
    return tuple(output[:cap])


async def _fetch_anchor_page(
    session: Any,
    url: str,
    *,
    scope_host: str,
    timeout_seconds: float,
    max_body_bytes: int,
) -> AnchorPageFetch:
    started = time.perf_counter()
    body = b""
    final_url = url
    status = 0
    error = ""
    content_type = ""
    try:
        async with session.get(
            url,
            allow_redirects=True,
            timeout=timeout_seconds,
        ) as response:
            status = int(response.status)
            final_url = canonicalize_url(str(response.url)) or str(response.url)
            content_type = str(response.headers.get("Content-Type") or "")
            body = await response.content.read(max_body_bytes + 1)
            if len(body) > max_body_bytes:
                body = b""
                error = "body_limit_exceeded"
            elif not 200 <= status < 300:
                body = b""
                error = f"http_{status}"
            elif content_type and not any(
                value in content_type.casefold() for value in ("html", "xhtml")
            ):
                body = b""
                error = "non_html"
            elif not _same_host_or_subdomain(final_url, scope_host):
                body = b""
                error = "cross_host_redirect"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:300]
    return AnchorPageFetch(
        body=body,
        final_url=final_url,
        status=status,
        content_type=content_type,
        elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        error=error,
    )


async def discover_anchor_navigation(
    session: Any,
    *,
    root_url: str,
    seed_urls: Sequence[str],
    question: str,
    query_views: Sequence[str],
    timeout_seconds: float = 7.0,
    max_page_fetches: int = 8,
    max_frontier_per_page: int = 24,
    max_depth: int = 3,
    max_links_per_page: int = 500,
    max_body_bytes: int = 2_000_000,
    page_cache: AnchorPageCache | None = None,
) -> AnchorNavigationResult:
    """Traverse a small same-site graph and expose every observed link URL."""

    scope_host = _host(root_url)
    requests: list[dict[str, Any]] = []
    fetched: list[str] = []
    fetched_set: set[str] = set()
    discovered: list[AnchorLink] = []
    discovered_index: dict[str, int] = {}
    frontier: list[tuple[int, int, int, str]] = []
    queued: set[str] = set()
    serial = 0
    cache = page_cache or AnchorPageCache()

    def enqueue(url: str, *, depth: int, rank: int) -> None:
        nonlocal serial
        canonical = canonicalize_url(url)
        if (
            not canonical
            or canonical in queued
            or canonical in fetched_set
            or not _same_host_or_subdomain(canonical, scope_host)
        ):
            return
        queued.add(canonical)
        heapq.heappush(frontier, (depth, rank, serial, canonical))
        serial += 1

    enqueue(root_url, depth=0, rank=0)
    for rank, seed in enumerate(seed_urls[:2], 1):
        enqueue(str(seed), depth=0, rank=rank)

    while frontier and len(fetched) < max(0, int(max_page_fetches)):
        depth, _, _, url = heapq.heappop(frontier)
        if url in fetched_set:
            continue
        fetched_set.add(url)
        page, cache_hit = await cache.fetch(
            session,
            url,
            scope_host=scope_host,
            timeout_seconds=timeout_seconds,
            max_body_bytes=max_body_bytes,
        )
        requests.append(
            {
                "method": "GET",
                "url": url,
                "final_url": page.final_url,
                "status": page.status,
                "bytes": len(page.body),
                "content_type": page.content_type[:120],
                "elapsed_ms": page.elapsed_ms,
                "error": page.error,
                "depth": depth,
                "cache_hit": cache_hit,
                "network_request": not cache_hit,
            }
        )
        fetched.append(url)
        if not page.body:
            continue
        links = extract_anchor_links(
            page.body,
            base_url=page.final_url,
            scope_host=scope_host,
            limit=max_links_per_page,
        )
        for link in links:
            existing_index = discovered_index.get(link.url)
            if existing_index is None:
                discovered_index[link.url] = len(discovered)
                discovered.append(link)
            else:
                existing = discovered[existing_index]
                if len(link.title) + len(link.context) > len(existing.title) + len(existing.context):
                    discovered[existing_index] = link
        if depth >= max(0, int(max_depth)):
            continue
        selected = select_navigation_links(
            question,
            query_views,
            links,
            max_links=max_frontier_per_page,
        )
        for rank, link in enumerate(selected, 1):
            enqueue(link.url, depth=depth + 1, rank=rank)

    error = "" if discovered else "no_anchor_candidates"
    return AnchorNavigationResult(
        candidates=tuple(discovered),
        fetched_urls=tuple(fetched),
        requests=tuple(requests),
        error=error,
    )
