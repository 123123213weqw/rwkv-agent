from __future__ import annotations

import asyncio
import gzip
import ipaddress
import socket
import time
import urllib.error
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit

from .config import CrawlConfig
from .db import SearchDatabase
from .text import canonicalize_url, extract_document


@dataclass
class FetchResult:
    requested_url: str
    final_url: str
    status: int
    content_type: str
    body: bytes
    etag: Optional[str]
    last_modified: Optional[str]


class FetchRejected(RuntimeError):
    pass


def public_url_allowed(url: str, allow_private: bool = False) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if allow_private:
        return True
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            return False
    return bool(infos)


class HttpFetcher:
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config

    async def fetch(
        self,
        url: str,
        *,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
    ) -> FetchResult:
        if not public_url_allowed(url, self.config.allow_private_networks):
            raise FetchRejected("URL is non-public, unresolved, or uses a blocked address")
        return await asyncio.to_thread(self._fetch_sync, url, etag, last_modified)

    def _fetch_sync(self, url: str, etag: Optional[str], last_modified: Optional[str]) -> FetchResult:
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml,text/xml,application/rss+xml,application/atom+xml;q=0.9,text/plain;q=0.7,*/*;q=0.2",
            "Accept-Encoding": "gzip",
        }
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            response = urllib.request.urlopen(request, timeout=self.config.timeout_seconds)
        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                return FetchResult(url, url, 304, "", b"", etag, last_modified)
            raise
        with response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > self.config.max_response_bytes:
                raise FetchRejected(f"response exceeds {self.config.max_response_bytes} bytes")
            body = response.read(self.config.max_response_bytes + 1)
            if len(body) > self.config.max_response_bytes:
                raise FetchRejected(f"response exceeds {self.config.max_response_bytes} bytes")
            if response.headers.get("Content-Encoding", "").casefold() == "gzip":
                body = gzip.decompress(body)
                if len(body) > self.config.max_response_bytes:
                    raise FetchRejected("decompressed response exceeds limit")
            return FetchResult(
                requested_url=url,
                final_url=response.geturl(),
                status=int(response.status),
                content_type=response.headers.get("Content-Type", ""),
                body=body,
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
            )


class RobotsManager:
    def __init__(self, database: SearchDatabase, config: CrawlConfig) -> None:
        self.database = database
        self.config = config
        self._memory: Dict[str, Tuple[float, urllib.robotparser.RobotFileParser, List[str]]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> Tuple[bool, List[str]]:
        parsed = urlsplit(url)
        origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        parser, sitemaps = await self._load(origin)
        return parser.can_fetch(self.config.user_agent, url), sitemaps

    async def _load(self, origin: str) -> Tuple[urllib.robotparser.RobotFileParser, List[str]]:
        now = time.time()
        cached_memory = self._memory.get(origin)
        if cached_memory and now - cached_memory[0] < self.config.robots_cache_seconds:
            return cached_memory[1], cached_memory[2]
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            cached_memory = self._memory.get(origin)
            if cached_memory and now - cached_memory[0] < self.config.robots_cache_seconds:
                return cached_memory[1], cached_memory[2]
            cached = self.database.get_robots(origin)
            if cached and now - float(cached["fetched_at"]) < self.config.robots_cache_seconds:
                parser, maps = self._parse(origin, str(cached["body"]))
                self._memory[origin] = (float(cached["fetched_at"]), parser, maps)
                return parser, maps
            status, body = await asyncio.to_thread(self._fetch_sync, origin + "/robots.txt")
            self.database.set_robots(origin, body, status)
            parser, maps = self._parse(origin, body)
            self._memory[origin] = (time.time(), parser, maps)
            return parser, maps

    def _fetch_sync(self, robots_url: str) -> Tuple[int, str]:
        if not public_url_allowed(robots_url, self.config.allow_private_networks):
            return 599, "User-agent: *\nDisallow: /\n"
        request = urllib.request.Request(
            robots_url,
            headers={"User-Agent": self.config.user_agent, "Accept": "text/plain,*/*;q=0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read(min(self.config.max_response_bytes, 512 * 1024)).decode("utf-8", "replace")
                return int(response.status), body
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500:
                return int(exc.code), "User-agent: *\nAllow: /\n"
            return int(exc.code), "User-agent: *\nDisallow: /\n"
        except (OSError, TimeoutError):
            return 599, "User-agent: *\nDisallow: /\n"

    def _parse(self, origin: str, body: str) -> Tuple[urllib.robotparser.RobotFileParser, List[str]]:
        parser = urllib.robotparser.RobotFileParser(origin + "/robots.txt")
        parser.parse(body.splitlines())
        maps = [canonical for value in (parser.site_maps() or []) if (canonical := canonicalize_url(value))]
        return parser, maps


class FocusedCrawler:
    def __init__(self, database: SearchDatabase, config: Optional[CrawlConfig] = None) -> None:
        self.database = database
        self.config = config or CrawlConfig()
        self.fetcher = HttpFetcher(self.config)
        self.robots = RobotsManager(database, self.config)
        self.global_semaphore = asyncio.Semaphore(self.config.global_concurrency)
        self.host_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.host_locks: Dict[str, asyncio.Lock] = {}
        self.host_last_fetch: Dict[str, float] = {}
        self.counters = {"processed": 0, "indexed": 0, "unchanged": 0, "skipped": 0, "failed": 0}

    def seed(self, urls: List[str]) -> int:
        count = 0
        for url in urls:
            canonical = canonicalize_url(url)
            if canonical:
                self.database.enqueue(canonical, depth=0, priority=100.0)
                count += 1
        return count

    async def run(self, max_pages: Optional[int] = None) -> Dict[str, int]:
        budget = max_pages or self.config.max_pages_per_run
        while self.counters["processed"] < budget:
            remaining = budget - self.counters["processed"]
            batch = self.database.lease_frontier(min(remaining, self.config.global_concurrency * 2))
            if not batch:
                break
            await asyncio.gather(*(self._process(row) for row in batch))
        return dict(self.counters)

    async def _process(self, item: Dict[str, Any]) -> None:
        url = str(item["url"])
        depth = int(item["depth"])
        self.counters["processed"] += 1
        if depth > self.config.max_depth:
            self.database.skip_frontier(url, "depth budget exceeded")
            self.counters["skipped"] += 1
            return
        if not public_url_allowed(url, self.config.allow_private_networks):
            self.database.skip_frontier(url, "blocked or unresolved address")
            self.counters["skipped"] += 1
            return
        try:
            allowed, sitemaps = await self.robots.allowed(url)
            for sitemap in sitemaps[:20]:
                self.database.enqueue(sitemap, source_url=url, depth=depth, priority=95.0)
            if not allowed:
                self.database.skip_frontier(url, "robots.txt disallow")
                self.counters["skipped"] += 1
                return
            result = await self._polite_fetch(url)
            if result.status == 304:
                self.database.complete_frontier(url, "not modified")
                self.counters["unchanged"] += 1
                return
            content_type = result.content_type.casefold()
            if self._is_discovery_document(result.final_url, content_type):
                links = self._xml_links(result.body, result.final_url)
                for link in links[:50000]:
                    if self._same_host(url, link):
                        self.database.enqueue(link, source_url=url, depth=depth, priority=90.0)
                self.database.complete_frontier(url, f"discovery document: {len(links)} links")
                return
            if not any(kind in content_type for kind in ("text/html", "application/xhtml+xml", "text/plain")):
                self.database.skip_frontier(url, f"unsupported content type: {content_type}")
                self.counters["skipped"] += 1
                return
            charset = self._charset(content_type)
            decoded = result.body.decode(charset, "replace")
            document = extract_document(decoded, result.final_url)
            if len(document.text) < 40:
                self.database.skip_frontier(url, "extracted body too short")
                self.counters["skipped"] += 1
                return
            canonical = canonicalize_url(document.canonical_url or result.final_url) or url
            _, changed = self.database.upsert_document(
                url=result.final_url,
                canonical_url=canonical,
                title=document.title,
                content=document.text,
                published_at=document.published_at,
                fetched_at=time.time(),
                etag=result.etag,
                last_modified=result.last_modified,
                content_type=result.content_type,
                language=document.language,
                authority=self._authority(canonical),
                response_bytes=len(result.body),
            )
            if changed:
                self.counters["indexed"] += 1
            else:
                self.counters["unchanged"] += 1
            if depth < self.config.max_depth:
                for link in document.links[: self.config.max_links_per_page]:
                    if self._same_host(canonical, link):
                        priority = max(1.0, float(item["priority"]) * 0.72 + self._url_bonus(link))
                        self.database.enqueue(link, source_url=canonical, depth=depth + 1, priority=priority)
            self.database.complete_frontier(url, f"indexed={changed}")
        except Exception as exc:
            self.database.fail_frontier(url, f"{type(exc).__name__}: {exc}")
            self.counters["failed"] += 1

    async def _polite_fetch(self, url: str) -> FetchResult:
        host = urlsplit(url).netloc.casefold()
        host_semaphore = self.host_semaphores.setdefault(host, asyncio.Semaphore(self.config.per_host_concurrency))
        host_lock = self.host_locks.setdefault(host, asyncio.Lock())
        existing = self.database.get_document_by_url(canonicalize_url(url) or url)
        async with self.global_semaphore, host_semaphore:
            async with host_lock:
                elapsed = time.monotonic() - self.host_last_fetch.get(host, 0.0)
                delay = self.config.per_host_delay_seconds - elapsed
                if delay > 0:
                    await asyncio.sleep(delay)
                self.host_last_fetch[host] = time.monotonic()
            return await self.fetcher.fetch(
                url,
                etag=str(existing.get("etag")) if existing and existing.get("etag") else None,
                last_modified=str(existing.get("last_modified"))
                if existing and existing.get("last_modified")
                else None,
            )

    @staticmethod
    def _same_host(parent: str, child: str) -> bool:
        return urlsplit(parent).hostname == urlsplit(child).hostname

    @staticmethod
    def _charset(content_type: str) -> str:
        for part in content_type.split(";")[1:]:
            if "charset=" in part:
                return part.split("=", 1)[1].strip().strip('"') or "utf-8"
        return "utf-8"

    @staticmethod
    def _is_discovery_document(url: str, content_type: str) -> bool:
        path = urlsplit(url).path.casefold()
        return (
            "xml" in content_type
            or "rss" in content_type
            or "atom" in content_type
            or path.endswith((".xml", ".rss", ".atom"))
            or "sitemap" in path
        )

    @staticmethod
    def _xml_links(body: bytes, base_url: str) -> List[str]:
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            return []
        links: List[str] = []
        seen = set()
        for element in root.iter():
            local = element.tag.rsplit("}", 1)[-1].casefold()
            candidates = []
            if local == "loc" and element.text:
                candidates.append(element.text.strip())
            elif local == "link":
                if element.attrib.get("href"):
                    candidates.append(element.attrib["href"])
                elif element.text:
                    candidates.append(element.text.strip())
            for candidate in candidates:
                canonical = canonicalize_url(urljoin(base_url, candidate))
                if canonical and canonical not in seen:
                    seen.add(canonical)
                    links.append(canonical)
        return links

    @staticmethod
    def _authority(url: str) -> float:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        path = parsed.path.casefold()
        score = 0.5
        if host.endswith((".gov", ".gov.cn", ".edu", ".edu.cn", ".org")):
            score += 0.25
        if host in {"github.com", "arxiv.org", "www.rfc-editor.org"}:
            score += 0.2
        if any(part in path for part in ("/docs", "/documentation", "/releases", "/paper")):
            score += 0.1
        return min(1.0, score)

    @staticmethod
    def _url_bonus(url: str) -> float:
        path = urlsplit(url).path.casefold()
        if any(part in path for part in ("docs", "news", "release", "blog", "paper")):
            return 10.0
        if any(part in path for part in ("tag", "category", "archive", "login", "signup")):
            return -10.0
        return 0.0
