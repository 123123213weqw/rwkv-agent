from __future__ import annotations

import asyncio
import ipaddress
import socket
import time
import zlib
from collections import defaultdict
from typing import Dict, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

from ..config import RealtimeSearchConfig
from ..text import canonicalize_url
from .cache import TTLByteCache
from .types import FetchedPage


class FetchError(RuntimeError):
    pass


class AsyncPageFetcher:
    """One-GET fetcher with manual redirects, SSRF checks and byte ceilings."""

    ALLOWED_TYPES = (
        "text/html",
        "application/xhtml+xml",
        "text/plain",
        "application/json",
        "application/ld+json",
        "text/markdown",
        "text/x-markdown",
    )

    def __init__(self, config: RealtimeSearchConfig, session: object) -> None:
        self.config = config
        self.session = session
        self.global_limit = asyncio.Semaphore(max(1, config.global_concurrency))
        self.host_limits: Dict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(max(1, config.per_host_concurrency))
        )
        self.cache: TTLByteCache[FetchedPage] = TTLByteCache(config.cache_max_bytes)

    async def fetch(self, url: str) -> FetchedPage:
        canonical = canonicalize_url(url)
        navigation_url = _navigation_url(url)
        if not canonical or not navigation_url:
            raise FetchError("invalid URL")
        cached = self.cache.get(canonical)
        if cached is not None:
            return cached
        host = (urlsplit(canonical).hostname or "").casefold()
        async with self.global_limit, self.host_limits[host]:
            page = await self._await_fetch_with_cleanup(
                navigation_url,
                timeout=self.config.page_timeout_seconds,
            )
        self.cache.put(
            canonical,
            page,
            self.config.page_cache_ttl_seconds,
            size=len(page.body) + len(page.final_url) + 512,
        )
        if page.final_url != canonical:
            self.cache.put(
                page.final_url,
                page,
                self.config.page_cache_ttl_seconds,
                size=len(page.body) + len(page.final_url) + 512,
            )
        return page

    async def _await_fetch_with_cleanup(
        self,
        navigation_url: str,
        *,
        timeout: float,
    ) -> FetchedPage:
        """Apply the page deadline and always retrieve the child task outcome.

        ``asyncio.wait_for`` can leave a raced aiohttp connection task with an
        unobserved exception when the outer request deadline cancels the fetch at
        the same instant as a connect timeout.  An explicit task plus a cleanup
        ``gather`` makes timeout and caller cancellation deterministic.
        """

        task = asyncio.create_task(self._fetch_redirects(navigation_url))
        try:
            done, _ = await asyncio.wait(
                (task,),
                timeout=max(0.001, float(timeout)),
            )
            if task not in done:
                raise asyncio.TimeoutError
            return task.result()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _fetch_redirects(self, requested_url: str) -> FetchedPage:
        current = requested_url
        started = time.perf_counter()
        for redirect_index in range(self.config.max_redirects + 1):
            await self._validate_public_url(current)
            response = await self.session.get(current, allow_redirects=False)  # type: ignore[attr-defined]
            async with response:
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location or redirect_index >= self.config.max_redirects:
                        raise FetchError("redirect limit exceeded")
                    target = _navigation_url(urljoin(current, location))
                    if not target:
                        raise FetchError("invalid redirect target")
                    if target == current:
                        raise FetchError("redirect loop detected")
                    current = target
                    continue
                if response.status < 200 or response.status >= 300:
                    raise FetchError(f"HTTP {response.status}")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type and not any(
                    content_type == allowed for allowed in self.ALLOWED_TYPES
                ):
                    raise FetchError(f"unsupported content type {content_type}")
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > self.config.max_compressed_bytes:
                            raise FetchError("compressed response too large")
                    except ValueError:
                        pass
                compressed = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    compressed.extend(chunk)
                    if len(compressed) > self.config.max_compressed_bytes:
                        raise FetchError("compressed response too large")
                body = self._decode_content(
                    bytes(compressed), response.headers.get("Content-Encoding", "")
                )
                final_url = canonicalize_url(str(response.url)) or current
                return FetchedPage(
                    requested_url=requested_url,
                    final_url=final_url,
                    status=response.status,
                    content_type=response.headers.get("Content-Type", ""),
                    body=body,
                    fetched_at=time.time(),
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    headers={
                        key.lower(): value
                        for key, value in response.headers.items()
                        if key.lower() in {"content-type", "last-modified", "etag", "date"}
                    },
                )
        raise FetchError("redirect limit exceeded")

    def _decode_content(self, raw: bytes, encoding: str) -> bytes:
        encoding = encoding.casefold().strip()
        try:
            if encoding in {"gzip", "x-gzip"}:
                value = zlib.decompress(raw, 16 + zlib.MAX_WBITS, self.config.max_decompressed_bytes + 1)
            elif encoding == "deflate":
                try:
                    value = zlib.decompress(raw, zlib.MAX_WBITS, self.config.max_decompressed_bytes + 1)
                except zlib.error:
                    value = zlib.decompress(raw, -zlib.MAX_WBITS, self.config.max_decompressed_bytes + 1)
            elif encoding == "br":
                try:
                    import brotli  # type: ignore
                except ImportError as exc:
                    raise FetchError("brotli support is unavailable") from exc
                value = brotli.decompress(raw)
            elif encoding in {"", "identity"}:
                value = raw
            else:
                raise FetchError(f"unsupported content encoding {encoding}")
        except (zlib.error, ValueError) as exc:
            raise FetchError("invalid compressed response") from exc
        if len(value) > self.config.max_decompressed_bytes:
            raise FetchError("decompressed response too large")
        return value

    async def _validate_public_url(self, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise FetchError("invalid URL scheme or host")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port not in {80, 443} and not self.config.allow_private_networks:
            raise FetchError("non-standard network port blocked")
        try:
            literal = ipaddress.ip_address(parsed.hostname)
            addresses = [literal]
        except ValueError:
            loop = asyncio.get_running_loop()
            try:
                records = await loop.getaddrinfo(
                    parsed.hostname,
                    port,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                )
            except socket.gaierror as exc:
                raise FetchError("DNS resolution failed") from exc
            addresses = []
            for record in records:
                try:
                    addresses.append(ipaddress.ip_address(record[4][0]))
                except ValueError:
                    continue
        if not addresses:
            raise FetchError("host has no IP address")
        if self.config.allow_private_networks:
            return
        for address in addresses:
            if any(
                (
                    address.is_private,
                    address.is_loopback,
                    address.is_link_local,
                    address.is_multicast,
                    address.is_reserved,
                    address.is_unspecified,
                )
            ):
                raise FetchError("private or special network target blocked")


def _navigation_url(url: str) -> Optional[str]:
    """Validate a request URL while preserving path syntax such as trailing '/'."""
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        return None
    try:
        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii").casefold()
        port = parsed.port
    except (UnicodeError, ValueError):
        return None
    netloc = host
    scheme = parsed.scheme.casefold()
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))
