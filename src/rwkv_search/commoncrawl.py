from __future__ import annotations

import gzip
import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence
from urllib.parse import urlsplit

from .db import SearchDatabase
from .text import canonicalize_url, extract_document


@dataclass
class CDXRecord:
    url: str
    timestamp: str
    filename: str
    offset: int
    length: int
    mime: str
    status: int
    digest: Optional[str] = None
    languages: Optional[str] = None


@dataclass
class ArchivedResponse:
    status: int
    headers: Dict[str, str]
    body: bytes


def iter_cdxj(path: str | Path) -> Iterator[CDXRecord]:
    """Read a locally downloaded Common Crawl CDXJ shard/export."""
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                _, timestamp, raw = line.split(" ", 2)
                data = json.loads(raw)
                yield CDXRecord(
                    url=str(data["url"]),
                    timestamp=timestamp,
                    filename=str(data["filename"]),
                    offset=int(data["offset"]),
                    length=int(data["length"]),
                    mime=str(data.get("mime") or data.get("mime-detected") or ""),
                    status=int(data.get("status") or 0),
                    digest=str(data.get("digest")) if data.get("digest") else None,
                    languages=str(data.get("languages")) if data.get("languages") else None,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue


def filter_records(
    records: Iterable[CDXRecord],
    *,
    domains: Sequence[str] = (),
    languages: Sequence[str] = (),
    max_record_bytes: int = 4 * 1024 * 1024,
) -> Iterator[CDXRecord]:
    domain_set = {item.casefold().lstrip(".") for item in domains if item}
    language_set = {item.casefold() for item in languages if item}
    seen_urls = set()
    for record in records:
        host = (urlsplit(record.url).hostname or "").casefold()
        if domain_set and not any(host == domain or host.endswith("." + domain) for domain in domain_set):
            continue
        if language_set:
            record_languages = {part.strip().casefold() for part in (record.languages or "").split(",")}
            if not record_languages & language_set:
                continue
        if record.status != 200 or record.length <= 0 or record.length > max_record_bytes:
            continue
        if "html" not in record.mime.casefold():
            continue
        canonical = canonicalize_url(record.url)
        if not canonical or canonical in seen_urls:
            continue
        seen_urls.add(canonical)
        yield record


def fetch_warc_record(record: CDXRecord, *, timeout: float = 20.0) -> bytes:
    """Fetch one WARC gzip member with a byte Range; this is direct data access, not a search API."""
    start = record.offset
    end = record.offset + record.length - 1
    url = "https://data.commoncrawl.org/" + record.filename.lstrip("/")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RWKVLocalSearch/0.1",
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        compressed = response.read(record.length + 1)
    if len(compressed) > record.length:
        compressed = compressed[: record.length]
    return gzip.decompress(compressed)


def parse_warc_http(record_bytes: bytes) -> ArchivedResponse:
    try:
        _, http_message = record_bytes.split(b"\r\n\r\n", 1)
        raw_headers, body = http_message.split(b"\r\n\r\n", 1)
    except ValueError as exc:
        raise ValueError("invalid WARC/HTTP record") from exc
    lines = raw_headers.decode("iso-8859-1", "replace").splitlines()
    if not lines or not lines[0].startswith("HTTP/"):
        raise ValueError("WARC payload is not an HTTP response")
    parts = lines[0].split()
    status = int(parts[1]) if len(parts) > 1 else 0
    headers: Dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().casefold()] = value.strip()
    if headers.get("content-encoding", "").casefold() == "gzip":
        body = gzip.decompress(body)
    return ArchivedResponse(status=status, headers=headers, body=body)


class CommonCrawlImporter:
    def __init__(self, database: SearchDatabase, *, delay_seconds: float = 0.2) -> None:
        self.database = database
        self.delay_seconds = delay_seconds

    def import_records(self, records: Iterable[CDXRecord], max_pages: int = 100) -> Dict[str, int]:
        counters = {"processed": 0, "indexed": 0, "unchanged": 0, "failed": 0}
        for record in records:
            if counters["processed"] >= max_pages:
                break
            counters["processed"] += 1
            try:
                archived = parse_warc_http(fetch_warc_record(record))
                if archived.status != 200:
                    counters["failed"] += 1
                    continue
                content_type = archived.headers.get("content-type", "text/html")
                charset = "utf-8"
                for part in content_type.split(";")[1:]:
                    if "charset=" in part:
                        charset = part.split("=", 1)[1].strip().strip('"') or "utf-8"
                page = extract_document(archived.body.decode(charset, "replace"), record.url)
                if len(page.text) < 80:
                    counters["failed"] += 1
                    continue
                timestamp = datetime.strptime(record.timestamp[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                canonical = canonicalize_url(page.canonical_url or record.url) or record.url
                _, changed = self.database.upsert_document(
                    url=record.url,
                    canonical_url=canonical,
                    title=page.title,
                    content=page.text,
                    published_at=page.published_at,
                    fetched_at=timestamp.timestamp(),
                    etag=None,
                    last_modified=archived.headers.get("last-modified"),
                    content_type=content_type,
                    language=page.language or record.languages,
                    source_type="common_crawl",
                    authority=0.45,
                    response_bytes=len(archived.body),
                )
                counters["indexed" if changed else "unchanged"] += 1
            except Exception:
                counters["failed"] += 1
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
        return counters
