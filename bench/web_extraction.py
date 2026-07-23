from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
import socket
import ssl
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urljoin, urlsplit

from rwkv_search.config import RealtimeSearchConfig
from rwkv_search.realtime.extractor import extract_page
from rwkv_search.realtime.fetcher import AsyncPageFetcher
from rwkv_search.realtime.hybrid_extractor import extract_hybrid_html
from rwkv_search.realtime.types import FetchedPage

SNAPSHOT_SCHEMA_VERSION = "web-extraction-snapshot.v1"
RUN_SCHEMA_VERSION = "web-extraction-result.v1"
SUMMARY_SCHEMA_VERSION = "web-extraction-summary.v1"
FETCH_FAILURES = {
    "ok",
    "dns_error",
    "connect_timeout",
    "connect_error",
    "tls_error",
    "request_timeout",
    "http_403",
    "http_429",
    "http_4xx",
    "http_5xx",
    "redirect_error",
    "unsupported_content_type",
    "response_too_large",
    "response_headers_too_large",
    "empty_response",
    "deadline_cancelled",
    "unknown_error",
}
EXTRACTION_FAILURES = {
    "ok",
    "snapshot_unavailable",
    "unsupported_content_type",
    "decode_error",
    "empty_html",
    "js_shell",
    "extractor_unavailable",
    "extractor_error",
    "extractor_empty",
    "low_quality",
}
_CHARSET_RE = re.compile(r"charset\s*=\s*['\"]?([A-Za-z0-9._-]+)", re.I)
_META_CHARSET_RE = re.compile(rb"<meta[^>]+charset\s*=\s*['\"]?([A-Za-z0-9._-]+)", re.I)
_SPACE_RE = re.compile(r"[\t\f\v ]+")
_JS_SHELL_MARKERS = (
    "enable javascript",
    "javascript is required",
    "please turn javascript on",
    "you need to enable javascript",
    "__next_data__",
    "__nuxt__",
    "id=\"root\"",
    "id='root'",
    "id=\"app\"",
    "id='app'",
)
@dataclass
class ExtractionOutput:
    extractor: str
    available: bool
    text: str = ""
    title: str = ""
    author: Optional[str] = None
    published_at: Optional[str] = None
    failure: str = "ok"
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    elapsed_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class _MetadataTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: List[str] = []
        self.text_parts: List[str] = []
        self.author: Optional[str] = None
        self.published_at: Optional[str] = None
        self._in_title = False
        self._suppressed = 0

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        lowered = tag.casefold()
        if lowered == "title":
            self._in_title = True
        if lowered in {"script", "style", "noscript", "svg", "canvas"}:
            self._suppressed += 1
        if lowered == "meta":
            values = {str(key).casefold(): str(value or "") for key, value in attrs}
            key = (values.get("property") or values.get("name") or "").casefold()
            content = values.get("content", "").strip()
            if content and key in {"author", "article:author", "byl"} and not self.author:
                self.author = content
            if content and key in {
                "article:published_time",
                "date",
                "datepublished",
                "pubdate",
                "publishdate",
            } and not self.published_at:
                self.published_at = content

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered == "title":
            self._in_title = False
        if lowered in {"script", "style", "noscript", "svg", "canvas"} and self._suppressed:
            self._suppressed -= 1
        if lowered in {"p", "div", "section", "article", "li", "tr", "pre", "br", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if not self._suppressed:
            self.text_parts.append(data)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decode_body(body: bytes, content_type: str) -> str:
    choices: List[str] = []
    match = _CHARSET_RE.search(content_type)
    if match:
        choices.append(match.group(1))
    match_bytes = _META_CHARSET_RE.search(body[:8192])
    if match_bytes:
        choices.append(match_bytes.group(1).decode("ascii", "ignore"))
    choices.extend(["utf-8", "gb18030", "windows-1252"])
    seen = set()
    for charset in choices:
        if not charset or charset.casefold() in seen:
            continue
        seen.add(charset.casefold())
        try:
            return body.decode(charset)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", "replace")


def normalize_text(value: str) -> str:
    lines: List[str] = []
    last = None
    for raw in html.unescape(value).splitlines():
        line = _SPACE_RE.sub(" ", raw).strip()
        if line and line != last:
            lines.append(line)
            last = line
    return "\n".join(lines)


def parse_metadata_and_text(raw_html: str) -> tuple[str, str, Optional[str], Optional[str]]:
    parser = _MetadataTextParser()
    parser.feed(raw_html)
    return (
        normalize_text(" ".join(parser.title_parts))[:500],
        normalize_text("".join(parser.text_parts)),
        parser.author,
        parser.published_at,
    )


def looks_like_js_shell(raw_html: str, visible_text: str) -> bool:
    lowered = raw_html.casefold()
    explicit = any(marker in lowered for marker in _JS_SHELL_MARKERS)
    script_count = lowered.count("<script")
    visible_length = len(visible_text.strip())
    if "enable javascript" in visible_text.casefold() or "javascript is required" in visible_text.casefold():
        return True
    if explicit and visible_length < 320 and script_count >= 2:
        return True
    # Extractors can return only a short summary from an otherwise valid page.
    # Use the raw server-rendered text, not that summary, for generic shell rules.
    raw_visible_length = len(parse_metadata_and_text(raw_html)[1].strip())
    # Generic shell signal for large app/templates that contain many scripts but
    # almost no server-rendered text. This deliberately avoids domain rules.
    if len(raw_html) >= 15_000 and raw_visible_length < 600 and script_count >= 12:
        return True
    return len(raw_html) >= 20_000 and raw_visible_length < 180 and script_count >= 4


def classify_fetch_exception(exc: BaseException) -> str:
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    if isinstance(exc, asyncio.CancelledError) or "cancel" in name:
        return "deadline_cancelled"
    if isinstance(exc, socket.gaierror) or "dns" in name or "name or service not known" in message:
        return "dns_error"
    if isinstance(exc, (ssl.SSLError, ssl.CertificateError)) or "ssl" in name or "certificate" in message:
        return "tls_error"
    if "connect" in name and "timeout" in name:
        return "connect_timeout"
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in name:
        return "request_timeout"
    if "redirect" in message:
        return "redirect_error"
    if "too large" in message:
        return "response_too_large"
    if "more than" in message and "bytes when reading" in message:
        return "response_headers_too_large"
    if "connect" in name or "connection" in message:
        return "connect_error"
    return "unknown_error"


def classify_http_status(status: int) -> str:
    if 200 <= status < 300:
        return "ok"
    if status == 403:
        return "http_403"
    if status == 429:
        return "http_429"
    if 400 <= status < 500:
        return "http_4xx"
    if status >= 500:
        return "http_5xx"
    if 300 <= status < 400:
        return "redirect_error"
    return "unknown_error"


def content_kind(content_type: str, url: str) -> str:
    mime = content_type.split(";", 1)[0].strip().casefold()
    path = urlsplit(url).path.casefold()
    if mime == "application/pdf" or path.endswith(".pdf"):
        return "pdf"
    if mime in {"application/json", "application/ld+json"} or path.endswith(".json"):
        return "json"
    if mime == "text/markdown" or path.endswith((".md", ".markdown")):
        return "markdown"
    if mime == "text/plain" or path.endswith(".txt"):
        return "plain_text"
    if mime in {"text/html", "application/xhtml+xml", ""}:
        return "html"
    return "unsupported"


def _snapshot_extension(kind: str) -> str:
    return {
        "html": ".html",
        "pdf": ".pdf",
        "json": ".json",
        "markdown": ".md",
        "plain_text": ".txt",
    }.get(kind, ".bin")


async def capture_cases(
    cases: Sequence[Mapping[str, Any]],
    output_dir: Path,
    manifest_path: Path,
    *,
    concurrency: int = 6,
    timeout_seconds: float = 12.0,
    max_bytes: int = 12 * 1024 * 1024,
    max_redirects: int = 5,
) -> List[Dict[str, Any]]:
    import aiohttp

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds, connect=min(5.0, timeout_seconds))
    connector = aiohttp.TCPConnector(limit=max(1, concurrency), limit_per_host=2, ttl_dns_cache=300)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/pdf,application/json,text/plain;q=0.9,*/*;q=0.5",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    semaphore = asyncio.Semaphore(max(1, concurrency))
    records: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers=headers,
        max_line_size=64 * 1024,
        max_field_size=64 * 1024,
    ) as session:
        validator = AsyncPageFetcher(RealtimeSearchConfig(), session)

        async def capture(case: Mapping[str, Any]) -> Dict[str, Any]:
            started = time.perf_counter()
            current = str(case["url"])
            status = 0
            response_headers: Dict[str, str] = {}
            try:
                async with semaphore:
                    for redirect_index in range(max_redirects + 1):
                        await validator._validate_public_url(current)
                        response = await session.get(current, allow_redirects=False)
                        async with response:
                            status = int(response.status)
                            response_headers = {
                                key.casefold(): value
                                for key, value in response.headers.items()
                                if key.casefold() in {"content-type", "content-length", "date", "last-modified", "etag", "location"}
                            }
                            if status in {301, 302, 303, 307, 308}:
                                location = response.headers.get("Location")
                                if not location or redirect_index >= max_redirects:
                                    raise RuntimeError("redirect limit exceeded")
                                current = urljoin(current, location)
                                continue
                            outcome = classify_http_status(status)
                            if outcome != "ok":
                                return _capture_record(case, current, status, response_headers, b"", outcome, started)
                            body = bytearray()
                            async for chunk in response.content.iter_chunked(64 * 1024):
                                body.extend(chunk)
                                if len(body) > max_bytes:
                                    raise RuntimeError("response too large")
                            raw = bytes(body)
                            if not raw:
                                return _capture_record(case, current, status, response_headers, raw, "empty_response", started)
                            kind = content_kind(response.headers.get("Content-Type", ""), current)
                            if kind == "unsupported":
                                return _capture_record(case, current, status, response_headers, raw, "unsupported_content_type", started)
                            filename = f"{case['id']}{_snapshot_extension(kind)}"
                            (output_dir / filename).write_bytes(raw)
                            record = _capture_record(case, current, status, response_headers, raw, "ok", started)
                            record.update({"body_path": filename, "content_kind": kind})
                            return record
                    raise RuntimeError("redirect limit exceeded")
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                record = _capture_record(
                    case,
                    current,
                    status,
                    response_headers,
                    b"",
                    classify_fetch_exception(exc),
                    started,
                )
                record["error_type"] = type(exc).__name__
                record["error_message"] = str(exc)[:500]
                return record

        records = list(await asyncio.gather(*(capture(case) for case in cases)))
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def _capture_record(
    case: Mapping[str, Any],
    final_url: str,
    status: int,
    headers: Mapping[str, str],
    body: bytes,
    outcome: str,
    started: float,
) -> Dict[str, Any]:
    if outcome not in FETCH_FAILURES:
        outcome = "unknown_error"
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "case_id": case["id"],
        "requested_url": case["url"],
        "final_url": final_url,
        "status": status,
        "content_type": str(headers.get("content-type") or ""),
        "headers": dict(headers),
        "body_path": None,
        "body_sha256": sha256_bytes(body) if body else None,
        "body_bytes": len(body),
        "captured_at": time.time(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "fetch_outcome": outcome,
        "error_type": None,
        "error_message": None,
    }


def load_snapshot_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported snapshot schema")
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in output:
            raise ValueError(f"invalid or duplicate snapshot case_id: {case_id!r}")
        output[case_id] = row
    return output


def run_extractor(
    name: str,
    body: bytes,
    content_type_value: str,
    url: str,
) -> ExtractionOutput:
    started = time.perf_counter()
    kind = content_kind(content_type_value, url)
    if kind not in {"html", "markdown", "plain_text", "json"}:
        return ExtractionOutput(
            name,
            True,
            failure="unsupported_content_type",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
    try:
        if kind in {"markdown", "plain_text"}:
            value = normalize_text(decode_body(body, content_type_value))
            return _finish_output(name, value, "", None, None, started, "")
        if kind == "json":
            parsed = json.loads(decode_body(body, content_type_value))
            value = normalize_text(json.dumps(parsed, ensure_ascii=False, indent=2))
            return _finish_output(name, value, "", None, None, started, "")
        raw_html = decode_body(body, content_type_value)
        fallback_title, visible_text, fallback_author, fallback_date = parse_metadata_and_text(raw_html)
        if not raw_html.strip():
            return ExtractionOutput(name, True, failure="empty_html", elapsed_ms=_elapsed(started))
        if name == "current":
            page = FetchedPage(
                requested_url=url,
                final_url=url,
                status=200,
                content_type=content_type_value,
                body=body,
                fetched_at=time.time(),
                elapsed_ms=0.0,
                headers={},
            )
            document = extract_page(page)
            if document is None:
                return ExtractionOutput(
                    name,
                    True,
                    failure="js_shell" if looks_like_js_shell(raw_html, visible_text) else "extractor_empty",
                    elapsed_ms=_elapsed(started),
                )
            return _finish_output(
                name,
                document.text,
                document.title,
                fallback_author,
                document.published_at or fallback_date,
                started,
                raw_html,
            )
        if name == "trafilatura":
            import trafilatura

            extracted = trafilatura.bare_extraction(
                raw_html,
                url=url,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
                include_links=True,
                favor_precision=True,
                deduplicate=True,
                with_metadata=True,
            )
            data = extracted.as_dict() if hasattr(extracted, "as_dict") else extracted or {}
            return _finish_output(
                name,
                str(data.get("text") or data.get("raw_text") or ""),
                str(data.get("title") or fallback_title),
                _optional(data.get("author")) or fallback_author,
                _optional(data.get("date")) or fallback_date,
                started,
                raw_html,
            )
        if name == "justext":
            import justext

            stoplist_name = "Chinese" if _looks_cjk(visible_text) else "English"
            try:
                stoplist = justext.get_stoplist(stoplist_name)
            except ValueError:
                stoplist = justext.get_stoplist("English")
            paragraphs = justext.justext(body, stoplist)
            text_value = "\n".join(item.text for item in paragraphs if not item.is_boilerplate)
            return _finish_output(
                name,
                text_value,
                fallback_title,
                fallback_author,
                fallback_date,
                started,
                raw_html,
            )
        if name == "readability":
            from readability import Document

            document = Document(raw_html, url=url)
            summary_html = document.summary(html_partial=True)
            _, summary_text, _, _ = parse_metadata_and_text(summary_html)
            return _finish_output(
                name,
                summary_text,
                str(document.short_title() or fallback_title),
                fallback_author,
                fallback_date,
                started,
                raw_html,
            )
        if name == "hybrid_fast":
            return _run_hybrid_fast(
                body,
                raw_html,
                url,
                started,
            )
        if name == "resiliparse":
            from resiliparse.extract.html2text import extract_plain_text
            from resiliparse.parse.html import HTMLTree

            tree = HTMLTree.parse_from_bytes(body)
            text_value = extract_plain_text(tree, main_content=True, preserve_formatting=False)
            return _finish_output(
                name,
                text_value,
                fallback_title,
                fallback_author,
                fallback_date,
                started,
                raw_html,
            )
        raise ValueError(f"unknown extractor: {name}")
    except ImportError as exc:
        return ExtractionOutput(
            name,
            False,
            failure="extractor_unavailable",
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
            elapsed_ms=_elapsed(started),
        )
    except Exception as exc:
        return ExtractionOutput(
            name,
            True,
            failure="decode_error" if isinstance(exc, UnicodeError) else "extractor_error",
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
            elapsed_ms=_elapsed(started),
        )


def _finish_output(
    name: str,
    text_value: str,
    title: str,
    author: Optional[str],
    published_at: Optional[str],
    started: float,
    raw_html: str,
) -> ExtractionOutput:
    clean = normalize_text(text_value)
    failure = "ok"
    if raw_html and looks_like_js_shell(raw_html, clean):
        failure = "js_shell"
    elif not clean:
        failure = "js_shell" if raw_html and looks_like_js_shell(raw_html, "") else "extractor_empty"
    elif len(clean) < 80:
        failure = "js_shell" if raw_html and looks_like_js_shell(raw_html, clean) else "low_quality"
    return ExtractionOutput(
        name,
        True,
        text=clean,
        title=normalize_text(title)[:500],
        author=_optional(author),
        published_at=_optional(published_at),
        failure=failure,
        elapsed_ms=_elapsed(started),
    )


def _run_hybrid_fast(
    body: bytes,
    raw_html: str,
    url: str,
    started: float,
) -> ExtractionOutput:
    extracted = extract_hybrid_html(body, raw_html, url)
    output = _finish_output(
        "hybrid_fast",
        extracted.document.text,
        extracted.document.title,
        extracted.author,
        extracted.document.published_at,
        started,
        raw_html,
    )
    output.elapsed_ms = _elapsed(started)
    output.details = dict(extracted.details)
    return output


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _optional(value: object) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _looks_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value[:4000]))


def evaluate_output(
    case: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    output: ExtractionOutput,
) -> Dict[str, Any]:
    title = output.title.casefold()
    text = output.text.casefold()
    title_hits = [
        value for value in case["title_contains_any"] if value.casefold() in title
    ]
    content_hits = [
        value for value in case["content_contains_any"] if value.casefold() in text
    ]
    forbidden_hits = [
        value for value in case["forbidden_content_any"] if value.casefold() in text
    ]
    table_hits = [value for value in case["table_text_any"] if value.casefold() in text]
    code_hits = [value for value in case["code_text_any"] if value.casefold() in text]
    expected = str(case["expected_static_outcome"])
    fetch_ok = snapshot.get("fetch_outcome") == "ok"
    static_usable = output.failure == "ok" and len(output.text) >= int(case["min_text_chars"])
    if expected == "usable":
        passed = bool(
            fetch_ok
            and static_usable
            and (not case["require_title"] or title_hits)
            and content_hits
            and not forbidden_hits
            and (not case["require_author"] or output.author)
            and (not case["require_published_at"] or output.published_at)
            and (not case["require_table"] or table_hits)
            and (not case["require_code"] or code_hits)
        )
    elif expected == "js_required":
        passed = bool(fetch_ok and output.failure == "js_shell")
    elif expected == "unsupported":
        passed = bool(
            snapshot.get("fetch_outcome") in {"ok", "unsupported_content_type"}
            and output.failure == "unsupported_content_type"
        )
    else:
        passed = not fetch_ok
    return {
        "passed": passed,
        "fetch_ok": fetch_ok,
        "static_usable": static_usable,
        "title_hit": bool(title_hits) if case["require_title"] else None,
        "title_hits": title_hits,
        "content_hit": bool(content_hits),
        "content_hits": content_hits,
        "forbidden_hits": forbidden_hits,
        "author_hit": bool(output.author) if case["require_author"] else None,
        "published_at_hit": bool(output.published_at) if case["require_published_at"] else None,
        "table_hit": bool(table_hits) if case["require_table"] else None,
        "code_hit": bool(code_hits) if case["require_code"] else None,
        "text_length": len(output.text),
        "text_sha256": sha256_bytes(output.text.encode("utf-8")) if output.text else None,
    }


def run_snapshot_benchmark(
    cases: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
    snapshot_dir: Path,
    extractors: Sequence[str],
    *,
    repetitions: int = 1,
) -> List[Dict[str, Any]]:
    if repetitions < 1 or repetitions > 20:
        raise ValueError("repetitions must be between 1 and 20")
    records: List[Dict[str, Any]] = []
    for case in cases:
        snapshot = dict(snapshots.get(str(case["id"])) or {})
        body_path = snapshot.get("body_path")
        body = b""
        if body_path:
            path = snapshot_dir / str(body_path)
            if path.exists():
                body = path.read_bytes()
                if snapshot.get("body_sha256") and sha256_bytes(body) != snapshot["body_sha256"]:
                    raise ValueError(f"snapshot hash mismatch: {case['id']}")
        for extractor in extractors:
            if not snapshot or snapshot.get("fetch_outcome") != "ok" or not body:
                output = ExtractionOutput(
                    extractor,
                    True,
                    failure="snapshot_unavailable",
                )
            else:
                outputs = []
                for _ in range(repetitions):
                    _reset_benchmark_extractor_state(extractor)
                    outputs.append(
                        run_extractor(
                            extractor,
                            body,
                            str(snapshot.get("content_type") or ""),
                            str(snapshot.get("final_url") or case["url"]),
                        )
                    )
                output = outputs[-1]
                output.elapsed_ms = round(
                    statistics.median(item.elapsed_ms for item in outputs), 3
                )
            signatures = {
                (
                    output.failure,
                    output.title,
                    output.author,
                    output.published_at,
                    sha256_bytes(output.text.encode("utf-8")) if output.text else None,
                )
            }
            if snapshot and snapshot.get("fetch_outcome") == "ok" and body:
                signatures = {
                    (
                        item.failure,
                        item.title,
                        item.author,
                        item.published_at,
                        sha256_bytes(item.text.encode("utf-8")) if item.text else None,
                    )
                    for item in outputs
                }
            metrics = evaluate_output(case, snapshot, output)
            records.append(
                {
                    "schema_version": RUN_SCHEMA_VERSION,
                    "case_id": case["id"],
                    "language": case["language"],
                    "page_type": case["page_type"],
                    "content_kind": case["content_kind"],
                    "expected_static_outcome": case["expected_static_outcome"],
                    "extractor": extractor,
                    "available": output.available,
                    "failure": output.failure,
                    "error_type": output.error_type,
                    "error_message": output.error_message,
                    "title": output.title,
                    "author_present": bool(output.author),
                    "published_at": output.published_at,
                    "elapsed_ms": round(output.elapsed_ms, 3),
                    "details": dict(output.details),
                    "repetitions": repetitions,
                    "deterministic": len(signatures) == 1,
                    "snapshot": {
                        "fetch_outcome": snapshot.get("fetch_outcome", "missing"),
                        "status": snapshot.get("status", 0),
                        "content_type": snapshot.get("content_type", ""),
                        "body_sha256": snapshot.get("body_sha256"),
                        "body_bytes": snapshot.get("body_bytes", 0),
                    },
                    "metrics": metrics,
                }
            )
    return records


def _reset_benchmark_extractor_state(name: str) -> None:
    """Prevent repeated-page timing from sharing Trafilatura's global dedup LRU."""
    if name not in {"current", "hybrid_fast", "trafilatura"}:
        return
    try:
        from trafilatura.deduplication import LRU_TEST
        from trafilatura.meta import reset_caches

        LRU_TEST.clear()
        reset_caches()
    except ImportError:
        return


def aggregate_results(
    cases: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    fetch_outcomes = Counter(
        str((snapshots.get(str(case["id"])) or {}).get("fetch_outcome") or "missing")
        for case in cases
    )
    by_extractor: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_extractor[str(record["extractor"])].append(record)
    extractor_summaries: Dict[str, Any] = {}
    for name, values in sorted(by_extractor.items()):
        executed = [
            value for value in values if value.get("failure") != "snapshot_unavailable"
        ]
        available_values = [value for value in executed if value.get("available")]
        usable_cases = [
            value
            for value in available_values
            if value["expected_static_outcome"] == "usable"
            and value["metrics"]["fetch_ok"]
        ]
        title_cases = [
            value for value in usable_cases if value["metrics"]["title_hit"] is not None
        ]
        author_cases = [
            value for value in usable_cases if value["metrics"]["author_hit"] is not None
        ]
        date_cases = [
            value
            for value in usable_cases
            if value["metrics"]["published_at_hit"] is not None
        ]
        table_cases = [
            value for value in usable_cases if value["metrics"]["table_hit"] is not None
        ]
        code_cases = [
            value for value in usable_cases if value["metrics"]["code_hit"] is not None
        ]
        elapsed = [float(value.get("elapsed_ms", 0.0)) for value in available_values]
        extractor_summaries[name] = {
            "total": len(values),
            "available": bool(executed) and bool(available_values),
            "evaluated": len(available_values),
            "pass_rate": _rate(sum(bool(value["metrics"]["passed"]) for value in available_values), len(available_values)),
            "usable_success_rate": _rate(sum(bool(value["metrics"]["static_usable"]) for value in usable_cases), len(usable_cases)),
            "title_hit_rate": _rate(sum(bool(value["metrics"]["title_hit"]) for value in title_cases), len(title_cases)),
            "content_marker_hit_rate": _rate(sum(bool(value["metrics"]["content_hit"]) for value in usable_cases), len(usable_cases)),
            "forbidden_leak_rate": _rate(sum(bool(value["metrics"]["forbidden_hits"]) for value in usable_cases), len(usable_cases)),
            "author_hit_rate": _rate(sum(bool(value["metrics"]["author_hit"]) for value in author_cases), len(author_cases)),
            "published_at_hit_rate": _rate(sum(bool(value["metrics"]["published_at_hit"]) for value in date_cases), len(date_cases)),
            "table_hit_rate": _rate(sum(bool(value["metrics"]["table_hit"]) for value in table_cases), len(table_cases)),
            "code_hit_rate": _rate(sum(bool(value["metrics"]["code_hit"]) for value in code_cases), len(code_cases)),
            "average_text_length": round(statistics.fmean([int(value["metrics"]["text_length"]) for value in usable_cases]), 2) if usable_cases else 0.0,
            "average_elapsed_ms": round(statistics.fmean(elapsed), 3) if elapsed else 0.0,
            "p95_elapsed_ms": _percentile(elapsed, 0.95),
            "failures": dict(sorted(Counter(str(value.get("failure") or "unknown") for value in available_values).items())),
            "fallback_trigger_rate": _rate(
                sum(
                    bool((value.get("details") or {}).get("fallback_triggered"))
                    for value in available_values
                ),
                len(available_values),
            ),
            "fallback_use_rate": _rate(
                sum(
                    bool((value.get("details") or {}).get("fallback_used"))
                    for value in available_values
                ),
                len(available_values),
            ),
            "fallback_reasons": dict(
                sorted(
                    Counter(
                        str((value.get("details") or {}).get("fallback_reason"))
                        for value in available_values
                        if (value.get("details") or {}).get("fallback_reason")
                    ).items()
                )
            ),
            "by_language": _group_pass_rates(available_values, "language"),
            "by_page_type": _group_pass_rates(available_values, "page_type"),
        }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "case_count": len(cases),
        "snapshot_count": len(snapshots),
        "fetch_outcomes": dict(sorted(fetch_outcomes.items())),
        "fetch_success_rate": _rate(fetch_outcomes.get("ok", 0), len(cases)),
        "extractors": extractor_summaries,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _group_pass_rates(
    values: Sequence[Mapping[str, Any]], field: str
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for value in values:
        grouped[str(value.get(field) or "unknown")].append(value)
    return {
        key: {
            "count": len(items),
            "pass_rate": _rate(
                sum(bool(item["metrics"]["passed"]) for item in items), len(items)
            ),
        }
        for key, items in sorted(grouped.items())
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.999999)))
    return round(ordered[index], 3)
