from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..text import (
    ExtractedDocument,
    canonicalize_url,
    extract_document,
    extract_document_fallback,
)


_SPACE_RE = re.compile(r"[\t\f\v ]+")
_JS_SHELL_MARKERS = (
    "enable javascript",
    "javascript is required",
    "please turn javascript on",
    "you need to enable javascript",
    "__next_data__",
    "__nuxt__",
    'id="root"',
    "id='root'",
    'id="app"',
    "id='app'",
)
_GENERIC_BOILERPLATE_MARKERS = (
    "cookie settings",
    "contact us",
    "privacy policy",
    "site map",
    "skip to content",
    "terms of use",
    "联系我们",
    "推荐服务",
    "服务条款",
    "网站声明",
    "网站地图",
    "隐私政策",
)


@dataclass
class HybridExtraction:
    document: ExtractedDocument
    author: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


def extract_hybrid_html(body: bytes, raw_html: str, url: str) -> HybridExtraction:
    """Extract HTML with a fast body path and a bounded quality fallback."""
    started = time.perf_counter()
    try:
        from resiliparse.extract.html2text import extract_plain_text
        from resiliparse.parse.html import HTMLTree
        from trafilatura.metadata import extract_metadata
    except ImportError:
        return _legacy_result(raw_html, url, started, "extractor_unavailable")

    try:
        shell = extract_document_fallback(raw_html, url)
        metadata_started = time.perf_counter()
        metadata = extract_metadata(raw_html, default_url=url, extensive=False)
        metadata_elapsed_ms = _elapsed(metadata_started)
        tree = HTMLTree.parse_from_bytes(body)
        primary_text = _normalize_text(
            extract_plain_text(tree, main_content=True, preserve_formatting=False)
        )
        primary_text_length = len(primary_text)
        primary_failure = _text_failure(raw_html, primary_text)
        title = _optional(getattr(metadata, "title", None)) or shell.title
        author = _optional(getattr(metadata, "author", None))
        published_at = (
            _optional(getattr(metadata, "date", None)) or shell.published_at
        )
        canonical_url = canonicalize_url(
            _optional(getattr(metadata, "url", None))
            or shell.canonical_url
            or url
        )
        language = _optional(getattr(metadata, "language", None)) or shell.language
        reason = (
            None
            if primary_failure == "js_shell"
            else hybrid_fallback_reason(primary_text, shell.text)
        )
        fallback_used = False
        fallback_failure: Optional[str] = None
        if reason:
            fallback, fallback_failure = _full_trafilatura(raw_html, url)
            if prefer_fallback(primary_text, fallback.document.text):
                primary_text = fallback.document.text
                title = fallback.document.title or title
                author = fallback.author or author
                published_at = fallback.document.published_at or published_at
                canonical_url = fallback.document.canonical_url or canonical_url
                language = fallback.document.language or language
                fallback_used = True

        return HybridExtraction(
            ExtractedDocument(
                title=title,
                text=primary_text,
                links=shell.links,
                canonical_url=canonical_url,
                published_at=published_at,
                language=language,
            ),
            author=author,
            details={
                "primary": "resiliparse",
                "metadata": "trafilatura_metadata",
                "metadata_elapsed_ms": metadata_elapsed_ms,
                "fallback_triggered": bool(reason),
                "fallback_used": fallback_used,
                "fallback_reason": reason,
                "primary_failure": primary_failure,
                "primary_text_length": primary_text_length,
                "fallback_failure": fallback_failure,
                "elapsed_ms": _elapsed(started),
            },
        )
    except Exception:
        return _legacy_result(raw_html, url, started, "extractor_error")


def hybrid_fallback_reason(primary_text: str, visible_text: str) -> Optional[str]:
    clean = primary_text.strip()
    if not clean:
        return "extractor_empty"
    if len(clean) < 80:
        return "low_quality"
    if len(clean) < 400 and len(visible_text.strip()) >= max(1200, len(clean) * 3):
        return "low_main_content_ratio"
    boilerplate_markers = boilerplate_marker_count(clean)
    if boilerplate_markers >= 3 or (
        boilerplate_markers >= 2 and len(clean) < 2000
    ):
        return "generic_boilerplate"
    if boilerplate_markers and len(clean) < 2000:
        lowered = clean.casefold()
        if any(
            lowered.rfind(marker) >= int(len(lowered) * 0.75)
            for marker in _GENERIC_BOILERPLATE_MARKERS
        ):
            return "generic_boilerplate_tail"
    return None


def boilerplate_marker_count(value: str) -> int:
    lowered = value.casefold()
    return sum(lowered.count(marker) for marker in _GENERIC_BOILERPLATE_MARKERS)


def prefer_fallback(primary_text: str, fallback_text: str) -> bool:
    if len(fallback_text.strip()) < 80:
        return False
    if len(primary_text.strip()) < 80:
        return True
    primary_boilerplate = boilerplate_marker_count(primary_text)
    fallback_boilerplate = boilerplate_marker_count(fallback_text)
    if fallback_boilerplate < primary_boilerplate:
        return len(fallback_text) >= max(80, int(len(primary_text) * 0.3))
    return len(primary_text) < 400 and len(fallback_text) > len(primary_text) * 1.5


def _full_trafilatura(
    raw_html: str, url: str
) -> tuple[HybridExtraction, Optional[str]]:
    try:
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
        text = _normalize_text(str(data.get("text") or data.get("raw_text") or ""))
        document = ExtractedDocument(
            title=_normalize_text(data.get("title"))[:500],
            text=text,
            links=[],
            canonical_url=canonicalize_url(str(data.get("url") or url)),
            published_at=_optional(data.get("date")),
            language=_optional(data.get("language")),
        )
        result = HybridExtraction(document, author=_optional(data.get("author")))
        if not text:
            return result, "extractor_empty"
        if len(text) < 80:
            return result, "low_quality"
        return result, None
    except ImportError:
        return _empty_result(url), "extractor_unavailable"
    except Exception:
        return _empty_result(url), "extractor_error"


def _normalize_text(value: object) -> str:
    lines = []
    last = None
    for raw in html.unescape(str(value or "")).splitlines():
        line = _SPACE_RE.sub(" ", raw).strip()
        if line and line != last:
            lines.append(line)
            last = line
    return "\n".join(lines)


def _optional(value: object) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _text_failure(raw_html: str, text: str) -> str:
    clean = text.strip()
    if looks_like_js_shell(raw_html, clean):
        return "js_shell"
    if not clean:
        return "extractor_empty"
    if len(clean) < 80:
        return "low_quality"
    return "ok"


def looks_like_js_shell(raw_html: str, text: str) -> bool:
    lowered = raw_html.casefold()
    marker = any(item in lowered for item in _JS_SHELL_MARKERS)
    script_count = lowered.count("<script")
    return len(text.strip()) < 120 and marker and script_count >= 1


def _legacy_result(
    raw_html: str, url: str, started: float, reason: str
) -> HybridExtraction:
    return HybridExtraction(
        extract_document(raw_html, url),
        details={
            "primary": "legacy_fallback",
            "fallback_triggered": False,
            "fallback_used": True,
            "fallback_reason": reason,
            "primary_failure": reason,
            "primary_text_length": 0,
            "fallback_failure": None,
            "elapsed_ms": _elapsed(started),
        },
    )


def _empty_result(url: str) -> HybridExtraction:
    return HybridExtraction(
        ExtractedDocument(
            title="",
            text="",
            links=[],
            canonical_url=canonicalize_url(url),
            published_at=None,
            language=None,
        )
    )
