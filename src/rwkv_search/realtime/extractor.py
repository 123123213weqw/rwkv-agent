from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlsplit

from ..text import extract_document, simhash64
from .hybrid_extractor import extract_hybrid_html
from .types import FetchedPage, RealtimeDocument


_CHARSET_RE = re.compile(r"charset\s*=\s*['\"]?([A-Za-z0-9._-]+)", re.I)
_META_CHARSET_RE = re.compile(
    rb"<meta[^>]+charset\s*=\s*['\"]?([A-Za-z0-9._-]+)", re.I
)


def extract_page(page: FetchedPage) -> Optional[RealtimeDocument]:
    raw_html = _decode(page.body, page.content_type)
    mime = page.content_type.split(";", 1)[0].strip().casefold()
    if mime in {"", "text/html", "application/xhtml+xml"}:
        document = extract_hybrid_html(
            page.body, raw_html, page.final_url
        ).document
    else:
        document = extract_document(raw_html, page.final_url)
    text = document.text.strip()
    if len(text) < 80:
        return None
    url = document.canonical_url or page.final_url
    title = (document.title or url).strip()[:500]
    source_type, authority = classify_source(url)
    quality = extraction_quality(title, text)
    return RealtimeDocument(
        url=url,
        title=title,
        text=text[:400000],
        published_at=document.published_at,
        fetched_at=page.fetched_at,
        source_type=source_type,
        authority=authority,
        extraction_quality=quality,
        simhash=simhash64(text[:120000]),
        links=document.links[:200],
    )


def _decode(body: bytes, content_type: str) -> str:
    choices = []
    match = _CHARSET_RE.search(content_type)
    if match:
        choices.append(match.group(1))
    match_bytes = _META_CHARSET_RE.search(body[:8192])
    if match_bytes:
        choices.append(match_bytes.group(1).decode("ascii", "ignore"))
    choices.extend(["utf-8", "gb18030"])
    for charset in choices:
        try:
            return body.decode(charset)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", "replace")


def extraction_quality(title: str, text: str) -> float:
    length_score = min(1.0, len(text) / 3000.0)
    lines = [line for line in text.splitlines() if line.strip()]
    unique_ratio = len(set(lines)) / max(1, len(lines))
    sentence_marks = len(re.findall(r"[。！？.!?]", text[:20000]))
    prose_score = min(1.0, sentence_marks / 12.0)
    title_score = 1.0 if 3 <= len(title) <= 240 else 0.35
    return max(
        0.0,
        min(1.0, 0.42 * length_score + 0.23 * unique_ratio + 0.2 * prose_score + 0.15 * title_score),
    )


def classify_source(url: str) -> tuple[str, float]:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.casefold()
    # Primary financial disclosure and exchange sources must outrank generic
    # government pages and media. Keep this before the broad .gov classifier.
    if any(
        host == value or host.endswith("." + value)
        for value in (
            "sec.gov",
            "cninfo.com.cn",
            "hkexnews.hk",
            "sse.com.cn",
            "szse.cn",
            "bse.cn",
            "hkex.com.hk",
        )
    ):
        return "company_filing", 0.98
    if any(
        host == value or host.endswith("." + value)
        for value in ("csrc.gov.cn", "sfc.hk", "finra.org")
    ):
        return "regulator", 0.98
    if host.endswith((".gov", ".gov.cn", ".gov.uk", ".europa.eu")):
        return "regulator", 0.98
    if host in {"arxiv.org", "export.arxiv.org"} or "doi.org" in host:
        return "paper", 0.93
    if host == "github.com":
        if "/releases" in path:
            return "github_release", 0.91
        return "official_repository", 0.84
    if host.startswith("docs.") or "/docs/" in path or "/documentation/" in path:
        return "official_docs", 0.89
    if any(value in host for value in ("reuters.com", "apnews.com", "bbc.", "caixin.com")):
        return "news", 0.86
    if any(value in host for value in ("reddit.com", "zhihu.com", "stackoverflow.com", "v2ex.com")):
        return "forum", 0.58
    if host.endswith((".edu", ".edu.cn", ".ac.uk")):
        return "academic", 0.86
    return "web", 0.62
