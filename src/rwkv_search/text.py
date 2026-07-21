from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Iterable, List, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit


TRACKING_KEYS = {
    "gclid",
    "fbclid",
    "msclkid",
    "dclid",
    "yclid",
    "mc_cid",
    "mc_eid",
    "spm",
    "ref_src",
}
TRACKING_PREFIXES = ("utm_", "pk_", "ga_")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
LATIN_RE = re.compile(r"[a-z0-9]+(?:[._+#-][a-z0-9]+)*", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
DATE_META_KEYS = {
    "article:published_time",
    "datepublished",
    "date",
    "pubdate",
    "publishdate",
    "dc.date",
}


@dataclass
class ExtractedDocument:
    title: str
    text: str
    links: List[str] = field(default_factory=list)
    canonical_url: Optional[str] = None
    published_at: Optional[str] = None
    language: Optional[str] = None


class _HTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "article",
        "main",
        "section",
        "div",
        "p",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "pre",
        "code",
        "blockquote",
        "td",
        "th",
        "br",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "form", "nav", "footer", "aside"}

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.skip_depth = 0
        self.parts: List[str] = []
        self.title_parts: List[str] = []
        self.links: List[str] = []
        self.in_title = False
        self.canonical_url: Optional[str] = None
        self.published_at: Optional[str] = None
        self.language: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: Sequence[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attrs_dict = {str(k).lower(): (v or "") for k, v in attrs}
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "html" and attrs_dict.get("lang"):
            self.language = attrs_dict["lang"][:16]
        if tag == "title":
            self.in_title = True
        if tag == "a" and attrs_dict.get("href"):
            self.links.append(urljoin(self.base_url, attrs_dict["href"]))
        if tag == "link" and "canonical" in attrs_dict.get("rel", "").lower():
            href = attrs_dict.get("href")
            if href:
                self.canonical_url = urljoin(self.base_url, href)
        if tag == "meta":
            key = (
                attrs_dict.get("property")
                or attrs_dict.get("name")
                or attrs_dict.get("itemprop")
                or ""
            ).lower()
            if key in DATE_META_KEYS and attrs_dict.get("content"):
                self.published_at = attrs_dict["content"][:64]
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = False
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        value = SPACE_RE.sub(" ", data).strip()
        if not value:
            return
        if self.in_title:
            self.title_parts.append(value)
        self.parts.append(value + " ")


def canonicalize_url(url: str) -> Optional[str]:
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return None
    host = parsed.hostname.rstrip(".").lower()
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    port = parsed.port
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    clean_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.casefold()
        if lowered in TRACKING_KEYS or any(lowered.startswith(prefix) for prefix in TRACKING_PREFIXES):
            continue
        clean_query.append((key, value))
    clean_query.sort()
    return urlunsplit((scheme, netloc, path, urlencode(clean_query, doseq=True), ""))


def _clean_text(value: str) -> str:
    lines = []
    last = None
    for raw in html.unescape(value).splitlines():
        line = SPACE_RE.sub(" ", raw).strip()
        if not line or line == last:
            continue
        lines.append(line)
        last = line
    return "\n".join(lines)


def extract_document(raw_html: str, url: str) -> ExtractedDocument:
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.bare_extraction(
            raw_html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
            deduplicate=True,
            with_metadata=True,
        )
        if extracted:
            if hasattr(extracted, "as_dict"):
                data = extracted.as_dict()
            elif isinstance(extracted, dict):
                data = extracted
            else:
                data = {}
            text_value = _clean_text(str(data.get("text") or data.get("raw_text") or ""))
            if len(text_value) >= 80:
                fallback = _HTMLTextExtractor(url)
                fallback.feed(raw_html)
                return ExtractedDocument(
                    title=str(data.get("title") or "").strip() or " ".join(fallback.title_parts).strip(),
                    text=text_value,
                    links=_canonical_links(fallback.links),
                    canonical_url=canonicalize_url(str(data.get("url") or fallback.canonical_url or url)),
                    published_at=str(data.get("date") or fallback.published_at or "") or None,
                    language=str(data.get("language") or fallback.language or "") or None,
                )
    except (ImportError, Exception):
        pass

    parser = _HTMLTextExtractor(url)
    parser.feed(raw_html)
    return ExtractedDocument(
        title=_clean_text(" ".join(parser.title_parts))[:500],
        text=_clean_text("".join(parser.parts)),
        links=_canonical_links(parser.links),
        canonical_url=canonicalize_url(parser.canonical_url or url),
        published_at=parser.published_at,
        language=parser.language,
    )


def _canonical_links(links: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for link in links:
        canonical = canonicalize_url(link)
        if canonical and canonical not in seen:
            seen.add(canonical)
            output.append(canonical)
    return output


def search_tokens(text: str) -> List[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: List[str] = []
    seen = set()

    def add(token: str) -> None:
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)

    for token in LATIN_RE.findall(normalized):
        if len(token) == 1 and token.isalpha():
            continue
        add(token)
    for match in CJK_RE.finditer(normalized):
        chars = list(match.group(0))
        if len(chars) == 1:
            add(chars[0])
        else:
            for index in range(len(chars) - 1):
                add(chars[index] + chars[index + 1])
            add("".join(chars[: min(8, len(chars))]))
    return tokens


def indexed_text(title: str, body: str) -> str:
    title_tokens = search_tokens(title)
    body_tokens = search_tokens(body)
    return " ".join(title_tokens * 3 + body_tokens)


def content_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", SPACE_RE.sub(" ", text)).strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def simhash64(text: str) -> str:
    tokens = search_tokens(text)
    if not tokens:
        return "0" * 16
    vector = [0] * 64
    for token in tokens[:20000]:
        value = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return f"{result:016x}"


def best_snippet(text: str, query: str, limit: int = 360) -> str:
    clean = SPACE_RE.sub(" ", text).strip()
    if len(clean) <= limit:
        return clean
    positions = []
    lowered = clean.casefold()
    for token in search_tokens(query):
        position = lowered.find(token.casefold())
        if position >= 0:
            positions.append(position)
    center = min(positions) if positions else 0
    start = max(0, center - limit // 3)
    end = min(len(clean), start + limit)
    prefix = "…" if start else ""
    suffix = "…" if end < len(clean) else ""
    return prefix + clean[start:end].strip() + suffix
