from __future__ import annotations

from dataclasses import dataclass, replace
import html
import json
import re
from typing import Any, Iterable, List, Tuple

from .wikipedia import (
    EXPLICIT_HEADING_PREFIX,
    WikipediaArticle,
    WikipediaChunk,
    WikipediaChunker,
    clean_wikipedia_text,
)


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://|/wiki/)[^)]*\)")
_BLANKS_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_HEADING_MARKUP_RE = re.compile(r"(?:\*\*|__|`)")
_VARIANT_VALUE_RE = re.compile(
    r"zh-(?:cn|hans|sg|my|tw|hant|hk|mo)\s*:\s*([^;|}\n]+)", re.I
)
_REDIRECT_TEMPLATE_RE = re.compile(
    r"\{\{\s*(?:redirect|重定向|重新導向|重新定向)\s*\|\s*([^|}\n]+)", re.I
)
_BOLD_ALIAS_RE = re.compile(r"'''([^'\n]{1,120})'''")
_WIKILINK_RE = re.compile(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]")
_FORWARD_NAME_RE = re.compile(
    r"(?:简称|簡稱|缩写|縮寫|又称|又稱|亦称|亦稱|通称|通稱|别称|別稱)"
    r"(?:为|為|作|是|：|:)?\s*[「『“\"]?([^，。；、」』”\"\n]{1,40})",
    re.I,
)
_REVERSE_NAME_RE = re.compile(
    r"[「『“\"]([^」』”\"\n]{1,40})[」』”\"](?:也)?(?:逐渐|逐漸)?"
    r"(?:成为|成為|是).{0,80}?(?:常见称呼|常見稱呼|通称|通稱|简称|簡稱|别称|別稱)",
    re.I,
)
_EXCLUDED_SECTIONS = {
    "参考", "参考资料", "參考資料", "参考文献", "參考文獻", "参考来源", "參考來源",
    "注释", "註釋", "脚注", "腳註", "备注", "備註", "来源", "來源",
    "外部链接", "外部連結", "外部联结", "延伸阅读", "延伸閱讀", "参见", "參見",
    "另见", "另見", "相关条目", "相關條目", "相关链接", "相關連結",
    "references", "notes", "footnotes", "external links", "see also", "further reading",
}
_EXCLUDED_SECTION_KEYS = {
    re.sub(r"[\s:：·•()（）\-—_]+", "", item).casefold() for item in _EXCLUDED_SECTIONS
}
_OPENCC_CONVERTERS: Any = None


@dataclass(frozen=True)
class FineWikiArticle:
    page_id: str
    url: str
    title: str
    text: str
    date_modified: str = ""
    wikidata_id: str = ""
    wikiname: str = "zhwiki"
    source_version: int = 0
    infoboxes: str = ""
    has_math: bool = False
    aliases: Tuple[str, ...] = ()


def language_from_wikiname(value: str) -> str:
    """Return the ISO-like language prefix used by a FineWiki subset."""

    normalized = (value or "").strip().casefold()
    return normalized[:-4] if normalized.endswith("wiki") and len(normalized) > 4 else normalized


def _plain_heading(value: str) -> str:
    value = _HEADING_MARKUP_RE.sub("", value)
    value = _LINK_RE.sub(r"\1", value)
    return value.strip().strip("#").strip()


def _heading_key(value: str) -> str:
    return re.sub(r"[\s:：·•()（）\-—_]+", "", value).casefold()


def clean_finewiki_markdown(title: str, value: str) -> str:
    """Convert FineWiki Markdown into paragraph-first text with explicit section blocks.

    The top-level title is removed because it is indexed separately. Reference-like
    tail sections are filtered even when the upstream extractor leaves localized
    headings behind.
    """

    value = html.unescape(value or "").replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u200b", "").replace("\ufeff", "").replace("\u00ad", "")
    value = _COMMENT_RE.sub("", value)
    value = _IMAGE_RE.sub(r"\1", value)
    value = _LINK_RE.sub(r"\1", value)
    value = _HTML_TAG_RE.sub("", value)
    output: List[str] = []
    excluded_level = 0
    normalized_title = _heading_key(clean_wikipedia_text(title))

    for raw_line in value.splitlines():
        line = raw_line.rstrip()
        match = _HEADING_RE.match(line.strip())
        if match:
            level = len(match.group(1))
            heading = _plain_heading(match.group(2))
            if excluded_level and level > excluded_level:
                continue
            excluded_level = 0
            if _heading_key(heading) in _EXCLUDED_SECTION_KEYS:
                excluded_level = level
                continue
            if level == 1 and _heading_key(heading) == normalized_title:
                continue
            if heading:
                output.extend(("", EXPLICIT_HEADING_PREFIX + heading, ""))
            continue
        if excluded_level:
            continue
        output.append(line)

    cleaned = "\n".join(output)
    cleaned = _BLANKS_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def flatten_infoboxes(value: str, *, max_chars: int = 4000) -> str:
    """Flatten the structured infobox JSON into a bounded retrieval-only field."""

    if not value or value.strip() in {"[]", "{}", "null"}:
        return ""
    try:
        parsed: Any = json.loads(value)
    except (TypeError, ValueError):
        return clean_wikipedia_text(value)[:max_chars]

    rows: List[str] = []

    def walk(item: Any, prefix: str = "") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).casefold() in {"image", "image_name", "image_url", "url"}:
                    continue
                walk(child, f"{prefix} {key}".strip())
            return
        if isinstance(item, list):
            for child in item:
                walk(child, prefix)
            return
        if item is None or isinstance(item, bool):
            return
        scalar = clean_wikipedia_text(str(item))
        if not scalar or scalar.startswith(("http://", "https://")):
            return
        rows.append(f"{prefix}: {scalar}" if prefix else scalar)

    walk(parsed)
    unique: List[str] = []
    seen = set()
    size = 0
    for row in rows:
        row = row.strip()
        if not row or row in seen:
            continue
        proposed = size + (1 if unique else 0) + len(row)
        if proposed > max_chars:
            break
        unique.append(row)
        seen.add(row)
        size = proposed
    return "\n".join(unique)


def _clean_alias(value: str) -> str:
    value = _WIKILINK_RE.sub(r"\1", html.unescape(value or ""))
    value = re.sub(r"<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/\s*>", "", value, flags=re.I | re.S)
    value = re.sub(r"\{\{[^{}]{0,300}\}\}", "", value)
    value = value.replace("'''", "").replace("''", "").strip(" ，,;；:：()（）[]【】")
    value = re.sub(r"\s+", " ", value).strip()
    if (
        not 2 <= len(value) <= 80
        or "http" in value.casefold()
        or "=" in value
        or any(marker in value for marker in ("{{", "}}", "[[", "]]", "-{", "}-"))
        or not re.search(r"[A-Za-z\u3400-\u9fff]", value)
        or value.count("(") != value.count(")")
        or value.count("（") != value.count("）")
    ):
        return ""
    return value


def _strip_leading_wikitext_templates(value: str) -> str:
    cursor = 0
    while True:
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if value.startswith("<!--", cursor):
            end = value.find("-->", cursor + 4)
            if end < 0:
                return value[cursor:]
            cursor = end + 3
            continue
        if not value.startswith("{{", cursor):
            return value[cursor:]
        depth = 0
        index = cursor
        while index < len(value) - 1:
            pair = value[index : index + 2]
            if pair == "{{":
                depth += 1
                index += 2
                continue
            if pair == "}}":
                depth -= 1
                index += 2
                if depth == 0:
                    cursor = index
                    break
                continue
            index += 1
        else:
            return value[cursor:]


def extract_wikitext_aliases(
    title: str,
    wikitext: str,
    *,
    rendered_text: str = "",
    max_aliases: int = 32,
) -> Tuple[str, ...]:
    """Extract corpus-derived aliases without maintaining a business keyword table.

    Chinese Wikipedia's ``noteTA`` templates retain regional variants that are
    absent from rendered text. Redirect hatnotes and bold lead synonyms provide
    two additional generic signals.
    """

    source = (wikitext or "")[:20_000]
    canonical = _clean_alias(title)
    candidates: List[str] = []
    lead = _strip_leading_wikitext_templates(source)
    leading_templates = source[: len(source) - len(lead)]
    for line in leading_templates.splitlines():
        variants = [_clean_alias(item) for item in _VARIANT_VALUE_RE.findall(line)]
        variants = list(dict.fromkeys(item for item in variants if item))
        if len(variants) < 2:
            continue
        for original in variants:
            if canonical and original in canonical:
                candidates.extend(canonical.replace(original, target) for target in variants)
        if canonical in variants:
            candidates.extend(variants)
    candidates.extend(_REDIRECT_TEMPLATE_RE.findall(leading_templates))
    lead = re.split(r"\n\s*==", lead, maxsplit=1)[0]
    lead = re.split(r"\n\s*\n", lead, maxsplit=1)[0][:2500]
    canonical_base = re.split(r"\s*[（(]", canonical, maxsplit=1)[0].strip()
    bold_aliases = _BOLD_ALIAS_RE.findall(lead)
    first_bold = _clean_alias(bold_aliases[0]) if bold_aliases else ""
    if (
        canonical_base
        and first_bold
        and (
            canonical_base.casefold() in first_bold.casefold()
            or first_bold.casefold() in canonical_base.casefold()
        )
    ):
        candidates.extend(bold_aliases)

    # FineWiki's rendered article retains useful naming sentences even when
    # the source template does not expose the short name as a bold alias.
    # These language-level patterns are corpus-derived and domain-independent.
    rendered_lead = clean_finewiki_markdown(title, rendered_text)[:6000]
    rendered_title = clean_wikipedia_text(title).casefold()
    for match in _FORWARD_NAME_RE.finditer(rendered_lead):
        sentence_start = max(
            rendered_lead.rfind(mark, 0, match.start()) for mark in "。！？!?；;\n"
        ) + 1
        prefix = rendered_lead[sentence_start : match.start()].casefold()
        if rendered_title and rendered_title in prefix:
            candidates.append(match.group(1))
    candidates.extend(_REVERSE_NAME_RE.findall(rendered_lead))

    output: List[str] = []
    canonical_key = canonical.casefold() if canonical else ""
    seen = {canonical_key} if canonical_key else set()
    for candidate in candidates:
        alias = _clean_alias(candidate)
        key = alias.casefold()
        if not alias or key in seen or (canonical_key and key in canonical_key):
            continue
        output.append(alias)
        seen.add(key)
        if len(output) >= max_aliases:
            break
    return tuple(output)


def expand_chinese_script_aliases(
    title: str,
    aliases: Iterable[str],
    *,
    max_aliases: int = 32,
) -> Tuple[str, ...]:
    """Add simplified/traditional variants when the optional OpenCC indexer is installed."""

    base = tuple(alias for alias in aliases if alias)
    global _OPENCC_CONVERTERS
    if _OPENCC_CONVERTERS is None:
        try:
            from opencc import OpenCC
            _OPENCC_CONVERTERS = (OpenCC("t2s"), OpenCC("s2t"))
        except ImportError:
            _OPENCC_CONVERTERS = ()
    if not _OPENCC_CONVERTERS:
        return base[:max_aliases]
    canonical = _clean_alias(title).casefold()
    output: List[str] = []
    seen = {canonical} if canonical else set()
    for alias in base:
        for candidate in (alias, *(converter.convert(alias) for converter in _OPENCC_CONVERTERS)):
            cleaned = _clean_alias(candidate)
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            output.append(cleaned)
            if len(output) >= max_aliases:
                return tuple(output)
    return tuple(output)


class FineWikiChunker:
    """Markdown-aware FineWiki adapter backed by the proven paragraph chunker."""

    def __init__(self, **kwargs: int) -> None:
        self.base = WikipediaChunker(**kwargs)

    def chunk(self, article: FineWikiArticle, *, snapshot_date: str = "20250801") -> List[WikipediaChunk]:
        title = clean_wikipedia_text(article.title)
        text = clean_finewiki_markdown(title, article.text)
        alias_label = "别名" if language_from_wikiname(article.wikiname) == "zh" else "Alias"
        metadata_rows = [f"{alias_label}: {alias}" for alias in article.aliases]
        infobox_text = flatten_infoboxes(article.infoboxes)
        if infobox_text:
            metadata_rows.append(infobox_text)
        metadata_text = "\n".join(metadata_rows)[:6000]
        if not text:
            text = metadata_text
        if not title or not text:
            return []
        chunks = self.base.chunk(
            WikipediaArticle(
                page_id=str(article.page_id),
                url=article.url,
                title=title,
                text=text,
            ),
            snapshot_date=snapshot_date,
        )
        output: List[WikipediaChunk] = []
        for index, chunk in enumerate(chunks):
            headings = tuple(
                heading.removeprefix(EXPLICIT_HEADING_PREFIX).strip()
                for heading in chunk.headings
            )
            output.append(
                replace(
                    chunk,
                    headings=headings,
                    source="finewiki",
                    language=language_from_wikiname(article.wikiname) or "und",
                    modified_at=article.date_modified,
                    wikidata_id=article.wikidata_id,
                    wikiname=article.wikiname,
                    source_version=int(article.source_version or 0),
                    has_math=bool(article.has_math),
                    metadata_text=metadata_text if index == 0 else "",
                    aliases=article.aliases if index == 0 else (),
                )
            )
        return output


def chunk_finewiki_articles(
    articles: Iterable[FineWikiArticle],
    chunker: FineWikiChunker,
    *,
    snapshot_date: str = "20250801",
) -> Iterable[WikipediaChunk]:
    for article in articles:
        yield from chunker.chunk(article, snapshot_date=snapshot_date)
