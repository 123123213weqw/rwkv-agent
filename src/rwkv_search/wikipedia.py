from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import re
from typing import Iterable, Iterator, List, Sequence, Tuple


_EDIT_LINK_RE = re.compile(r"\[\[\s*(?:编辑|編輯|edit)\s*\]\]", re.I)
_MAGIC_WORD_RE = re.compile(r"__(?:NOTOC|NOEDITSECTION|TOC|FORCETOC)__", re.I)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_REF_RE = re.compile(r"<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/\s*>", re.I | re.S)
_TAG_RE = re.compile(r"</?(?:br|small|span|div|center|sup|sub)\b[^>]*>", re.I)
_VARIANT_RE = re.compile(r"-\{([^{}]{1,2000})\}-")
_BLANKS_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_HSPACE_RE = re.compile(r"[ \t\u00a0]+")
_BLOCK_RE = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.S)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.])\s+|\n+")
_SOFT_BOUNDARY_RE = re.compile(r"(?<=[，,：:、])\s*")
_HEADING_PUNCTUATION_RE = re.compile(r"[。！？!?；;，,：:]|https?://")
_DISAMBIGUATION_RE = re.compile(r"(?:可以|可能|通常)指(?:以下|下列|的是|：|:)")
_LIST_TITLE_RE = re.compile(r"(?:列表(?:/|$)|名单(?:/|$)|名錄(?:/|$)|年表(?:/|$)|一览(?:/|$)|清单(?:/|$))")
EXPLICIT_HEADING_PREFIX = "\u241e "


@dataclass(frozen=True)
class WikipediaArticle:
    page_id: str
    url: str
    title: str
    text: str


@dataclass(frozen=True)
class WikipediaChunk:
    doc_id: str
    page_id: str
    chunk_id: int
    title: str
    text: str
    headings: Tuple[str, ...]
    url: str
    snapshot_date: str
    page_type: str
    char_start: int
    char_end: int
    source: str = "wikipedia"
    language: str = "zh"
    modified_at: str = ""
    wikidata_id: str = ""
    wikiname: str = ""
    source_version: int = 0
    has_math: bool = False
    metadata_text: str = ""
    aliases: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["headings"] = list(self.headings)
        data["aliases"] = list(self.aliases)
        return data


@dataclass(frozen=True)
class _TextUnit:
    text: str
    start: int
    end: int
    heading: str = ""


def _select_variant(match: re.Match[str]) -> str:
    value = match.group(1)
    choices = {}
    fallback = ""
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            fallback = fallback or part
            continue
        key, candidate = part.split(":", 1)
        choices[key.strip().casefold()] = candidate.strip()
        fallback = fallback or candidate.strip()
    for key in ("zh-cn", "zh-hans", "zh-sg", "zh", "zh-tw", "zh-hant", "zh-hk"):
        if choices.get(key):
            return choices[key]
    return fallback


def _strip_leading_templates(value: str) -> str:
    """Remove complete MediaWiki templates only when they lead the article."""

    cursor = 0
    while True:
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if not value.startswith("{{", cursor):
            return value[cursor:] if cursor else value
        depth = 0
        index = cursor
        end = -1
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
                    end = index
                    break
                continue
            index += 1
        if end < 0 or end - cursor > 20_000:
            return value
        cursor = end


def clean_wikipedia_text(value: str) -> str:
    value = html.unescape(value or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u200b", "").replace("\ufeff", "").replace("\u00ad", "")
    value = _strip_leading_templates(value)
    value = _COMMENT_RE.sub("", value)
    value = _REF_RE.sub("", value)
    value = _TAG_RE.sub("", value)
    value = _EDIT_LINK_RE.sub("", value)
    value = _MAGIC_WORD_RE.sub("", value)
    value = _VARIANT_RE.sub(_select_variant, value)
    lines: List[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if stripped.startswith(("{|", "|}", "! colspan=", "| colspan=")):
            continue
        lines.append(_HSPACE_RE.sub(" ", line).strip())
    value = "\n".join(lines)
    value = _BLANKS_RE.sub("\n\n", value)
    return value.strip()


def classify_wikipedia_page(title: str, text: str) -> str:
    normalized_title = title.strip()
    if "消歧义" in normalized_title or "消歧義" in normalized_title:
        return "disambiguation"
    if _DISAMBIGUATION_RE.search(text[:400]):
        return "disambiguation"
    if _LIST_TITLE_RE.search(normalized_title):
        return "list"
    return "article"


class WikipediaChunker:
    """Paragraph-first chunking for the pre-cleaned Wikipedia Monthly corpus."""

    def __init__(
        self,
        *,
        target_chars: int = 700,
        max_chars: int = 900,
        overlap_chars: int = 100,
        min_tail_chars: int = 80,
    ) -> None:
        if not 0 < target_chars <= max_chars:
            raise ValueError("target_chars must be positive and <= max_chars")
        if not 0 <= overlap_chars < max_chars:
            raise ValueError("overlap_chars must be >= 0 and < max_chars")
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_tail_chars = max(0, min_tail_chars)

    def chunk(
        self,
        article: WikipediaArticle,
        *,
        snapshot_date: str,
    ) -> List[WikipediaChunk]:
        title = clean_wikipedia_text(article.title)
        text = clean_wikipedia_text(article.text)
        if not title or not text:
            return []
        page_type = classify_wikipedia_page(title, text)
        units = list(self._units(text))
        if not units:
            units = [_TextUnit(text=text, start=0, end=len(text))]
        groups = self._group_units(units)
        chunks: List[WikipediaChunk] = []
        for chunk_id, group in enumerate(groups):
            chunk_text = "\n\n".join(unit.text for unit in group).strip()
            if not chunk_text:
                continue
            headings = tuple(dict.fromkeys(unit.heading for unit in group if unit.heading))
            chunks.append(
                WikipediaChunk(
                    doc_id=f"{article.page_id}#{chunk_id}",
                    page_id=str(article.page_id),
                    chunk_id=chunk_id,
                    title=title,
                    text=chunk_text,
                    headings=headings,
                    url=article.url,
                    snapshot_date=snapshot_date,
                    page_type=page_type,
                    char_start=min(unit.start for unit in group),
                    char_end=max(unit.end for unit in group),
                )
            )
        return chunks

    def _units(self, text: str) -> Iterator[_TextUnit]:
        heading = ""
        for match in _BLOCK_RE.finditer(text):
            block = match.group(0).strip()
            if not block:
                continue
            start = match.start() + (len(match.group(0)) - len(match.group(0).lstrip()))
            if self._is_heading(block):
                heading = block
                continue
            for unit in self._split_long(block, start, heading):
                yield unit

    @staticmethod
    def _is_heading(value: str) -> bool:
        if value.startswith(EXPLICIT_HEADING_PREFIX):
            return "\n" not in value and len(value) <= 500
        if "\n" in value or not 1 < len(value) <= 40:
            return False
        if _HEADING_PUNCTUATION_RE.search(value):
            return False
        if value[0].isdigit() or value.startswith(("*", "#", "-", "•", "·")):
            return False
        return True

    def _split_long(self, value: str, base_start: int, heading: str) -> Iterator[_TextUnit]:
        if len(value) <= self.max_chars:
            yield _TextUnit(value, base_start, base_start + len(value), heading)
            return
        cursor = 0
        for sentence in filter(None, _SENTENCE_BOUNDARY_RE.split(value)):
            local = value.find(sentence, cursor)
            if local < 0:
                local = cursor
            cursor = local + len(sentence)
            if len(sentence) <= self.max_chars:
                yield _TextUnit(sentence.strip(), base_start + local, base_start + cursor, heading)
                continue
            yield from self._split_oversized_sentence(sentence, base_start + local, heading)

    def _split_oversized_sentence(self, value: str, base_start: int, heading: str) -> Iterator[_TextUnit]:
        pieces = [piece for piece in _SOFT_BOUNDARY_RE.split(value) if piece]
        buffer = ""
        offset = 0
        for piece in pieces:
            if buffer and len(buffer) + len(piece) > self.max_chars:
                start = value.find(buffer, offset)
                start = offset if start < 0 else start
                yield _TextUnit(buffer.strip(), base_start + start, base_start + start + len(buffer), heading)
                offset = start + len(buffer)
                buffer = ""
            if len(piece) > self.max_chars:
                for index in range(0, len(piece), self.max_chars):
                    fragment = piece[index : index + self.max_chars].strip()
                    if fragment:
                        start = value.find(fragment, offset)
                        start = offset if start < 0 else start
                        yield _TextUnit(fragment, base_start + start, base_start + start + len(fragment), heading)
                        offset = start + len(fragment)
            else:
                buffer += piece
        if buffer.strip():
            start = value.find(buffer, offset)
            start = offset if start < 0 else start
            yield _TextUnit(buffer.strip(), base_start + start, base_start + start + len(buffer), heading)

    def _group_units(self, units: Sequence[_TextUnit]) -> List[List[_TextUnit]]:
        groups: List[List[_TextUnit]] = []
        current: List[_TextUnit] = []
        current_chars = 0
        for unit in units:
            separator = 2 if current else 0
            proposed = current_chars + separator + len(unit.text)
            if current and (
                proposed > self.max_chars
                or (current_chars >= self.target_chars and proposed > self.target_chars)
            ):
                groups.append(current)
                current = self._overlap_tail(current)
                current_chars = sum(len(item.text) for item in current) + max(0, len(current) - 1) * 2
            current.append(unit)
            current_chars += (2 if len(current) > 1 else 0) + len(unit.text)
        if current:
            merged_chars = (
                sum(len(item.text) for item in groups[-1])
                + sum(len(item.text) for item in current if item not in groups[-1])
                + max(0, len(groups[-1]) + len(current) - 1) * 2
                if groups
                else 0
            )
            if groups and current_chars < self.min_tail_chars and merged_chars <= self.max_chars:
                groups[-1].extend(item for item in current if item not in groups[-1])
            else:
                groups.append(current)
        return groups

    def _overlap_tail(self, units: Sequence[_TextUnit]) -> List[_TextUnit]:
        if not self.overlap_chars:
            return []
        output: List[_TextUnit] = []
        size = 0
        for unit in reversed(units):
            if not output and len(unit.text) > self.overlap_chars:
                text = unit.text[-self.overlap_chars :]
                output.append(
                    _TextUnit(
                        text=text,
                        start=max(unit.start, unit.end - len(text)),
                        end=unit.end,
                        heading=unit.heading,
                    )
                )
                break
            if output and size + len(unit.text) > self.overlap_chars:
                break
            output.append(unit)
            size += len(unit.text)
            if size >= self.overlap_chars:
                break
        output.reverse()
        return output


def chunk_articles(
    articles: Iterable[WikipediaArticle],
    chunker: WikipediaChunker,
    *,
    snapshot_date: str,
) -> Iterator[WikipediaChunk]:
    for article in articles:
        yield from chunker.chunk(article, snapshot_date=snapshot_date)
