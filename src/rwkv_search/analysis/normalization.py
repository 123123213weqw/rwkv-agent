from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import regex


_GRAPHEME_RE = regex.compile(r"\X", regex.VERSION1)
_SCRIPT_PATTERNS: Sequence[Tuple[str, regex.Pattern[str]]] = (
    ("han", regex.compile(r"\p{Script=Han}", regex.VERSION1)),
    ("latin", regex.compile(r"\p{Script=Latin}", regex.VERSION1)),
    ("cyrillic", regex.compile(r"\p{Script=Cyrillic}", regex.VERSION1)),
    ("hangul", regex.compile(r"\p{Script=Hangul}", regex.VERSION1)),
    ("hiragana", regex.compile(r"\p{Script=Hiragana}", regex.VERSION1)),
    ("katakana", regex.compile(r"\p{Script=Katakana}", regex.VERSION1)),
    ("greek", regex.compile(r"\p{Script=Greek}", regex.VERSION1)),
    ("arabic", regex.compile(r"\p{Script=Arabic}", regex.VERSION1)),
    ("hebrew", regex.compile(r"\p{Script=Hebrew}", regex.VERSION1)),
    ("thai", regex.compile(r"\p{Script=Thai}", regex.VERSION1)),
    ("devanagari", regex.compile(r"\p{Script=Devanagari}", regex.VERSION1)),
)

_PUNCTUATION_MAP: Dict[str, str] = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
}


@dataclass(frozen=True)
class NormalizedText:
    original: str
    text: str
    source_spans: Tuple[Tuple[int, int], ...]

    def source_span(self, start: int, end: int) -> Tuple[int, int]:
        if not self.source_spans or start >= end:
            return (0, 0)
        start = max(0, min(start, len(self.source_spans) - 1))
        end = max(start + 1, min(end, len(self.source_spans)))
        spans = self.source_spans[start:end]
        return min(item[0] for item in spans), max(item[1] for item in spans)


def normalize_text(value: str) -> NormalizedText:
    """NFKC/casefold text while retaining a best-effort source offset map."""

    output: List[str] = []
    spans: List[Tuple[int, int]] = []
    pending_space: Tuple[int, int] | None = None
    for match in _GRAPHEME_RE.finditer(value):
        source_start, source_end = match.span()
        normalized = unicodedata.normalize("NFKC", match.group(0)).casefold()
        for char in normalized:
            char = _PUNCTUATION_MAP.get(char, char)
            category = unicodedata.category(char)
            if char.isspace() or category in {"Cc", "Cf", "Zl", "Zp", "Zs"}:
                if output and output[-1] != " ":
                    pending_space = (source_start, source_end)
                continue
            if pending_space is not None:
                output.append(" ")
                spans.append(pending_space)
                pending_space = None
            output.append(char)
            spans.append((source_start, source_end))
    if output and output[-1] == " ":
        output.pop()
        spans.pop()
    return NormalizedText(original=value, text="".join(output), source_spans=tuple(spans))


def normalize_token(value: str) -> str:
    return normalize_text(value).text.strip()


def char_script(char: str) -> str:
    if char.isdigit():
        return "number"
    for name, pattern in _SCRIPT_PATTERNS:
        if pattern.fullmatch(char):
            return name
    return "common"


def script_for(value: str) -> str:
    found = [name for name, pattern in _SCRIPT_PATTERNS if pattern.search(value)]
    if not found:
        return "number" if any(char.isdigit() for char in value) else "common"
    return "mixed" if len(found) > 1 else found[0]


def scripts_in(value: str) -> Tuple[str, ...]:
    return tuple(name for name, pattern in _SCRIPT_PATTERNS if pattern.search(value))
