from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import regex

from .normalization import normalize_token


@dataclass(frozen=True)
class EntitySpan:
    start: int
    end: int
    normalized: str
    entity_type: str
    priority: int


_PATTERNS: Sequence[Tuple[str, int, regex.Pattern[str]]] = (
    (
        "url",
        100,
        regex.compile(
            r"(?i)(?<![\p{Latin}\p{N}_])(?:https?://|www\.)[^\s<>\"']+",
            regex.VERSION1,
        ),
    ),
    (
        "email",
        95,
        regex.compile(
            r"(?i)(?<![\p{Latin}\p{N}._%+-])[\p{Latin}\p{N}._%+-]+@"
            r"[\p{Latin}\p{N}.-]+\.[\p{Latin}]{2,}",
            regex.VERSION1,
        ),
    ),
    ("quoted", 90, regex.compile(r"(?P<q>[\"'])(?P<value>[^\"'\n]{2,80})(?P=q)", regex.VERSION1)),
    (
        "symbol_name",
        88,
        regex.compile(
            r"(?<![\p{Latin}\p{N}_])[\p{Latin}][\p{Latin}\p{N}_]*(?:\+\+|#)"
            r"(?![\p{Latin}\p{N}_])",
            regex.VERSION1,
        ),
    ),
    (
        "dot_name",
        86,
        regex.compile(
            r"(?<![\p{Latin}\p{N}_])\.[\p{Latin}][\p{Latin}\p{N}_-]{1,15}"
            r"(?![\p{Latin}\p{N}_])",
            regex.VERSION1,
        ),
    ),
    (
        "structured_identifier",
        84,
        regex.compile(
            r"(?<![\p{Latin}\p{N}_])[\p{Latin}\p{N}][\p{Latin}\p{N}]*"
            r"(?:[._+#-]+[\p{Latin}\p{N}]+)+(?![\p{Latin}\p{N}_])",
            regex.VERSION1,
        ),
    ),
    (
        "domain",
        82,
        regex.compile(
            r"(?i)(?<![\p{Latin}\p{N}_])(?:[\p{Latin}\p{N}]"
            r"(?:[\p{Latin}\p{N}-]{0,61}[\p{Latin}\p{N}])?\.)+"
            r"[\p{Latin}]{2,24}(?![\p{Latin}\p{N}_])",
            regex.VERSION1,
        ),
    ),
    (
        "alphanumeric_identifier",
        81,
        regex.compile(
            r"(?<![\p{Latin}\p{N}_])(?=[\p{Latin}\p{N}]*\p{Latin})"
            r"(?=[\p{Latin}\p{N}]*\p{N})[\p{Latin}\p{N}]{2,}"
            r"(?![\p{Latin}\p{N}_])",
            regex.VERSION1,
        ),
    ),
    (
        "date_or_number",
        60,
        regex.compile(
            r"(?<![\p{L}\p{N}])(?:\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?|"
            r"\d+(?:\.\d+)?(?:%|％|万|亿|kb|mb|gb|tb|b)?)(?![\p{L}\p{N}])",
            regex.IGNORECASE | regex.VERSION1,
        ),
    ),
)


class EntityProtector:
    """Protect structured terms before language-specific segmentation."""

    def __init__(self, protected_terms: Iterable[str] = ()) -> None:
        normalized = {normalize_token(item) for item in protected_terms}
        self.protected_terms = tuple(sorted((item for item in normalized if item), key=len, reverse=True))

    def find(self, text: str) -> Tuple[EntitySpan, ...]:
        candidates: List[EntitySpan] = []
        for entity_type, priority, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span("value") if entity_type == "quoted" else match.span()
                if entity_type == "url":
                    while end > start and text[end - 1] in ".,;:!?)]}，。；：！？）】":
                        end -= 1
                value = text[start:end]
                if value:
                    candidates.append(EntitySpan(start, end, value, entity_type, priority))

        for term in self.protected_terms:
            cursor = 0
            while True:
                start = text.find(term, cursor)
                if start < 0:
                    break
                end = start + len(term)
                if self._valid_boundary(text, start, end, term):
                    candidates.append(EntitySpan(start, end, term, "dictionary", 92))
                cursor = start + 1

        selected: List[EntitySpan] = []
        occupied = [False] * len(text)
        for item in sorted(candidates, key=lambda value: (-value.priority, -(value.end - value.start), value.start)):
            if any(occupied[item.start:item.end]):
                continue
            selected.append(item)
            for index in range(item.start, item.end):
                occupied[index] = True
        return tuple(sorted(selected, key=lambda item: (item.start, item.end)))

    @staticmethod
    def _valid_boundary(text: str, start: int, end: int, term: str) -> bool:
        def ascii_word(char: str) -> bool:
            return char.isascii() and (char.isalnum() or char == "_")

        if ascii_word(term[0]) and start and ascii_word(text[start - 1]):
            return False
        if ascii_word(term[-1]) and end < len(text) and ascii_word(text[end]):
            return False
        return True
