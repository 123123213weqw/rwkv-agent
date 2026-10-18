from __future__ import annotations

from dataclasses import replace
import logging
import threading
from typing import Iterable, Iterator, List, Sequence, Tuple

import regex

from .entities import EntityProtector, EntitySpan
from .models import AnalyzedToken, FieldAnalysis, TraceEvent
from .normalization import NormalizedText, normalize_text, script_for, scripts_in


_RUN_RE = regex.compile(
    r"\p{Script=Han}+|"
    r"(?=[\p{Script=Latin}\p{M}\p{N}]*[\p{Script=Latin}\p{M}])"
    r"[\p{Script=Latin}\p{M}\p{N}]+(?:['’][\p{Script=Latin}\p{M}\p{N}]+)*|"
    r"[\p{Script=Cyrillic}\p{M}]+|"
    r"[\p{Script=Hangul}\p{M}]+|"
    r"[\p{Script=Hiragana}\p{M}]+|"
    r"[\p{Script=Katakana}\p{M}]+|"
    r"[\p{Script=Greek}\p{M}]+|"
    r"[\p{Script=Arabic}\p{M}]+|"
    r"[\p{Script=Hebrew}\p{M}]+|"
    r"[\p{Script=Thai}\p{M}]+|"
    r"\p{N}+(?:[.,]\p{N}+)*|"
    r"\p{Extended_Pictographic}",
    regex.VERSION1,
)
_SUBPART_RE = regex.compile(r"\p{Script=Han}+|[\p{L}\p{M}]+|\p{N}+", regex.VERSION1)
_INNER_STRUCTURED_RE = regex.compile(
    r"(?<![\p{Latin}\p{N}_])[\p{Latin}\p{N}][\p{Latin}\p{N}]*"
    r"(?:[._+#-]+[\p{Latin}\p{N}]+)+(?![\p{Latin}\p{N}_])",
    regex.VERSION1,
)
_HAN_RE = regex.compile(r"\p{Script=Han}+", regex.VERSION1)
_EMOJI_RE = regex.compile(r"\p{Extended_Pictographic}", regex.VERSION1)
_JIEBA_TOKENIZER = None
_JIEBA_LOCK = threading.Lock()


class _ChineseSegmenter:
    def __init__(self, enabled: bool = True) -> None:
        self._tokenizer = None
        self.name = "cjk-whole-run-fallback"
        if not enabled:
            return
        try:
            import jieba  # type: ignore

            global _JIEBA_TOKENIZER
            with _JIEBA_LOCK:
                if _JIEBA_TOKENIZER is None:
                    jieba.setLogLevel(logging.WARNING)
                    tokenizer = jieba.Tokenizer()
                    tokenizer.initialize()
                    _JIEBA_TOKENIZER = tokenizer
            self._tokenizer = _JIEBA_TOKENIZER
            self.name = "jieba"
        except (ImportError, OSError):
            self._tokenizer = None

    def segment(self, value: str) -> Iterator[Tuple[str, int, int]]:
        if self._tokenizer is not None:
            for word, start, end in self._tokenizer.tokenize(value, mode="default"):
                if word.strip():
                    yield word, int(start), int(end)
            return
        if value:
            yield value, 0, len(value)


class AnalyzerCore:
    """Shared normalization and tokenization without query-specific policy."""

    BASE_WEIGHTS = {"exact": 4.0, "number": 1.5, "word": 1.0, "bigram": 0.4, "symbol": 0.2}

    def __init__(self, *, protected_terms: Iterable[str] = (), enable_jieba: bool = True) -> None:
        self.entities = EntityProtector(protected_terms)
        self.chinese = _ChineseSegmenter(enable_jieba)

    def analyze(
        self,
        value: str,
        *,
        keep_duplicates: bool,
        include_bigrams: bool = True,
    ) -> FieldAnalysis:
        normalized = normalize_text(value)
        entities = self.entities.find(normalized.text)
        tokens: List[AnalyzedToken] = []
        protected_mask = [False] * len(normalized.text)

        for entity in entities:
            for index in range(entity.start, entity.end):
                protected_mask[index] = True
            tokens.append(self._token(normalized, entity.start, entity.end, "exact", protected=True))
            tokens.extend(self._entity_subparts(normalized, entity))
            if include_bigrams:
                tokens.extend(self._bigrams(normalized, entity.start, entity.end, protected=True))

        for match in _RUN_RE.finditer(normalized.text):
            start, end = match.span()
            for free_start, free_end in self._free_ranges(start, end, protected_mask):
                run = normalized.text[free_start:free_end]
                if not run:
                    continue
                if _HAN_RE.fullmatch(run):
                    for _, local_start, local_end in self.chinese.segment(run):
                        tokens.append(
                            self._token(normalized, free_start + local_start, free_start + local_end, "word")
                        )
                    if include_bigrams:
                        tokens.extend(self._bigrams(normalized, free_start, free_end))
                else:
                    kind = "number" if run[0].isdigit() else "symbol" if self._is_symbol(run) else "word"
                    tokens.append(self._token(normalized, free_start, free_end, kind))

        tokens.extend(self._mixed_adjacent_entities(normalized, tokens))

        tokens.sort(key=lambda token: (token.start, token.end, self._kind_order(token.kind), token.normalized))
        if not keep_duplicates:
            seen = set()
            unique: List[AnalyzedToken] = []
            for token in tokens:
                key = (token.kind, token.normalized)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(token)
            tokens = unique

        positions = {}
        positioned: List[AnalyzedToken] = []
        for token in tokens:
            position = positions.get(token.kind, 0)
            positioned.append(replace(token, position=position))
            positions[token.kind] = position + 1

        trace = (
            TraceEvent(
                "normalize",
                {"input_chars": len(value), "output_chars": len(normalized.text), "normalized": normalized.text},
            ),
            TraceEvent(
                "protect_entities",
                {
                    "count": len(entities),
                    "entities": [
                        {"value": item.normalized, "type": item.entity_type, "start": item.start, "end": item.end}
                        for item in entities
                    ],
                },
            ),
            TraceEvent(
                "segment",
                {
                    "segmenter": self.chinese.name,
                    "token_counts": {
                        kind: sum(1 for token in positioned if token.kind == kind)
                        for kind in ("exact", "word", "number", "bigram", "symbol")
                    },
                    "keep_duplicates": keep_duplicates,
                },
            ),
        )
        return FieldAnalysis(
            original=value,
            normalized=normalized.text,
            tokens=tuple(positioned),
            scripts=scripts_in(normalized.text),
            segmenter=self.chinese.name,
            trace=trace,
        )

    def _entity_subparts(self, normalized: NormalizedText, entity: EntitySpan) -> List[AnalyzedToken]:
        output: List[AnalyzedToken] = []
        value = normalized.text[entity.start:entity.end]
        for match in _INNER_STRUCTURED_RE.finditer(value):
            start, end = entity.start + match.start(), entity.start + match.end()
            part = normalized.text[start:end]
            if part != entity.normalized:
                output.append(self._token(normalized, start, end, "exact", protected=True))
        for match in _SUBPART_RE.finditer(value):
            start, end = entity.start + match.start(), entity.start + match.end()
            part = normalized.text[start:end]
            if part == entity.normalized:
                continue
            kind = "number" if part[0].isdigit() else "word"
            output.append(self._token(normalized, start, end, kind, protected=True))
        return output

    def _mixed_adjacent_entities(
        self,
        normalized: NormalizedText,
        tokens: Sequence[AnalyzedToken],
    ) -> List[AnalyzedToken]:
        """Protect adjacent uppercase identifiers and their CJK classifier.

        This handles open-ended mixed forms such as A股, B站, C语言 and
        RWKV模型 without maintaining a business-domain keyword table.
        """

        words = [token for token in tokens if token.kind == "word" and not token.protected]
        output: List[AnalyzedToken] = []
        for left, right in zip(words, words[1:]):
            if left.end != right.start or left.script != "latin" or right.script != "han":
                continue
            # Only compose single-letter classifiers (A股/B站/C语言). Longer
            # acronyms followed by ordinary Chinese, such as RNN一样, must stay
            # as separate terms rather than becoming a false exact entity.
            if not (len(left.normalized) == 1 and 1 <= len(right.normalized) <= 4):
                continue
            surface = left.surface
            if not surface or not any(char.isalpha() for char in surface) or surface != surface.upper():
                continue
            output.append(
                AnalyzedToken(
                    surface=normalized.original[left.start:right.end],
                    normalized=left.normalized + right.normalized,
                    start=left.start,
                    end=right.end,
                    position=0,
                    kind="exact",
                    script="mixed",
                    weight=self.BASE_WEIGHTS["exact"],
                    protected=True,
                )
            )
        return output

    def _bigrams(
        self,
        normalized: NormalizedText,
        start: int,
        end: int,
        *,
        protected: bool = False,
    ) -> List[AnalyzedToken]:
        output: List[AnalyzedToken] = []
        for match in _HAN_RE.finditer(normalized.text, start, end):
            for index in range(match.start(), match.end() - 1):
                output.append(self._token(normalized, index, index + 2, "bigram", protected=protected))
        return output

    def _token(
        self,
        normalized: NormalizedText,
        start: int,
        end: int,
        kind: str,
        *,
        protected: bool = False,
    ) -> AnalyzedToken:
        source_start, source_end = normalized.source_span(start, end)
        lemma = normalized.text[start:end]
        return AnalyzedToken(
            surface=normalized.original[source_start:source_end],
            normalized=lemma,
            start=source_start,
            end=source_end,
            position=0,
            kind=kind,
            script=script_for(lemma),
            weight=self.BASE_WEIGHTS[kind],
            protected=protected,
        )

    @staticmethod
    def _free_ranges(start: int, end: int, occupied: Sequence[bool]) -> Iterator[Tuple[int, int]]:
        cursor = start
        while cursor < end:
            while cursor < end and occupied[cursor]:
                cursor += 1
            free_start = cursor
            while cursor < end and not occupied[cursor]:
                cursor += 1
            if free_start < cursor:
                yield free_start, cursor

    @staticmethod
    def _kind_order(kind: str) -> int:
        return {"exact": 0, "word": 1, "number": 2, "bigram": 3, "symbol": 4}.get(kind, 9)

    @staticmethod
    def _is_symbol(value: str) -> bool:
        return bool(_EMOJI_RE.fullmatch(value))
