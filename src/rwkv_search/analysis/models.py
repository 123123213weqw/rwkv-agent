from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class AnalyzedToken:
    """One token with both its source spelling and retrieval form."""

    surface: str
    normalized: str
    start: int
    end: int
    position: int
    kind: str
    script: str
    weight: float
    protected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceEvent:
    stage: str
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "data": dict(self.data)}


@dataclass(frozen=True)
class FieldAnalysis:
    original: str
    normalized: str
    tokens: Tuple[AnalyzedToken, ...]
    scripts: Tuple[str, ...]
    segmenter: str
    trace: Tuple[TraceEvent, ...] = ()

    def terms(self, *kinds: str) -> List[str]:
        accepted = set(kinds)
        return [token.normalized for token in self.tokens if token.kind in accepted]

    @property
    def exact_terms(self) -> List[str]:
        return self.terms("exact")

    @property
    def word_terms(self) -> List[str]:
        return self.terms("word", "number", "symbol")

    @property
    def bigram_terms(self) -> List[str]:
        return self.terms("bigram")

    def to_dict(self, *, include_tokens: bool = True, include_trace: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "original": self.original,
            "normalized": self.normalized,
            "scripts": list(self.scripts),
            "segmenter": self.segmenter,
            "exact_terms": self.exact_terms,
            "word_terms": self.word_terms,
            "bigram_terms": self.bigram_terms,
        }
        if include_tokens:
            data["tokens"] = [token.to_dict() for token in self.tokens]
        if include_trace:
            data["trace"] = [event.to_dict() for event in self.trace]
        return data


@dataclass(frozen=True)
class QueryAnalysis:
    original: str
    resolved_query: str
    normalized: str
    tokens: Tuple[AnalyzedToken, ...]
    exact_terms: Tuple[str, ...]
    word_terms: Tuple[str, ...]
    bigram_terms: Tuple[str, ...]
    constraints: Mapping[str, Any]
    scripts: Tuple[str, ...]
    needs_multi_query: bool
    search_queries: Tuple[str, ...]
    elapsed_ms: float
    trace: Tuple[TraceEvent, ...] = ()

    def to_dict(self, *, include_tokens: bool = True, include_trace: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "original": self.original,
            "resolved_query": self.resolved_query,
            "normalized": self.normalized,
            "exact_terms": list(self.exact_terms),
            "word_terms": list(self.word_terms),
            "bigram_terms": list(self.bigram_terms),
            "constraints": dict(self.constraints),
            "scripts": list(self.scripts),
            "needs_multi_query": self.needs_multi_query,
            "search_queries": list(self.search_queries),
            "elapsed_ms": self.elapsed_ms,
        }
        if include_tokens:
            data["tokens"] = [token.to_dict() for token in self.tokens]
        if include_trace:
            data["trace"] = [event.to_dict() for event in self.trace]
        return data


@dataclass(frozen=True)
class DocumentAnalysis:
    title: FieldAnalysis
    body: FieldAnalysis
    headings: Tuple[FieldAnalysis, ...] = ()
    url: FieldAnalysis | None = None
    elapsed_ms: float = 0.0

    def to_index_payload(self) -> Dict[str, Any]:
        """Return channel-separated fields for a whitespace-analyzed shadow index."""

        heading_exact: List[str] = []
        heading_words: List[str] = []
        heading_bigrams: List[str] = []
        for heading in self.headings:
            heading_exact.extend(heading.exact_terms)
            heading_words.extend(heading.word_terms)
            heading_bigrams.extend(heading.bigram_terms)
        return {
            "title_original": self.title.original,
            "title_normalized": self.title.normalized,
            "title_exact": self.title.exact_terms,
            "title_words": " ".join(self.title.word_terms),
            "title_bigrams": " ".join(self.title.bigram_terms),
            "heading_original": [heading.original for heading in self.headings],
            "heading_exact": heading_exact,
            "heading_words": " ".join(heading_words),
            "heading_bigrams": " ".join(heading_bigrams),
            "body_original": self.body.original,
            "body_exact": self.body.exact_terms,
            "body_words": " ".join(self.body.word_terms),
            "body_bigrams": " ".join(self.body.bigram_terms),
            "url_exact": self.url.exact_terms if self.url else [],
        }

    def to_dict(self, *, include_tokens: bool = True, include_trace: bool = True) -> Dict[str, Any]:
        return {
            "title": self.title.to_dict(include_tokens=include_tokens, include_trace=include_trace),
            "body": self.body.to_dict(include_tokens=include_tokens, include_trace=include_trace),
            "headings": [
                item.to_dict(include_tokens=include_tokens, include_trace=include_trace)
                for item in self.headings
            ],
            "url": self.url.to_dict(include_tokens=include_tokens, include_trace=include_trace)
            if self.url
            else None,
            "elapsed_ms": self.elapsed_ms,
        }
