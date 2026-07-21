from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import regex

from .core import AnalyzerCore
from .cleaning import clean_query_surface
from .models import AnalyzedToken, QueryAnalysis, TraceEvent


_NOISE_TERMS = {
    "什么", "怎么", "怎样", "如何", "为什么", "哪些", "哪个", "请问", "一下", "帮我",
    "搜索", "查询", "介绍", "介绍一下", "what", "which", "how", "please",
}
_FRESH_REALTIME = regex.compile(r"今天|今日|现在|当前|实时|刚刚|此刻|\btoday\b|\bnow\b", regex.I)
_FRESH_LATEST = regex.compile(r"最新|最近|近期|目前|进展|新版本|\blatest\b|\brecent\b", regex.I)
_TIME_TERM = regex.compile(
    r"\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?|"
    r"\d{4}年|本周|上周|本月|上月|今年|去年|过去\d+(?:天|周|月|年)",
    regex.I,
)
_SITE_FILTER = regex.compile(r"\bsite:([^\s]+)", regex.I)
_CLAUSE_SPLIT = regex.compile(r"[。！？!?；;\n]+|(?<=[，,])\s*", regex.VERSION1)
_ANSWER_TYPE = regex.compile(
    r"是什么\s*(国家|组织|机构|公司|项目|产品|技术|语言|疾病|人物)[？?。.!！\s]*$",
    regex.I,
)


class QueryAnalyzer:
    """Deterministic query analyzer that does not classify business domains."""

    def __init__(
        self,
        core: Optional[AnalyzerCore] = None,
        *,
        idf_lookup: Optional[Callable[[str], float]] = None,
        long_query_chars: int = 80,
        long_query_terms: int = 24,
        max_queries: int = 3,
    ) -> None:
        self.core = core or AnalyzerCore()
        self.idf_lookup = idf_lookup
        self.long_query_chars = long_query_chars
        self.long_query_terms = long_query_terms
        self.max_queries = max(1, max_queries)

    def analyze(self, query: str, *, resolved_query: Optional[str] = None) -> QueryAnalysis:
        started = time.perf_counter()
        supplied = resolved_query if resolved_query is not None else query
        cleaned = clean_query_surface(supplied)
        resolved = cleaned.text
        field = self.core.analyze(resolved, keep_duplicates=False, include_bigrams=True)
        tokens = tuple(self._weight_token(token) for token in field.tokens)
        exact_terms = self._unique(token.normalized for token in tokens if token.kind == "exact")
        word_terms = self._unique(
            token.normalized for token in tokens if token.kind in {"word", "number", "symbol"}
        )
        bigram_terms = self._unique(token.normalized for token in tokens if token.kind == "bigram")
        constraints = self._constraints(resolved, original=query)
        content_terms = [token for token in tokens if token.kind != "bigram" and token.weight > 0.15]
        needs_multi_query = len(field.normalized) >= self.long_query_chars or len(content_terms) > self.long_query_terms
        search_queries = self._search_queries(resolved, tokens, needs_multi_query)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        trace = (
            TraceEvent(
                "clean_query",
                {
                    "changed": cleaned.changed,
                    "operations": list(cleaned.operations),
                    "resolved_query": resolved,
                },
            ),
        ) + field.trace + (
            TraceEvent("query_constraints", constraints),
            TraceEvent(
                "query_plan",
                {
                    "needs_multi_query": needs_multi_query,
                    "search_queries": list(search_queries),
                    "content_terms": len(content_terms),
                },
            ),
        )
        return QueryAnalysis(
            original=query,
            resolved_query=resolved,
            normalized=field.normalized,
            tokens=tokens,
            exact_terms=exact_terms,
            word_terms=word_terms,
            bigram_terms=bigram_terms,
            constraints=constraints,
            scripts=field.scripts,
            needs_multi_query=needs_multi_query,
            search_queries=search_queries,
            elapsed_ms=elapsed_ms,
            trace=trace,
        )

    def _weight_token(self, token: AnalyzedToken) -> AnalyzedToken:
        weight = token.weight
        if token.normalized in _NOISE_TERMS:
            weight = min(weight, 0.1)
        elif token.protected:
            weight *= 1.15
        if self.idf_lookup and token.kind != "bigram":
            idf = max(0.0, float(self.idf_lookup(token.normalized)))
            weight *= 1.0 + min(1.5, idf / 8.0)
        return replace(token, weight=round(weight, 4))

    def _constraints(self, value: str, *, original: str = "") -> Dict[str, Any]:
        constraints: Dict[str, Any] = {}
        if _FRESH_REALTIME.search(value):
            constraints["freshness"] = "realtime"
        elif _FRESH_LATEST.search(value):
            constraints["freshness"] = "latest"
        time_terms = self._unique(match.group(0) for match in _TIME_TERM.finditer(value))
        if time_terms:
            constraints["time_terms"] = list(time_terms)
        sites = self._unique(match.group(1).casefold().rstrip(".,;，。；") for match in _SITE_FILTER.finditer(value))
        if sites:
            constraints["site"] = list(sites)
        answer_type = _ANSWER_TYPE.search(original)
        if answer_type:
            constraints["answer_type"] = answer_type.group(1).casefold()
        return constraints

    def _search_queries(
        self,
        resolved: str,
        tokens: Sequence[AnalyzedToken],
        multi: bool,
    ) -> Tuple[str, ...]:
        if not resolved:
            return ()
        resolved = resolved.strip()
        if not resolved:
            return ()
        if not multi:
            return (resolved,)

        protected = [token for token in tokens if token.kind == "exact"]
        content = [
            token for token in tokens if token.kind in {"word", "number"} and token.weight > 0.15
        ]
        ranked = sorted(
            content,
            key=lambda token: (-token.weight, -min(len(token.normalized), 12), token.start),
        )
        primary_terms = self._unique(
            [token.normalized for token in protected]
            + [token.normalized for token in ranked[:16]]
        )
        candidates: List[str] = [" ".join(primary_terms)] if primary_terms else []
        clauses = self._clauses(resolved)
        scored_clauses = []
        for clause, start, end in clauses:
            clause_tokens = [token for token in tokens if token.start >= start and token.end <= end]
            score = sum(
                token.weight + min(len(token.normalized), 8) * 0.05
                for token in clause_tokens
                if token.kind != "bigram" and token.normalized not in _NOISE_TERMS
            )
            scored_clauses.append((score, len(clause), clause))
        scored_clauses.sort(key=lambda item: (-item[0], -item[1]))
        candidates.extend(item[2] for item in scored_clauses)
        return self._unique(item for item in candidates if item)[: self.max_queries]

    @staticmethod
    def _clauses(value: str) -> List[Tuple[str, int, int]]:
        output: List[Tuple[str, int, int]] = []
        cursor = 0
        for delimiter in _CLAUSE_SPLIT.finditer(value):
            QueryAnalyzer._append_clause(output, value, cursor, delimiter.start())
            cursor = delimiter.end()
        QueryAnalyzer._append_clause(output, value, cursor, len(value))
        return output

    @staticmethod
    def _append_clause(output: List[Tuple[str, int, int]], value: str, start: int, end: int) -> None:
        while start < end and (value[start].isspace() or value[start] in "，,"):
            start += 1
        while end > start and (value[end - 1].isspace() or value[end - 1] in "，,"):
            end -= 1
        if start < end:
            output.append((value[start:end], start, end))

    @staticmethod
    def _unique(values: Iterable[str]) -> Tuple[str, ...]:
        output: List[str] = []
        seen = set()
        for value in values:
            if value and value not in seen:
                seen.add(value)
                output.append(value)
        return tuple(output)
