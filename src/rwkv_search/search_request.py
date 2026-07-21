from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .analysis import QueryAnalyzer
from .g1i_tool_call import important_entities


_OFFICIAL_RE = re.compile(
    r"官网|官方网站|官方(?:网站|文档|公告|政策|来源)?|以.+?为准|"
    r"\bofficial\b|\baccording to\b|\bprimary source\b",
    re.I,
)
_ORIGINAL_RE = re.compile(r"原文|首发|最初来源|\boriginal source\b|\bfirst published\b", re.I)
_CITATION_RE = re.compile(r"来源|引用|出处|链接|\bcite\b|\bsources?\b|\breferences?\b", re.I)
_FRESH_RE = re.compile(
    r"今天|今日|现在|当前|实时|刚刚|最新|最近|近期|目前|本周|这周|本月|"
    r"\btoday\b|\bnow\b|\bcurrent(?:ly)?\b|\blatest\b|\bnewest\b|\brecent(?:ly)?\b",
    re.I,
)
_TIME_RE = re.compile(
    r"\d{4}(?:[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?)?|"
    r"最近一个月|过去\d+(?:天|周|月|年)|今天|明天|本周|这周|本月|今年|去年|"
    r"\btoday\b|\btomorrow\b|\bthis (?:week|month|year)\b|\blast (?:week|month|year)\b",
    re.I,
)
_SITE_RE = re.compile(r"\bsite:([^\s]+)", re.I)


@dataclass(frozen=True)
class SearchRequest:
    raw_query: str
    model_query: str
    execution_queries: Tuple[str, ...]
    entities: Tuple[str, ...]
    freshness: str
    time_terms: Tuple[str, ...]
    source_policy: str
    sites: Tuple[str, ...]
    depth: str
    trace: Tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["execution_queries"] = list(self.execution_queries)
        value["entities"] = list(self.entities)
        value["time_terms"] = list(self.time_terms)
        value["sites"] = list(self.sites)
        value["trace"] = list(self.trace)
        return value


class SearchRequestBuilder:
    """Use the model query as the search expression and preserve only hard constraints.

    The original user utterance remains available as metadata, but it is not sent
    to a search engine after a successful P4 rewrite.  Replaying both forms was a
    major source of broad, first-word matches on long natural-language questions.
    """

    def __init__(self, analyzer: Optional[QueryAnalyzer] = None) -> None:
        self.analyzer = analyzer or QueryAnalyzer()

    def build(self, raw_query: str, model_query: str) -> SearchRequest:
        raw = self.analyzer.analyze(raw_query)
        model = self.analyzer.analyze(model_query)
        freshness = str(raw.constraints.get("freshness") or "stable")
        sites = self._unique(
            list(raw.constraints.get("site") or ())
            + [match.group(1).rstrip(".,;，。；") for match in _SITE_RE.finditer(raw_query)]
        )
        time_terms = self._unique(
            list(raw.constraints.get("time_terms") or ())
            + [match.group(0) for match in _TIME_RE.finditer(raw_query)]
        )
        source_policy = self._source_policy(raw_query)
        entities = self._unique(important_entities(raw_query))
        cleaned_model_query = self._clean_query(model_query)
        primary_query = cleaned_model_query or self._clean_query(raw.resolved_query)
        enriched = self._enrich(
            primary_query,
            freshness=freshness,
            time_terms=time_terms,
            source_policy=source_policy,
            sites=sites,
        )
        execution_queries = self._unique((self._clean_query(enriched),))
        depth = "single"
        trace = (
            {
                "stage": "raw_analysis",
                "resolved_query": raw.resolved_query,
                "constraints": dict(raw.constraints),
                "search_queries": list(raw.search_queries),
            },
            {
                "stage": "model_analysis",
                "resolved_query": model.resolved_query,
                "search_queries": list(model.search_queries),
            },
            {
                "stage": "query_selection",
                "primary_source": "model_query" if cleaned_model_query else "raw_fallback",
                "raw_query_executed": not bool(cleaned_model_query),
                "depth_policy": "ordinary_single_pass",
            },
            {
                "stage": "constraint_merge",
                "entities": list(entities),
                "freshness": freshness,
                "time_terms": list(time_terms),
                "source_policy": source_policy,
                "sites": list(sites),
                "execution_queries": list(execution_queries),
            },
        )
        return SearchRequest(
            raw_query=raw_query.strip(),
            model_query=model_query.strip(),
            execution_queries=execution_queries,
            entities=entities,
            freshness=freshness,
            time_terms=time_terms,
            source_policy=source_policy,
            sites=sites,
            depth=depth,
            trace=trace,
        )

    @staticmethod
    def _source_policy(value: str) -> str:
        if _ORIGINAL_RE.search(value):
            return "original_source"
        if _OFFICIAL_RE.search(value):
            return "official_preferred"
        if _CITATION_RE.search(value):
            return "citations_required"
        return "any"

    @staticmethod
    def _enrich(
        query: str,
        *,
        freshness: str,
        time_terms: Tuple[str, ...],
        source_policy: str,
        sites: Tuple[str, ...],
    ) -> str:
        parts = [query.strip()]
        folded = query.casefold()
        for term in time_terms:
            if term.casefold() not in folded:
                parts.append(term)
        if freshness in {"latest", "realtime"} and not _FRESH_RE.search(query):
            parts.append("latest")
        if source_policy == "official_preferred" and not _OFFICIAL_RE.search(query):
            parts.append("official")
        elif source_policy == "original_source" and not _ORIGINAL_RE.search(query):
            parts.append("original source")
        for site in sites:
            marker = f"site:{site}"
            if marker.casefold() not in folded:
                parts.append(marker)
        return " ".join(part for part in parts if part)

    @staticmethod
    def _clean_query(value: str) -> str:
        return " ".join(value.strip(" ？?。.!！,，;；").split())

    @staticmethod
    def _unique(values: Iterable[str]) -> Tuple[str, ...]:
        output: List[str] = []
        seen = set()
        for value in values:
            clean = str(value or "").strip()
            key = clean.casefold()
            if clean and key not in seen:
                seen.add(key)
                output.append(clean)
        return tuple(output)
