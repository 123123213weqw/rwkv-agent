from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Tuple

from .pipeline.query_compiler import QueryCompiler, QueryHints


@dataclass(frozen=True)
class SearchRequest:
    raw_query: str
    model_query: str
    execution_queries: Tuple[str, ...]
    entities: Tuple[str, ...]
    freshness: str
    time_terms: Tuple[str, ...]
    source_policy: str
    source_preference: str
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
    """Compatibility facade over the domain-neutral :class:`QueryCompiler`.

    P4 owns semantic query formation.  This builder no longer runs a second
    keyword classifier for freshness, source type, or user intent.  Callers
    with explicit UI/tool constraints may pass ``QueryHints``.
    """

    _SOURCE_POLICY = {
        "any": "any",
        "primary": "primary_required",
        "official": "official_preferred",
        "original": "original_source",
    }

    def __init__(
        self,
        analyzer: Optional[Any] = None,
        *,
        compiler: QueryCompiler | None = None,
    ) -> None:
        # ``analyzer`` remains accepted for source compatibility. Semantic
        # constraints are intentionally not inferred from it anymore.
        self.analyzer = analyzer
        self.compiler = compiler or QueryCompiler()

    def build(
        self,
        raw_query: str,
        model_query: str,
        *,
        hints: QueryHints | None = None,
        preserve_raw_query: bool = False,
        raw_query_max_characters: int = 256,
    ) -> SearchRequest:
        compiled = self.compiler.compile(raw_query, model_query, hints=hints)
        execution_queries = list(compiled.execution_queries)
        trace = list(compiled.trace)
        if preserve_raw_query and compiled.model_query:
            raw_hints = replace(
                hints or QueryHints(),
                sites=compiled.sites,
            )
            raw_compiled = self.compiler.compile(raw_query, "", hints=raw_hints)
            raw_execution = raw_compiled.execution_queries[0]
            if raw_execution in execution_queries:
                raw_lane_status = "duplicate"
            elif len(raw_execution) > max(1, int(raw_query_max_characters)):
                raw_lane_status = "skipped_too_long"
            else:
                execution_queries.append(raw_execution)
                raw_lane_status = "added"
            trace.append(
                {
                    "stage": "raw_query_recall_lane",
                    "enabled": True,
                    "status": raw_lane_status,
                    "max_characters": max(1, int(raw_query_max_characters)),
                }
            )
        return SearchRequest(
            raw_query=compiled.raw_query,
            model_query=compiled.model_query,
            execution_queries=tuple(execution_queries),
            entities=(),
            freshness=compiled.freshness,
            time_terms=compiled.time_terms,
            source_policy=self._SOURCE_POLICY[compiled.source_preference],
            source_preference=compiled.source_preference,
            sites=compiled.sites,
            depth=compiled.depth,
            trace=tuple(trace),
        )


__all__ = ["QueryHints", "SearchRequest", "SearchRequestBuilder"]
