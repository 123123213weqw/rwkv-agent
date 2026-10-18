from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable, Literal


Freshness = Literal["stable", "latest", "realtime"]
SourcePreference = Literal["any", "primary", "official", "original"]
Depth = Literal["single", "multi"]

_SITE = re.compile(r"(?<!\w)site:([^\s]+)", re.I)
_SOURCE_PREFERENCE_ALIASES: dict[str, SourcePreference] = {
    "": "any",
    "any": "any",
    "primary": "primary",
    "primary_required": "primary",
    "official": "official",
    "official_preferred": "official",
    "official_required": "official",
    "original": "original",
    "original_source": "original",
    "original_required": "original",
}


def normalize_source_preference(value: str | None) -> SourcePreference:
    """Normalize external policy names without inferring intent from a query."""

    key = str(value or "").strip().casefold()
    try:
        return _SOURCE_PREFERENCE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"invalid source preference: {value}") from exc


@dataclass(frozen=True)
class QueryHints:
    """Explicit constraints supplied by UI/tool state, never inferred by topic rules."""

    freshness: Freshness = "stable"
    source_preference: SourcePreference = "any"
    sites: tuple[str, ...] = ()
    time_terms: tuple[str, ...] = ()
    depth: Depth = "single"


@dataclass(frozen=True)
class CompiledQuery:
    raw_query: str
    model_query: str
    execution_queries: tuple[str, ...]
    freshness: Freshness
    source_preference: SourcePreference
    sites: tuple[str, ...]
    time_terms: tuple[str, ...]
    depth: Depth
    trace: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in ("execution_queries", "sites", "time_terms", "trace"):
            value[field] = list(value[field])
        return value


class QueryCompiler:
    """Compile the model's P4 query without re-classifying the user request.

    The only syntax parsed here is the search-engine ``site:`` operator.  Time,
    freshness, and source requirements must either remain in the model query or
    arrive as explicit ``QueryHints``.  This avoids a second, regex-based intent
    parser silently changing a valid Tool Call.
    """

    def compile(
        self,
        raw_query: str,
        model_query: str,
        *,
        hints: QueryHints | None = None,
    ) -> CompiledQuery:
        raw = self._clean(raw_query)
        model = self._clean(model_query)
        if not raw and not model:
            raise ValueError("at least one query must be non-empty")
        hints = hints or QueryHints()
        self._validate_hints(hints)
        explicit_sites = self._unique(
            [*hints.sites, *self._sites(raw), *self._sites(model)]
        )
        primary = model or raw
        folded = primary.casefold()
        parts = [primary]
        for site in explicit_sites:
            marker = f"site:{site}"
            if marker.casefold() not in folded:
                parts.append(marker)
        execution = self._clean(" ".join(parts))
        return CompiledQuery(
            raw_query=raw,
            model_query=model,
            execution_queries=(execution,),
            freshness=hints.freshness,
            source_preference=hints.source_preference,
            sites=explicit_sites,
            time_terms=self._unique(hints.time_terms),
            depth=hints.depth,
            trace=(
                {
                    "stage": "model_query_selection",
                    "source": "model_query" if model else "raw_fallback",
                    "raw_query_executed": not bool(model),
                },
                {
                    "stage": "explicit_constraint_merge",
                    "origin": "ui_or_tool_state",
                    "freshness": hints.freshness,
                    "source_preference": hints.source_preference,
                    "sites": list(explicit_sites),
                    "time_terms": list(hints.time_terms),
                    "depth": hints.depth,
                },
            ),
        )

    @staticmethod
    def _validate_hints(hints: QueryHints) -> None:
        if hints.freshness not in {"stable", "latest", "realtime"}:
            raise ValueError("invalid freshness")
        normalize_source_preference(hints.source_preference)
        if hints.depth not in {"single", "multi"}:
            raise ValueError("invalid depth")

    @staticmethod
    def _sites(value: str) -> tuple[str, ...]:
        return QueryCompiler._unique(
            match.group(1).rstrip(".,;，。；").casefold()
            for match in _SITE.finditer(value)
        )

    @staticmethod
    def _clean(value: str) -> str:
        return " ".join(str(value or "").strip(" ？?。.!！,，;；").split())

    @staticmethod
    def _unique(values: Iterable[str]) -> tuple[str, ...]:
        output: list[str] = []
        seen: set[str] = set()
        for raw in values:
            value = str(raw or "").strip()
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                output.append(value)
        return tuple(output)
