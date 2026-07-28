from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..semantic_selection import PairScorer, rank_capabilities
from ..text import search_tokens


@dataclass(frozen=True)
class SourceCapability:
    name: str
    description: str
    always: bool = False


class SourceSelector:
    """Select adapters by their declared capabilities, not query branches."""

    def __init__(self, *, scorer: PairScorer | None = None) -> None:
        self.scorer = scorer

    def select(
        self,
        query: str,
        configured: Sequence[str],
        capabilities: Mapping[str, SourceCapability],
        *,
        max_optional: int = 2,
    ) -> tuple[str, ...]:
        allowed = tuple(
            dict.fromkeys(str(value).strip().casefold() for value in configured if str(value).strip())
        )
        selected = [
            name
            for name in allowed
            if name in capabilities and capabilities[name].always
        ]
        optional = {
            name: capabilities[name].description
            for name in allowed
            if name in capabilities and not capabilities[name].always
        }
        if self.scorer is None and optional:
            query_terms = set(search_tokens(query))
            baseline_terms = {
                term
                for name in selected
                for term in search_tokens(capabilities[name].description)
            }
            optional = {
                name: description
                for name, description in optional.items()
                if query_terms.intersection(
                    set(search_tokens(description)) - baseline_terms
                )
            }
        ranked = rank_capabilities(
            str(query or ""),
            optional,
            scorer=self.scorer,
            limit=max(0, int(max_optional)),
        )
        return tuple(dict.fromkeys([*selected, *ranked]))
