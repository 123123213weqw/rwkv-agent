from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

from .pipeline.query_compiler import normalize_source_preference


@dataclass
class RouteDecision:
    intent: str
    tools: List[str]
    freshness: str
    depth: str
    needs_clarification: bool
    queries: List[str]
    missing_context: List[str]
    reason: str
    source_preference: str = "any"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SemanticRoute = Callable[[str], Mapping[str, Any] | RouteDecision]


class RuleRouter:
    """Compatibility router with no topic/intent keyword table.

    The legacy class name is retained for API compatibility.  Only clock
    expressions are handled deterministically; all semantic search decisions
    must come from the injected model resolver.  Without a resolver the safe
    default is ordinary chat.
    """

    CLOCK = re.compile(
        r"(?:星期几|周几|礼拜几|几点|几号|什么日期|当前时间|现在时间|"
        r"\bweekday\b|\bwhat\s+time\b|\bcurrent\s+date\b)",
        re.I,
    )

    def __init__(self, semantic_resolver: SemanticRoute | None = None) -> None:
        self.semantic_resolver = semantic_resolver

    def route(self, query: str, timezone: Optional[str] = "Asia/Shanghai") -> RouteDecision:
        clean = " ".join(str(query or "").strip().split())
        if not clean:
            raise ValueError("query must not be empty")
        if self.CLOCK.search(clean):
            missing = [] if timezone and timezone != "unknown" else ["timezone"]
            return RouteDecision(
                intent="time",
                tools=["clock"],
                freshness="realtime",
                depth="direct",
                needs_clarification=bool(missing),
                queries=[],
                missing_context=missing,
                reason="clock expressions use the timezone-aware clock tool",
            )
        if self.semantic_resolver is not None:
            resolved = self.semantic_resolver(clean)
            if isinstance(resolved, RouteDecision):
                return resolved
            use_search = bool(resolved.get("use_tool"))
            query_value = str(resolved.get("query") or clean).strip()
            return RouteDecision(
                intent="search" if use_search else "chat",
                tools=["local_search", "web_search"] if use_search else [],
                freshness=str(resolved.get("freshness") or "stable"),
                depth=str(resolved.get("depth") or ("single" if use_search else "direct")),
                needs_clarification=False,
                queries=[query_value] if use_search else [],
                missing_context=[],
                reason=str(resolved.get("reason") or "semantic route resolver"),
                source_preference=normalize_source_preference(
                    str(resolved.get("source_preference") or "any")
                ),
            )
        return RouteDecision(
            intent="chat",
            tools=[],
            freshness="stable",
            depth="direct",
            needs_clarification=False,
            queries=[],
            missing_context=[],
            reason="semantic search need was not supplied; default to chat",
        )

    @staticmethod
    def _queries(query: str, **_: Any) -> List[str]:
        value = " ".join(str(query or "").strip(" ？?。.!！,，;；").split())
        return [value] if value else []
