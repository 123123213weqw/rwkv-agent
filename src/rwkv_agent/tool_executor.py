"""Bounded Function Call execution independent from Agent orchestration."""

from __future__ import annotations

from typing import Any


class ToolExecutor:
    """Dispatch validated tool names to adapters with session isolation."""

    def __init__(
        self,
        *,
        web: Any,
        knowledge: Any,
    ) -> None:
        self.web = web
        self.knowledge = knowledge

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        session_id: str = "default",
        original_query: str | None = None,
    ) -> dict[str, Any]:
        if name == "web_search":
            return self.web.execute(
                str(arguments.get("query") or ""),
                original_query=original_query,
            )
        if name == "knowledge_search":
            query = str(arguments.get("query") or "").strip()
            if not query:
                return {
                    "status": "invalid",
                    "evidence": [],
                    "message": "knowledge_search requires a non-empty query.",
                }
            return self.knowledge.execute(query)
        if name in {"memory", "memory_save"}:
            return {
                "status": "disabled",
                "reason": "context_only_mode",
                "message": (
                    "Long-term memory is disabled. Only the current session "
                    "transcript is used as context."
                ),
            }
        return {
            "status": "invalid",
            "message": f"unknown tool: {name}",
        }

