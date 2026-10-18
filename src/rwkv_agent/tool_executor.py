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
        long_text: Any,
        session_text: Any,
    ) -> None:
        self.web = web
        self.knowledge = knowledge
        self.long_text = long_text
        self.session_text = session_text

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
        if name == "long_text_qa":
            return self._execute_long_text(arguments, session_id=session_id)
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

    def _execute_long_text(
        self,
        arguments: dict[str, Any],
        *,
        session_id: str,
    ) -> dict[str, Any]:
        if set(arguments) != {"question"}:
            return {
                "status": "invalid",
                "evidence": [],
                "message": "long_text_qa accepts exactly one argument: question.",
            }
        pasted = self.session_text.get(session_id)
        if pasted is None:
            return {
                "status": "empty",
                "evidence": [],
                "message": (
                    "No pasted long text is active in this session. "
                    "Paste the text first, then ask a question."
                ),
            }
        return self.long_text.execute(
            pasted.text,
            str(arguments.get("question") or ""),
            document_name=pasted.name,
        )
