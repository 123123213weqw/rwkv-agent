from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Mapping


SearchMode = Literal["auto", "always", "never"]


@dataclass(frozen=True)
class SearchNeedDecision:
    """The small contract between UI policy and the semantic Search Gate."""

    use_tool: bool
    label: str
    source: str
    reason: str
    forced: bool = False
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearchNeedGate:
    """Route search without classifying topics in application code.

    ``always`` and ``never`` are explicit UI controls.  ``auto`` delegates to
    the supplied semantic one-token gate.  No query keyword, domain, or intent
    table is consulted here.
    """

    VALID_MODES = frozenset({"auto", "always", "never"})

    def policy(self, message: str, *, mode: SearchMode = "auto") -> SearchNeedDecision | None:
        clean = str(message or "").strip()
        if not clean:
            raise ValueError("message must not be empty")
        if mode not in self.VALID_MODES:
            raise ValueError(f"unsupported search mode: {mode}")
        if mode == "always":
            return SearchNeedDecision(
                use_tool=True,
                label="tool",
                source="ui_policy",
                reason="user explicitly enabled search",
                forced=True,
                confidence=1.0,
            )
        if mode == "never":
            return SearchNeedDecision(
                use_tool=False,
                label="chat",
                source="ui_policy",
                reason="user explicitly disabled search",
                forced=True,
                confidence=1.0,
            )
        return None

    def decide(
        self,
        message: str,
        *,
        semantic_gate: Callable[..., Mapping[str, Any]],
        mode: SearchMode = "auto",
        **semantic_kwargs: Any,
    ) -> dict[str, Any]:
        policy = self.policy(message, mode=mode)
        if policy is not None:
            return policy.to_dict()
        result = dict(semantic_gate(str(message).strip(), **semantic_kwargs))
        if "use_tool" not in result:
            raise ValueError("semantic gate result must contain use_tool")
        result.setdefault("label", "tool" if result["use_tool"] else "chat")
        result.setdefault("source", "semantic_gate")
        result.setdefault("reason", "one-token semantic Search Gate")
        result.setdefault("forced", False)
        return result
