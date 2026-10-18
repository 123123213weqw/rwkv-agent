"""Tool-routing decision service.

The controller owns request orchestration.  This module owns only the decision
contract between explicit UI policy and the model's semantic one-token gate.
"""

from __future__ import annotations

import time
from typing import Any

from .tool_protocol import policy_tool_gate


class ToolRouter:
    """Resolve chat-versus-tool without owning the model client."""

    def __init__(self, *, default_threshold: float = 0.7) -> None:
        self.default_threshold = self.validate_threshold(default_threshold)

    @staticmethod
    def validate_threshold(value: float) -> float:
        threshold = float(value)
        if not -20.0 <= threshold <= 20.0:
            raise ValueError("tool_gate_threshold out of range")
        return threshold

    def decide(
        self,
        model: Any,
        message: str,
        *,
        threshold: float | None = None,
        context: str = "",
        has_pasted_text: bool = False,
        search_mode: str = "auto",
    ) -> dict[str, Any]:
        clean = str(message or "").strip()
        if not clean:
            raise ValueError("message must not be empty")
        effective_threshold = self.validate_threshold(
            self.default_threshold if threshold is None else threshold
        )
        started = time.perf_counter()
        policy = policy_tool_gate(clean, search_mode=search_mode)
        if policy is not None:
            return {
                **policy,
                "threshold": effective_threshold,
                "margin": None,
                "elapsed_ms": round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                ),
            }
        result = model.gate_tool(
            clean,
            threshold=effective_threshold,
            context=context,
            has_pasted_text=has_pasted_text,
        )
        result["source"] = "g1i"
        result["reason"] = "ambiguous request resolved by one-token G1I gate"
        return result
