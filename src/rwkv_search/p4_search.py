from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

from .g1i_tool_call import (
    P4_SYSTEM_PROMPT as P4_SYSTEM_PROMPT,
    evaluate_web_search_tool_call,
    reconstruct_stopped_output,
    render_p4_prompt,
)
from .g1i_types import G1ICompletion
from .search_request import SearchRequest, SearchRequestBuilder


@dataclass(frozen=True)
class P4Plan:
    raw_output: str
    stop: str
    token_ids: tuple[int, ...]
    elapsed_ms: float
    format_evaluation: Dict[str, Any]
    search_request: Optional[SearchRequest]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_output": self.raw_output,
            "stop": self.stop,
            "token_ids": list(self.token_ids),
            "token_count": len(self.token_ids),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "format_evaluation": self.format_evaluation,
            "search_request": self.search_request.to_dict() if self.search_request else None,
        }


class P4SearchPlanner:
    def __init__(
        self,
        complete: Callable[[str, Sequence[str], int], G1ICompletion],
        *,
        builder: Optional[SearchRequestBuilder] = None,
        max_tokens: int = 192,
    ) -> None:
        self.complete = complete
        self.builder = builder or SearchRequestBuilder()
        self.max_tokens = max_tokens

    def plan(self, user_query: str) -> P4Plan:
        completion = self.complete(
            render_p4_prompt(user_query),
            ("</tool_call>", "</tool_calls>", "</tool_code>", "\n\nUser:", "</s>"),
            self.max_tokens,
        )
        raw = reconstruct_stopped_output(completion.text, completion.stop)
        evaluation = evaluate_web_search_tool_call(raw)
        request = None
        if evaluation.get("strict_success") and evaluation.get("query"):
            request = self.builder.build(user_query, str(evaluation["query"]))
        return P4Plan(
            raw_output=raw,
            stop=completion.stop,
            token_ids=completion.token_ids,
            elapsed_ms=float(completion.elapsed_ms or 0.0),
            format_evaluation=evaluation,
            search_request=request,
        )
