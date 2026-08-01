"""Shared greedy-token and decoded-stop semantics.

Model execution stays in the scheduler.  These helpers make EOS, token budget,
replacement-character handling and earliest-stop selection identical across
short-lived completions, persistent Agent states and direct scheduler decode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GreedyTokenStatus:
    token: int
    eos: bool
    budget_reached: bool

    @property
    def finished(self) -> bool:
        return self.eos or self.budget_reached


@dataclass(frozen=True, slots=True)
class DecodedTextStatus:
    text: str
    stop_reason: str


def append_greedy_token(
    output_ids: list[int],
    token: int,
    *,
    eos_token_id: int,
    max_tokens: int,
) -> GreedyTokenStatus:
    """Append one non-EOS token and report the exact token-budget boundary."""

    value = int(token)
    if value == int(eos_token_id):
        return GreedyTokenStatus(token=value, eos=True, budget_reached=False)
    output_ids.append(value)
    return GreedyTokenStatus(
        token=value,
        eos=False,
        budget_reached=len(output_ids) >= int(max_tokens),
    )


def decode_text_stops(
    tokenizer: Any,
    output_ids: list[int],
    *,
    previous_text: str,
    stops: Sequence[str],
) -> DecodedTextStatus:
    """Decode accumulated IDs and select the earliest complete stop string."""

    decoded = str(tokenizer.decode(output_ids))
    if "\ufffd" in decoded:
        return DecodedTextStatus(text=previous_text, stop_reason="")
    hits = [
        (decoded.find(stop), stop)
        for stop in stops
        if stop and stop in decoded
    ]
    if not hits:
        return DecodedTextStatus(text=decoded, stop_reason="")
    index, reason = min(hits)
    return DecodedTextStatus(text=decoded[:index], stop_reason=reason)
