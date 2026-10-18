"""Conversation context, direct-chat and evidence-answer prompt rendering."""

from __future__ import annotations

import re
from typing import Any

from .prompts import render_evidence_answer_prompt as render_base_answer_prompt


LEADING_THINK_BLOCKS = re.compile(
    r"\A\s*(?:<think>.*?</think>\s*)+",
    re.S,
)
CHAT_STOPS = (
    "\n\nUser:",
    "\nUser:",
    "\nSystem:",
    "</s>",
)
CHAT_USER_STOPS = frozenset({"\n\nUser:", "\nUser:"})
CHAT_UNSAFE_STOPS = frozenset({"\nSystem:"})
DIRECT_SYSTEM_PROMPT = (
    "System: You are a helpful conversational assistant. Answer the user "
    "directly in the user's language. Do not claim to have searched, do not "
    "invent sources or citation IDs, and do not emit a tool call. Use the "
    "supplied conversation when relevant. The conversation is the only memory "
    "available; there is no extracted long-term profile. Do not mention memory "
    "machinery unless the user asks about it. Never output <think> tags or "
    "hidden reasoning.\n\n"
)


def strip_leading_think_blocks(raw: str) -> tuple[str, bool]:
    """Hide complete leading reasoning blocks without relaxing protocols."""

    value = str(raw or "")
    match = LEADING_THINK_BLOCKS.match(value)
    if match is None:
        return value.strip(), False
    return value[match.end() :].strip(), True


def _bounded_context(
    entries: list[tuple[str, str]],
    *,
    max_chars: int,
) -> str:
    selected: list[str] = []
    used = 0
    for label, value in reversed(entries):
        clean = str(value or "").strip()
        if not clean:
            continue
        line = f"{label}: {clean}"
        separator = 1 if selected else 0
        remaining = max_chars - used - separator
        if remaining <= 0:
            break
        if len(line) > remaining:
            line = line[-remaining:]
        selected.append(line)
        used += separator + len(line)
    return "\n".join(reversed(selected))


def render_session_context(history: list[Any]) -> str:
    return _bounded_context(
        [
            (
                "User" if item.role == "user" else "Assistant",
                (
                    item.content
                    if item.role == "user"
                    else strip_leading_think_blocks(item.content)[0]
                ),
            )
            for item in history
        ],
        max_chars=8000,
    )


def render_routing_context(history: list[Any]) -> str:
    """Keep routing context small while preserving recent follow-up referents."""

    return _bounded_context(
        [
            (
                "User" if item.role == "user" else "Assistant",
                (
                    item.content
                    if item.role == "user"
                    else strip_leading_think_blocks(item.content)[0]
                ),
            )
            for item in history
        ],
        max_chars=2000,
    )


def render_direct_chat_prefix(*, context: str = "") -> str:
    prompt = DIRECT_SYSTEM_PROMPT
    if context:
        prompt += context + "\n\n"
    return prompt


def render_direct_chat_turn(
    message: str,
    *,
    continuation: bool,
    previous_stop: str = "",
) -> str:
    clean = str(message or "").strip()
    if not clean:
        raise ValueError("message must not be empty")
    if continuation and previous_stop in CHAT_USER_STOPS:
        return f" {clean}\n\nAssistant:"
    prefix = "\n\n" if continuation else ""
    return prefix + f"User: {clean}\n\nAssistant:"


def render_direct_answer_prompt(message: str, *, context: str = "") -> str:
    return render_direct_chat_prefix(context=context) + render_direct_chat_turn(
        message,
        continuation=False,
    )


def render_evidence_answer_prompt(
    message: str,
    result: dict[str, Any],
    *,
    context: str = "",
) -> str:
    prompt = render_base_answer_prompt(message, result)
    if not context:
        return prompt
    current_turn = "\n\nUser: " + message.strip()
    prefix, separator, suffix = prompt.rpartition(current_turn)
    if not separator:
        return prompt
    return prefix + "\n\n" + context + separator + suffix
