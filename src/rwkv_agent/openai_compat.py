"""Small, explicit OpenAI wire helpers for the RWKV-native serving path.

The module deliberately contains no model-runtime imports.  It translates the
stable OpenAI chat wire format into RWKV's native ``System/User/Assistant``
prompt convention and translates deterministic scheduler results back into
OpenAI response objects.  Sampling and tool execution remain runtime
capabilities rather than fields that are silently accepted here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any


CHAT_STOP_STRINGS = ("\n\nUser:", "\nUser:", "\nSystem:", "</s>")
_ROLE_LABELS = {
    "system": "System",
    "developer": "System",
    "user": "User",
    "assistant": "Assistant",
}


def render_chat_prompt(messages: Any) -> str:
    """Render a bounded text-only OpenAI message list for an RWKV chat model."""

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    rendered: list[str] = []
    last_role = ""
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role")
        if role not in _ROLE_LABELS:
            raise ValueError(
                f"messages[{index}].role must be system, developer, user or assistant"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"messages[{index}].content must be a non-empty string"
            )
        clean = content.strip()
        rendered.append(f"{_ROLE_LABELS[role]}: {clean}")
        last_role = str(role)
    if last_role != "user":
        raise ValueError("the final chat message must have role user")
    return "\n\n".join(rendered) + "\n\nAssistant:"


def normalize_stops(value: Any, *, chat: bool = False) -> list[str]:
    """Validate caller stops and append RWKV role-boundary stops for chat."""

    if value is None:
        supplied: list[str] = []
    elif isinstance(value, str):
        supplied = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        supplied = list(value)
    else:
        raise ValueError("stop must be a string or string array")
    if len(supplied) > 8 or any(not item or len(item) > 256 for item in supplied):
        raise ValueError("stop must contain at most 8 non-empty strings of 256 chars")
    values = supplied + (list(CHAT_STOP_STRINGS) if chat else [])
    return list(dict.fromkeys(values))


def openai_finish_reason(reason: Any) -> str:
    """Map the scheduler's exact stop marker onto the OpenAI enum."""

    return "length" if str(reason) == "max_tokens" else "stop"


def completion_usage(result: Mapping[str, Any]) -> dict[str, int]:
    prompt_tokens = max(0, int(result.get("input_tokens", 0)))
    token_ids = result.get("token_ids", [])
    completion_tokens = len(token_ids) if isinstance(token_ids, Sequence) else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def sse_data(payload: Mapping[str, Any] | str) -> str:
    """Encode one OpenAI-compatible server-sent event."""

    if isinstance(payload, str):
        body = payload
    else:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {body}\n\n"
