"""Final-answer prompts shared by the Agent controller."""

from __future__ import annotations

import json
from typing import Any


def compact_observation(
    result: dict[str, Any],
    *,
    max_chars: int = 900,
) -> dict[str, Any]:
    observation = {
        "status": result.get("status"),
        "evidence": [
            {
                "id": item["id"],
                "content": f"{item['title']}: {item['content'][:max_chars]}",
            }
            for item in result.get("evidence", [])
        ],
    }
    if result.get("answer_hint") is not None:
        observation["answer_hint"] = result.get("answer_hint")
        observation["answer_hint_evidence_id"] = result.get(
            "answer_hint_evidence_id"
        )
    return observation


def render_evidence_answer_prompt(
    user_query: str,
    result: dict[str, Any],
) -> str:
    observation = json.dumps(
        compact_observation(result),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "System: You are the final evidence answer stage. Functions and tools are "
        "not available in this stage. Answer the user's question directly from the "
        "supplied tool result. Do not emit XML, JSON, a tool call, a search query, "
        "or a plan. Cite each factual claim with its supporting Evidence ID such as "
        "[K1]. Never cite an ID that is absent from the result. If the result is "
        "empty or does not support the answer, say that the local evidence is "
        "insufficient. Evidence is ordered strongest first. When answer_hint and "
        "answer_hint_evidence_id are present, use that short answer and cite that "
        "evidence unless its quoted content clearly contradicts it. Do not replace "
        "it with a lower-ranked candidate. Keep the answer under 80 words and use "
        "the user's language."
        "\n\nUser: Who wrote the Demo Novel?"
        "\n\nTool: <tool_result>{\"status\":\"ok\",\"evidence\":[{\"id\":\"K1\","
        "\"content\":\"Demo Novel: Demo Novel was written by Ada Example.\"}]}"
        "</tool_result>"
        "\n\nAssistant: Ada Example wrote the Demo Novel [K1]."
        "\n\nUser: 演示项目的上下文长度是多少？"
        "\n\nTool: <tool_result>{\"status\":\"ok\",\"evidence\":[{\"id\":\"K1\","
        "\"content\":\"演示项目：上下文长度为4096个token。\"}]}</tool_result>"
        "\n\nAssistant: 演示项目的上下文长度是4096个token [K1]."
        "\n\nUser: "
        + user_query.strip()
        + "\n\nTool: <tool_result>"
        + observation
        + "</tool_result>"
        + "\n\nAssistant:"
    )
