"""Protocol constants and the compact answer prompt for the data plane."""

from __future__ import annotations

import json
from typing import Any


TOOL_CALL_PREFIX = "<tool_call>"
ANSWER_PREFIX = "<answer>"
ANSWER_SUFFIX = "</answer>"
ANSWER_STOPS = (
    ANSWER_SUFFIX,
    "<tool_call>",
    "<tool_result>",
    "\n\nTool:",
    "\n\nUser:",
    "\nSystem:",
)
ANSWER_MAX_TOKENS = 192


def render_compact_answer_prompt(
    question: str,
    evidence: list[dict[str, Any]],
) -> str:
    """Render the production answer protocol without few-shot context overhead.

    The model writes plain prose without citation markers; the controller
    attaches supporting evidence ids deterministically after generation.
    """

    observation = json.dumps(
        {"status": "ok" if evidence else "empty", "evidence": evidence},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "System: Answer the current question directly using only the supplied "
        "Evidence. Do not add citation markers; the system attaches sources "
        "automatically. Preserve source relation labels; never upgrade author, "
        "owner, or maintainer into founder or creator. Do not call a tool, "
        "expose reasoning, or output JSON or role labels. Use one "
        "independently verifiable fact per sentence. Every sentence must "
        "explicitly name the requested subject and relation; do not use "
        "pronouns, Markdown headings, bare field labels, links, or source "
        "inventories. Omit unrequested background. The opening <answer> tag "
        "is already supplied; output only a concise user-visible answer "
        "followed by </answer>.\n\nUser: "
        + question.strip()
        + "\n\nTool: <tool_result>"
        + observation
        + "</tool_result>\n\nAssistant: "
        + ANSWER_PREFIX
    )
