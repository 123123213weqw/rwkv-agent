"""Strict bounded Tool Call protocol and semantic routing helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from rwkv_search.pipeline.search_need import SearchNeedGate

from .chat_prompts import strip_leading_think_blocks


TOOLS = ("web_search", "knowledge_search", "long_text_qa")
TOOL_SCHEMAS = {
    "web_search": {
        "name": "web_search",
        "description": "Search live or current public Internet information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "knowledge_search": {
        "name": "knowledge_search",
        "description": "Search local files and the local knowledge index.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "long_text_qa": {
        "name": "long_text_qa",
        "description": (
            "Answer a question about the long text currently pasted into this "
            "chat session by parallel chunk analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question to answer from the active pasted text.",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}
TOOL_ENVELOPE = re.compile(
    r"<tool_call>\s*(\{.*\})\s*</tool_call>",
    re.S,
)


def render_tool_prompt(
    message: str,
    *,
    has_pasted_text: bool = False,
    context: str = "",
) -> str:
    prompt = (
        "System: Call exactly one function and output only "
        '<tool_call>{"name":...,"arguments":...}</tool_call>.\n'
        "Functions:\n"
        "- web_search(query): live public Internet\n"
        "- knowledge_search(query): local indexed knowledge, no file path\n"
        "- long_text_qa(question): active pasted session text only\n"
        f"Active pasted long text: {'yes' if has_pasted_text else 'no'}.\n"
        "Do not copy source text into arguments. Do not answer. "
        "Do not emit <think> or any text before the tool call."
    )
    prompt += (
        "\nUser: What is the current stable Python version?\n\nAssistant:"
        '<tool_call>{"name":"web_search","arguments":{"query":"Python current '
        'stable version official"}}</tool_call>\n\n'
        "User: Search the local knowledge base for the RWKV Agent design."
        "\n\nAssistant:"
        '<tool_call>{"name":"knowledge_search","arguments":{"query":"RWKV Agent '
        'design"}}</tool_call>\n'
        "\nSystem: Active pasted long text: yes.\n"
        "User: Who founded the Red Coast base?\n\nAssistant:"
        '<tool_call>{"name":"long_text_qa","arguments":{"question":'
        '"Who founded the Red Coast base?"}}</tool_call>\n'
    )
    if context.strip():
        prompt += (
            "\nSystem: Use this recent conversation only to resolve pronouns "
            "or omitted entities in the next User request:\n"
            + context.strip()
            + "\nEnd recent conversation.\n"
        )
    return prompt + "\nUser: " + message.strip() + "\n\nAssistant:"


def parse_tool_call(raw: str) -> dict[str, Any]:
    candidate, reasoning_stripped = strip_leading_think_blocks(raw)
    match = TOOL_ENVELOPE.fullmatch(candidate)
    if not match:
        return {
            "strict": False,
            "tool": "",
            "arguments": {},
            "error": "envelope",
            "reasoning_stripped": reasoning_stripped,
        }
    try:
        value = json.loads(match.group(1))
        if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
            raise ValueError("payload keys")
        name = value["name"]
        if name not in TOOLS:
            raise ValueError("tool name")
        arguments = value["arguments"]
        if not isinstance(arguments, dict):
            raise ValueError("arguments")
        expected_keys = {"question"} if name == "long_text_qa" else {"query"}
        if set(arguments) != expected_keys:
            raise ValueError("argument keys")
        if name == "long_text_qa":
            question = arguments["question"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError("question")
            normalized = {"question": question.strip()}
        else:
            query = arguments["query"]
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query")
            normalized = {"query": query.strip()}
        return {
            "strict": True,
            "tool": name,
            "arguments": normalized,
            "error": "",
            "reasoning_stripped": reasoning_stripped,
        }
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        return {
            "strict": False,
            "tool": "",
            "arguments": {},
            "error": str(exc),
            "reasoning_stripped": reasoning_stripped,
        }


def policy_tool_gate(
    message: str,
    *,
    search_mode: str = "auto",
) -> dict[str, Any] | None:
    """Apply only the explicit UI search switch; semantics belong to G1I."""

    decision = SearchNeedGate().policy(message, mode=search_mode)
    return decision.to_dict() if decision is not None else None
