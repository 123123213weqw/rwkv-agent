from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple


P4_SYSTEM_PROMPT = (
    'Call web_search once. Output only <tool_call>{"name":"web_search","arguments":'
    '{"query":QUERY_STRING}}</tool_call>, replacing QUERY_STRING with a valid JSON string. '
    "Preserve entities, versions, time and source constraints."
)

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.I)
_ANY_TOOL_TAG_RE = re.compile(r"</?(?:tool_call|tool_calls|tool_code|tool_output)>", re.I)
_LATIN_ENTITY_RE = re.compile(
    r"(?<![\w])(?:[A-Za-z][A-Za-z0-9+#._-]{1,}|\d+(?:\.\d+)+(?:[A-Za-z]\w*)?)(?![\w])"
)
_ENTITY_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "check", "current", "find",
    "for", "from", "in", "is", "it", "latest", "most", "new", "newest", "of", "on",
    "or", "recent", "release", "search", "the", "this", "to", "today", "version", "what",
    "which", "with",
}


def web_search_schema() -> Dict[str, Any]:
    return {
        "name": "web_search",
        "description": "Search the public web using one concise query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    }


def render_p4_prompt(user_query: str) -> str:
    return (
        "System: "
        + P4_SYSTEM_PROMPT
        + "\n\nSystem: "
        + json.dumps(web_search_schema(), ensure_ascii=False, indent=2)
        + "\n</functions>\n\nUser: "
        + user_query.strip()
        + "\n\nAssistant:"
    )


def important_entities(text: str) -> Tuple[str, ...]:
    output = []
    seen = set()
    for match in _LATIN_ENTITY_RE.finditer(text):
        value = match.group(0)
        folded = value.casefold()
        if folded in _ENTITY_STOP or folded in seen:
            continue
        seen.add(folded)
        output.append(value)
    return tuple(output)


def reconstruct_stopped_output(text: str, stop: str | None) -> str:
    return text + stop if stop and stop.startswith("</tool") else text


def evaluate_web_search_tool_call(raw_output: str) -> Dict[str, Any]:
    """Strictly parse the selected singular, flat-JSON web_search protocol."""

    raw = str(raw_output or "").strip()
    opener, closer = "<tool_call>", "</tool_call>"
    target_tags = raw.count(opener) == 1 and raw.count(closer) == 1
    query = ""
    error = ""
    parse_success = False
    block = ""
    if target_tags:
        start = raw.find(opener)
        end = raw.find(closer, start + len(opener))
        block = raw[start : end + len(closer)]
        body = raw[start + len(opener) : end].strip()
        try:
            value = json.loads(body)
            if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
                raise ValueError("tool call must contain exactly name and arguments")
            if value.get("name") != "web_search":
                raise ValueError("tool name is not web_search")
            arguments = value.get("arguments")
            if not isinstance(arguments, dict) or set(arguments) != {"query"}:
                raise ValueError("arguments must contain exactly query")
            candidate = arguments.get("query")
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError("query is empty or not a string")
            query = candidate.strip()
            parse_success = True
        except (ValueError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    else:
        error = "target opener/closer did not occur exactly once"

    without_think = _THINK_RE.sub("", raw).strip()
    exact_block_only = bool(block and without_think == block)
    tags = _ANY_TOOL_TAG_RE.findall(raw)
    no_other_tool_format = sorted(tag.casefold() for tag in tags) == sorted(
        [opener.casefold(), closer.casefold()]
    )
    return {
        "target_tags": target_tags,
        "parse_success": parse_success,
        "exact_block_only": exact_block_only,
        "no_other_tool_format": no_other_tool_format,
        "strict_success": parse_success and exact_block_only and no_other_tool_format,
        "query": query,
        "error": error,
    }
