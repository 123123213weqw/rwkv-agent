from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

from rwkv_search.pipeline.search_need import SearchNeedGate

from .memory import MemoryStore
from .prompts import render_evidence_answer_prompt as render_base_answer_prompt
from .session_text import SessionTextBuffer
from .state_agent import StateNativeSearchAgent
from .tools import KnowledgeSearchAdapter, LongTextQAAdapter, WebSearchAdapter


TOOLS = ("web_search", "knowledge_search", "long_text_qa")
LONG_TEXT_CAPTURE_CHARS = 4000
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


def build_semantic_scorer_from_env() -> Any | None:
    """Build one lazy local reranker shared by routing and evidence selection."""

    model_name = os.getenv("RWKV_AGENT_RERANKER_MODEL", "").strip()
    if not model_name:
        return None
    from rwkv_search.semantic_selection import TransformersPairScorer

    return TransformersPairScorer(
        model_name,
        device=os.getenv("RWKV_AGENT_RERANKER_DEVICE", "auto").strip() or "auto",
        batch_size=int(os.getenv("RWKV_AGENT_RERANKER_BATCH_SIZE", "16")),
        max_length=int(os.getenv("RWKV_AGENT_RERANKER_MAX_LENGTH", "512")),
        fp16=os.getenv("RWKV_AGENT_RERANKER_FP16", "1").strip().casefold()
        not in {"0", "false", "no"},
        local_files_only=True,
    )


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


def render_session_context(
    history: list[Any],
) -> str:
    return _bounded_context(
        [
            ("User" if item.role == "user" else "Assistant", item.content)
            for item in history
        ],
        max_chars=8000,
    )


def render_routing_context(history: list[Any]) -> str:
    """Keep routing context small while preserving recent follow-up referents."""

    return _bounded_context(
        [
            ("User" if item.role == "user" else "Assistant", item.content)
            for item in history
        ],
        max_chars=2000,
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
        "Do not copy source text into arguments. Do not answer."
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
    candidate = str(raw or "").strip()
    match = TOOL_ENVELOPE.fullmatch(candidate)
    if not match:
        return {
            "strict": False,
            "tool": "",
            "arguments": {},
            "error": "envelope",
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
        }
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        return {
            "strict": False,
            "tool": "",
            "arguments": {},
            "error": str(exc),
        }


def policy_tool_gate(
    message: str,
    *,
    search_mode: str = "auto",
) -> dict[str, Any] | None:
    """Apply only the explicit UI search switch; semantics belong to G1I."""

    decision = SearchNeedGate().policy(message, mode=search_mode)
    return decision.to_dict() if decision is not None else None


def render_direct_answer_prompt(message: str, *, context: str = "") -> str:
    prompt = (
        "System: You are a helpful conversational assistant. Answer the user "
        "directly in the user's language. Do not claim to have searched, do not "
        "invent sources or citation IDs, and do not emit a tool call. Use the "
        "supplied conversation when relevant. The conversation is the only memory "
        "available; there is no extracted long-term profile. Do not mention memory "
        "machinery unless the user asks about it.\n\n"
    )
    if context:
        prompt += context + "\n\n"
    return prompt + f"User: {message.strip()}\n\nAssistant:"


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


class ModelClient:
    def __init__(self, urls: list[str]) -> None:
        if not urls:
            raise ValueError("at least one G1I sidecar URL is required")
        self.urls = [value.rstrip("/") for value in urls]
        self._index = 0
        self._lock = threading.Lock()

    def _next_url(self) -> str:
        with self._lock:
            url = self.urls[self._index % len(self.urls)]
            self._index += 1
            return url

    @staticmethod
    def _get(url: str) -> dict[str, Any]:
        with urlopen(url, timeout=10) as response:
            return json.load(response)

    @staticmethod
    def _post(
        url: str,
        payload: dict[str, Any],
        *,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode()
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=timeout) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise RuntimeError("sidecar returned a non-object response")
        return value

    def health(self) -> list[dict[str, Any]]:
        return [self._get(url + "/health") for url in self.urls]

    def state_prefill(
        self,
        *,
        owner_id: str,
        prompt: str,
    ) -> dict[str, Any]:
        home_url = self._next_url()
        result = self._post(
            home_url + "/v1/states/prefill",
            {
                "owner_id": owner_id,
                "prompt": prompt,
                "branch": "root",
            },
        )
        state = dict(result["state"])
        state["home_url"] = home_url
        return state

    def state_fork(
        self,
        *,
        home_url: str,
        owner_id: str,
        parent_state_id: str,
        branches: list[str],
    ) -> list[dict[str, Any]]:
        result = self._post(
            home_url.rstrip("/") + f"/v1/states/{parent_state_id}/fork",
            {"owner_id": owner_id, "branches": branches},
        )
        return [dict(value) for value in result["states"]]

    def state_batch_continue(
        self,
        *,
        home_url: str,
        owner_id: str,
        items: list[dict[str, str]],
        stops: list[str],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        result = self._post(
            home_url.rstrip("/") + "/v1/states/batch_continue",
            {
                "owner_id": owner_id,
                "items": items,
                "stop": stops,
                "max_tokens": max_tokens,
            },
        )
        return [dict(value) for value in result["results"]]

    def state_batch_classify(
        self,
        *,
        home_url: str,
        owner_id: str,
        items: list[dict[str, str]],
        labels: dict[str, str],
    ) -> list[dict[str, Any]]:
        result = self._post(
            home_url.rstrip("/") + "/v1/states/batch_classify",
            {"owner_id": owner_id, "items": items, "labels": labels},
        )
        return [dict(value) for value in result["results"]]

    def state_release(
        self,
        *,
        home_url: str,
        owner_id: str,
        state_ids: list[str],
    ) -> dict[str, Any]:
        return self._post(
            home_url.rstrip("/") + "/v1/states/release",
            {"owner_id": owner_id, "state_ids": state_ids},
        )

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 192,
        stops: list[str] | None = None,
    ) -> dict[str, Any]:
        url = self._next_url()
        stop_values = stops or [
            "</tool_call>",
            "</tool_calls>",
            "</tool_code>",
            "\nUser:",
            "\nSystem:",
            "\n\nUser:",
            "</s>",
        ]
        body = json.dumps(
            {
                "prompt": prompt,
                "stop": stop_values,
                "max_tokens": max_tokens,
            },
            ensure_ascii=False,
        ).encode()
        started = time.perf_counter()
        request = Request(
            url + "/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=180) as response:
            data = json.load(response)
        g1i = data["g1i"]
        stop = str(g1i.get("stop_reason") or "")
        raw = str(g1i.get("text") or "") + (stop if stop.startswith("</tool") else "")
        return {
            "raw": raw,
            "stop": stop,
            "output_tokens": len(g1i.get("token_ids") or []),
            "model_elapsed_ms": float(g1i.get("elapsed_ms") or 0.0),
            "request_elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "model": data.get("model"),
            "url": url,
        }

    def classify(
        self,
        prompt: str,
        *,
        labels: dict[str, str],
    ) -> dict[str, Any]:
        url = self._next_url()
        started = time.perf_counter()
        result = self._post(
            url + "/v1/classify",
            {"prompt": prompt, "labels": labels},
        )
        result["request_elapsed_ms"] = round(
            (time.perf_counter() - started) * 1000.0,
            3,
        )
        result["url"] = url
        return result

    def gate_tool(
        self,
        message: str,
        *,
        threshold: float = 0.7,
        context: str = "",
        has_pasted_text: bool = False,
    ) -> dict[str, Any]:
        url = self._next_url()
        body = json.dumps(
            {
                "message": message,
                "threshold": threshold,
                "context": context,
                "has_pasted_text": has_pasted_text,
            },
            ensure_ascii=False,
        ).encode()
        started = time.perf_counter()
        request = Request(
            url + "/v1/gate/tool",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        result["request_elapsed_ms"] = round(
            (time.perf_counter() - started) * 1000.0,
            3,
        )
        result["url"] = url
        return result


class AgentController:
    def __init__(
        self,
        *,
        model_urls: list[str],
        knowledge_endpoint: str = "http://127.0.0.1:19220",
        memory_path: str = "var/sessions.sqlite3",
        web_config: str = "configs/default.json",
        web_adapter: Any | None = None,
        knowledge_adapter: Any | None = None,
        long_text_adapter: Any | None = None,
        long_text_capture_chars: int = LONG_TEXT_CAPTURE_CHARS,
        session_text_buffer: SessionTextBuffer | None = None,
        tool_gate_threshold: float = 0.7,
        semantic_scorer: Any | None = None,
    ) -> None:
        self.model = ModelClient(model_urls)
        self.memory = MemoryStore(memory_path)
        self.semantic_scorer = semantic_scorer or build_semantic_scorer_from_env()
        self.web = web_adapter or WebSearchAdapter(
            web_config,
            semantic_scorer=self.semantic_scorer,
        )
        self.knowledge = knowledge_adapter or KnowledgeSearchAdapter(knowledge_endpoint)
        self.long_text = long_text_adapter or LongTextQAAdapter(self.model.complete)
        self.long_text_capture_chars = max(256, int(long_text_capture_chars))
        self.tool_gate_threshold = float(tool_gate_threshold)
        self.search_need_gate = SearchNeedGate()
        if not -20.0 <= self.tool_gate_threshold <= 20.0:
            raise ValueError("tool_gate_threshold out of range")
        max_text_chars = int(getattr(self.long_text, "max_document_chars", 1_000_000))
        self.session_text = session_text_buffer or SessionTextBuffer(
            max_chars=max_text_chars
        )
        self.state_search = StateNativeSearchAgent(
            state_model=self.model,
            parse_tool_call=parse_tool_call,
            execute_tool=self.execute_tool,
            evidence_scorer=self.semantic_scorer,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "tools": list(TOOLS),
            "model": self.model.health(),
            "context": {
                "mode": "session_transcript",
                "history_messages": 12,
                "long_term_memory": False,
            },
            "tool_gate": {
                "mode": "semantic_single_token",
                "threshold": self.tool_gate_threshold,
            },
            "pasted_text": {
                **self.session_text.health(),
                "capture_chars": self.long_text_capture_chars,
            },
            "state_parallel_search": {
                "enabled": True,
                "mode": "opt_in_experiment",
                "endpoint": "/v1/agent/run_stateful",
                "max_branches": 4,
                "max_rounds": 3,
                "semantic_selector": {
                    "strategy": "query_view_mmr_v1",
                    "model": str(
                        getattr(self.semantic_scorer, "model_name", "")
                    ),
                    "enabled": self.semantic_scorer is not None,
                },
                "default_branches": 4,
                "default_rounds": 2,
            },
        }

    def decide_tool(
        self,
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
        effective_threshold = (
            self.tool_gate_threshold if threshold is None else float(threshold)
        )
        if not -20.0 <= effective_threshold <= 20.0:
            raise ValueError("threshold out of range")
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
        result = self.model.gate_tool(
            clean,
            threshold=effective_threshold,
            context=context,
            has_pasted_text=has_pasted_text,
        )
        result["source"] = "g1i"
        result["reason"] = "ambiguous request resolved by one-token G1I gate"
        return result

    def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        session_id: str = "default",
    ) -> dict[str, Any]:
        if name == "web_search":
            return self.web.execute(str(arguments.get("query") or ""))
        if name == "knowledge_search":
            query = str(arguments.get("query") or "").strip()
            if not query:
                return {
                    "status": "invalid",
                    "evidence": [],
                    "message": "knowledge_search requires a non-empty query.",
                }
            return self.knowledge.execute(query)
        if name == "long_text_qa":
            if set(arguments) != {"question"}:
                return {
                    "status": "invalid",
                    "evidence": [],
                    "message": ("long_text_qa accepts exactly one argument: question."),
                }
            pasted = self.session_text.get(session_id)
            if pasted is None:
                return {
                    "status": "empty",
                    "evidence": [],
                    "message": (
                        "No pasted long text is active in this session. "
                        "Paste the text first, then ask a question."
                    ),
                }
            return self.long_text.execute(
                pasted.text,
                str(arguments.get("question") or ""),
                document_name=pasted.name,
            )
        if name in {"memory", "memory_save"}:
            return {
                "status": "disabled",
                "reason": "context_only_mode",
                "message": (
                    "Long-term memory is disabled. Only the current session "
                    "transcript is used as context."
                ),
            }
        return {
            "status": "invalid",
            "message": f"unknown tool: {name}",
        }

    def run(
        self,
        message: str,
        *,
        session_id: str = "default",
        search_mode: str = "auto",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        case = {"query": str(message or "").strip()}
        if not case["query"]:
            return {"status": "invalid", "message": "message must not be empty"}
        if len(case["query"]) >= self.long_text_capture_chars:
            pasted = self.session_text.put(session_id, case["query"])
            is_chinese = any("\u3400" <= char <= "\u9fff" for char in case["query"])
            answer = (
                f"已接收长文本，共{pasted.chars}个字符。请继续提问。"
                if is_chinese
                else (
                    f"Received {pasted.chars} characters of long text. "
                    "Ask a question when ready."
                )
            )
            placeholder = (
                f"[pasted long text: {pasted.name}, {pasted.chars} characters]"
            )
            self.memory.append_exchange(
                session_id=session_id,
                user=placeholder,
                assistant=answer,
            )
            return {
                "status": "ok",
                "session_id": session_id,
                "route": {
                    "mode": "document_capture",
                    "tool": None,
                },
                "tool_result": {
                    "status": "accepted",
                    "document": {
                        "source": "session_pasted_text",
                        "name": pasted.name,
                        "chars": pasted.chars,
                        "sha256": pasted.sha256,
                    },
                },
                "answer": answer,
                "trace": {
                    "model_called": False,
                    "context": {
                        "mode": "transient_session_text",
                    },
                    "elapsed_ms": round(
                        (time.perf_counter() - started) * 1000.0,
                        3,
                    ),
                },
            }
        history = self.memory.history(session_id=session_id, limit=12)
        context = render_session_context(history)
        routing_context = render_routing_context(history)
        pasted = self.session_text.get(session_id)
        gate = self.decide_tool(
            case["query"],
            context=routing_context,
            has_pasted_text=pasted is not None,
            search_mode=search_mode,
        )
        if not bool(gate.get("use_tool")):
            answer_completion = self.model.complete(
                render_direct_answer_prompt(case["query"], context=context),
                max_tokens=256,
            )
            answer = answer_completion["raw"].strip()
            self.memory.append_exchange(
                session_id=session_id,
                user=case["query"],
                assistant=answer,
            )
            return {
                "status": "ok",
                "session_id": session_id,
                "message": message,
                "route": {
                    "mode": "direct",
                    "tool": None,
                },
                "tool_result": None,
                "answer": answer,
                "trace": {
                    "gate": gate,
                    "context": {
                        "history_messages": len(history),
                        "mode": "session_transcript",
                    },
                    "routing_completion": None,
                    "answer_completion": answer_completion,
                    "memory_consolidation": {
                        "enabled": False,
                        "reason": "context_only_mode",
                    },
                    "elapsed_ms": round(
                        (time.perf_counter() - started) * 1000.0,
                        3,
                    ),
                },
            }
        routing = self.model.complete(
            render_tool_prompt(
                case["query"],
                has_pasted_text=pasted is not None,
                context=routing_context,
            )
        )
        parsed = parse_tool_call(routing["raw"])
        if not parsed["strict"]:
            return {
                "status": "route_error",
                "route": parsed,
                "routing_completion": routing,
            }
        tool_result = self.execute_tool(
            parsed["tool"],
            parsed["arguments"],
            session_id=session_id,
        )
        answer_completion: dict[str, Any] | None = None
        if tool_result.get("status") == "empty":
            answer = (
                "没有找到可用证据。"
                if any("\u3400" <= char <= "\u9fff" for char in message)
                else "No usable evidence was found."
            )
        else:
            answer_completion = self.model.complete(
                render_evidence_answer_prompt(
                    message,
                    tool_result,
                    context=context,
                ),
                max_tokens=192,
            )
            answer = answer_completion["raw"].strip()
        self.memory.append_exchange(
            session_id=session_id,
            user=case["query"],
            assistant=answer,
        )
        return {
            "status": "ok",
            "session_id": session_id,
            "message": message,
            "route": {
                "mode": "tool",
                "tool": parsed["tool"],
                "arguments": parsed["arguments"],
                "strict": True,
            },
            "tool_result": tool_result,
            "answer": answer,
            "trace": {
                "gate": gate,
                "context": {
                    "history_messages": len(history),
                    "mode": "session_transcript",
                },
                "routing_completion": routing,
                "answer_completion": answer_completion,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            },
        }

    def run_stateful_search(
        self,
        message: str,
        *,
        session_id: str,
        branch_width: int = 4,
        max_rounds: int = 2,
    ) -> dict[str, Any]:
        """Run the opt-in bounded State-native Web search experiment."""

        clean_session = str(session_id or "").strip()
        clean_message = str(message or "").strip()
        if not clean_session:
            raise ValueError("session_id must not be empty")
        if not clean_message:
            raise ValueError("message must not be empty")
        result = self.state_search.run(
            clean_message,
            session_id=clean_session,
            branch_width=branch_width,
            max_rounds=max_rounds,
        )
        self.memory.append_exchange(
            session_id=clean_session,
            user=clean_message,
            assistant=str(result.get("answer") or ""),
        )
        return result

    def close(self) -> None:
        self.session_text.close()
        knowledge_close = getattr(self.knowledge, "close", None)
        if callable(knowledge_close):
            knowledge_close()
        self.web.close()
