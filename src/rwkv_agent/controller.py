from __future__ import annotations

import os
import time
from typing import Any

from .chat_prompts import (
    render_direct_answer_prompt as render_direct_answer_prompt,
    render_evidence_answer_prompt,
    render_routing_context,
    render_session_context,
    strip_leading_think_blocks,
)
from .chat_session import DirectChatSession
from .chat_state import ChatStateCache
from .data_plane import AgentDataPlane
from .evidence_admission import EntityEvidenceAdmission
from .memory import MemoryStore
from .model_client import ModelClient
from .session_text import SessionTextBuffer
from .state_agent import StateNativeSearchAgent
from .state_answer import coordinate_answer_output
from .tool_protocol import (
    TOOLS,
    TOOL_SCHEMAS as TOOL_SCHEMAS,
    parse_tool_call,
    policy_tool_gate as policy_tool_gate,
    render_tool_prompt,
)
from .tool_routing import ToolRouter
from .tools import KnowledgeSearchAdapter, LongTextQAAdapter, WebSearchAdapter
from rwkv_search.pipeline.answer_policy import AnswerPolicy


LONG_TEXT_CAPTURE_CHARS = 4000


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
        preserve_query_view_evidence: bool = False,
        chat_state_enabled: bool | None = None,
        chat_state_capacity: int | None = None,
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
        self.tool_gate_threshold = ToolRouter.validate_threshold(tool_gate_threshold)
        self.tool_router = ToolRouter(default_threshold=self.tool_gate_threshold)
        self.answer_policy = AnswerPolicy()
        self.evidence_admission = EntityEvidenceAdmission()
        max_text_chars = int(getattr(self.long_text, "max_document_chars", 1_000_000))
        self.session_text = session_text_buffer or SessionTextBuffer(
            max_chars=max_text_chars
        )
        self.data_plane = AgentDataPlane(
            web=self.web,
            knowledge=self.knowledge,
            long_text=self.long_text,
            session_text=self.session_text,
            semantic_scorer=self.semantic_scorer,
            evidence_admission=self.evidence_admission,
            answer_policy=self.answer_policy,
        )
        self.tool_executor = self.data_plane.tool_executor
        if chat_state_enabled is None:
            chat_state_enabled = (
                os.getenv("RWKV_AGENT_CHAT_STATE_ENABLED", "1")
                .strip()
                .casefold()
                not in {"0", "false", "no"}
            )
        if chat_state_capacity is None:
            chat_state_capacity = int(
                os.getenv("RWKV_AGENT_CHAT_STATE_CAPACITY", "3")
            )
        self.chat_state_enabled = bool(chat_state_enabled)
        self.chat_states = ChatStateCache(capacity=int(chat_state_capacity))
        self.chat_session = DirectChatSession(
            model_provider=lambda: self.model,
            enabled_provider=lambda: self.chat_state_enabled,
            cache=self.chat_states,
        )
        self.state_search = StateNativeSearchAgent(
            state_model=self.model,
            parse_tool_call=parse_tool_call,
            execute_tool=self.execute_tool,
            evidence_scorer=self.semantic_scorer,
            answer_policy=self.answer_policy,
            evidence_admission=self.evidence_admission,
            preserve_query_view_evidence=preserve_query_view_evidence,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "tools": list(TOOLS),
            "model": self.model.health(),
            "context": {
                "mode": (
                    "recurrent_session_state_with_transcript_fallback"
                    if self.chat_state_enabled
                    else "session_transcript"
                ),
                "history_messages": 12,
                "long_term_memory": False,
                "session_state": self.chat_states.health(
                    enabled=self.chat_state_enabled
                ),
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
        return self.tool_router.decide(
            self.model,
            message,
            threshold=self.tool_gate_threshold if threshold is None else threshold,
            context=context,
            has_pasted_text=has_pasted_text,
            search_mode=search_mode,
        )

    def execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        session_id: str = "default",
        original_query: str | None = None,
    ) -> dict[str, Any]:
        return self.data_plane.execute_raw(
            name,
            arguments,
            session_id=session_id,
            original_query=original_query,
        )

    def run(
        self,
        message: str,
        *,
        session_id: str = "default",
        search_mode: str = "auto",
    ) -> dict[str, Any]:
        clean_session = self.chat_states.normalize_session_id(session_id)
        with self.chat_states.turn_lock(clean_session):
            return self._run_locked(
                message,
                session_id=clean_session,
                search_mode=search_mode,
            )

    def _run_locked(
        self,
        message: str,
        *,
        session_id: str,
        search_mode: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        case = {"query": str(message or "").strip()}
        if not case["query"]:
            return {"status": "invalid", "message": "message must not be empty"}
        if len(case["query"]) >= self.long_text_capture_chars:
            self.chat_session.invalidate(session_id)
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
            answer_completion, chat_state, chat_state_trace = (
                self.chat_session.complete(
                    case["query"],
                    session_id=session_id,
                    history=history,
                    context=context,
                )
            )
            answer, reasoning_stripped = strip_leading_think_blocks(
                answer_completion["raw"]
            )
            answer_completion["reasoning_stripped"] = reasoning_stripped
            try:
                _user_record, assistant_record = self.memory.append_exchange(
                    session_id=session_id,
                    user=case["query"],
                    assistant=answer,
                )
            except Exception:
                if chat_state is not None:
                    self.chat_session.discard(chat_state)
                raise
            self.chat_session.store_completed(
                chat_state,
                assistant_message_id=assistant_record.id,
                reasoning_stripped=reasoning_stripped,
                trace=chat_state_trace,
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
                        "mode": (
                            "recurrent_session_state"
                            if chat_state_trace["used"]
                            else "session_transcript"
                        ),
                        "session_state": chat_state_trace,
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
        # A tool turn is generated from a separate evidence prompt. Release the
        # conversational state now instead of occupying a GPU slot while Web or
        # knowledge I/O runs. The next direct turn rebuilds once from the
        # durable transcript and then resumes incremental State continuation.
        self.chat_session.invalidate(session_id)
        routing = self.model.complete(
            render_tool_prompt(
                case["query"],
                has_pasted_text=pasted is not None,
                context=routing_context,
            )
        )
        parsed = parse_tool_call(routing["raw"])
        routing["reasoning_stripped"] = parsed["reasoning_stripped"]
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
            original_query=case["query"],
        )
        if parsed["tool"] == "web_search" and tool_result.get("status") == "ok":
            admitted, admission_trace = self.evidence_admission.admit(
                case["query"],
                list(tool_result.get("evidence") or []),
            )
            tool_result = {
                **tool_result,
                "status": "ok" if admitted else "empty",
                "evidence": admitted,
                "evidence_admission": admission_trace.to_dict(),
            }
        answer_completion: dict[str, Any] | None = None
        answer_protocol: dict[str, Any] | None = None
        response_status = "ok"
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
            answer, reasoning_stripped = strip_leading_think_blocks(
                answer_completion["raw"]
            )
            answer_completion["reasoning_stripped"] = reasoning_stripped
            if parsed["tool"] == "web_search":
                answer_protocol = coordinate_answer_output(
                    answer,
                    list(tool_result.get("evidence") or []),
                    scorer=self.semantic_scorer,
                    question=case["query"],
                )
                if answer_protocol.get("valid"):
                    answer = str(answer_protocol["answer"])
                    if answer_protocol.get("partial_answer"):
                        answer += "\n" + self.answer_policy.partial_support_notice(
                            case["query"]
                        )
                else:
                    response_status = "insufficient_evidence"
                    answer = self.answer_policy.insufficient_support_answer(
                        case["query"]
                    )
        self.memory.append_exchange(
            session_id=session_id,
            user=case["query"],
            assistant=answer,
        )
        return {
            "status": response_status,
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
                "answer_protocol": answer_protocol,
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
        with self.chat_states.turn_lock(clean_session):
            self.chat_session.invalidate(clean_session)
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
        self.chat_session.close()
        self.data_plane.close()
