"""Python model-adjacent data plane shared by Python and Rust controllers."""

from __future__ import annotations

from typing import Any

from .evidence_admission import EntityEvidenceAdmission
from .query_coordinator import QueryCoordinator
from .state_answer import coordinate_answer_output
from .state_evidence import _merge_evidence, compact_answer_evidence
from .tool_executor import ToolExecutor
from rwkv_search.pipeline.answer_policy import AnswerPolicy


class AgentDataPlane:
    """Own retrieval, long-text, Evidence and answer-validation capabilities.

    The class deliberately owns no Agent loop, chat routing, recurrent-State
    lifecycle or durable transcript. Those are control-plane responsibilities.
    """

    def __init__(
        self,
        *,
        web: Any,
        knowledge: Any,
        long_text: Any,
        session_text: Any,
        semantic_scorer: Any | None = None,
        evidence_admission: EntityEvidenceAdmission | None = None,
        answer_policy: AnswerPolicy | None = None,
        query_coordinator: QueryCoordinator | None = None,
    ) -> None:
        self.web = web
        self.knowledge = knowledge
        self.long_text = long_text
        self.session_text = session_text
        self.semantic_scorer = semantic_scorer
        self.evidence_admission = evidence_admission or EntityEvidenceAdmission()
        self.answer_policy = answer_policy or AnswerPolicy()
        self.query_coordinator = query_coordinator or QueryCoordinator()
        self.tool_executor = ToolExecutor(
            web=self.web,
            knowledge=self.knowledge,
            long_text=self.long_text,
            session_text=self.session_text,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "mode": "python_model_retrieval_data_plane",
            "tools": ["web_search", "knowledge_search", "long_text_qa"],
            "session_text": self.session_text.health(),
            "answer_validation": "fitgen-claim-lexical-v2",
            "query_coordination": "structured-query-view",
        }

    def execute_raw(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        session_id: str,
        original_query: str | None = None,
    ) -> dict[str, Any]:
        return self.tool_executor.execute(
            name,
            arguments,
            session_id=session_id,
            original_query=original_query,
        )

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        session_id: str,
        original_query: str | None = None,
    ) -> dict[str, Any]:
        result = self.execute_raw(
            name,
            arguments,
            session_id=session_id,
            original_query=original_query,
        )
        if name != "web_search" or result.get("status") != "ok":
            return result
        admitted, trace = self.evidence_admission.admit(
            str(original_query or arguments.get("query") or ""),
            list(result.get("evidence") or []),
        )
        return {
            **result,
            "status": "ok" if admitted else "empty",
            "evidence": admitted,
            "evidence_admission": trace.to_dict(),
        }

    def capture_text(self, session_id: str, text: str) -> dict[str, Any]:
        pasted = self.session_text.put(session_id, text)
        return {
            "status": "accepted",
            "document": {
                "source": "session_pasted_text",
                "name": pasted.name,
                "chars": pasted.chars,
                "sha256": pasted.sha256,
            },
        }

    def text_status(self, session_id: str) -> dict[str, Any]:
        pasted = self.session_text.get(session_id)
        if pasted is None:
            return {"status": "empty", "active": False}
        return {
            "status": "ok",
            "active": True,
            "document": {
                "source": "session_pasted_text",
                "name": pasted.name,
                "chars": pasted.chars,
                "sha256": pasted.sha256,
            },
        }

    def validate_answer(
        self,
        *,
        question: str,
        answer: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return coordinate_answer_output(
            answer,
            evidence,
            scorer=self.semantic_scorer,
            question=question,
        )

    def reduce_evidence(
        self,
        *,
        question: str,
        tool_results: list[dict[str, Any]],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        merged = _merge_evidence(
            tool_results,
            question=question,
            limit=max(1, min(int(limit), 16)),
            scorer=self.semantic_scorer,
            preserve_query_views=False,
        )
        return compact_answer_evidence(
            question,
            merged,
            max_chars_per_source=900,
        )

    def coordinate_query(
        self,
        *,
        question: str,
        generated_query: str,
        branch_index: int,
        round_index: int,
        observation: dict[str, Any] | None,
        used_queries: list[str],
    ) -> dict[str, Any]:
        view = self.query_coordinator.coordinate(
            question,
            generated_query,
            branch_index=int(branch_index),
            round_index=int(round_index),
            observation=observation,
            used_queries=set(used_queries),
        )
        return view.to_trace()

    def close(self) -> None:
        self.session_text.close()
        knowledge_close = getattr(self.knowledge, "close", None)
        if callable(knowledge_close):
            knowledge_close()
        self.web.close()
