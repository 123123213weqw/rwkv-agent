"""Bounded State-native parallel Web-search Agent experiment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Callable
import uuid

from .evidence_admission import EntityEvidenceAdmission
from .query_coordinator import QueryCoordinator
from .state_answer import (
    _answer_validation_rank,
    _completion_metadata,
    attach_evidence_citations as attach_evidence_citations,
    coordinate_answer_output as coordinate_answer_output,
    reattribute_unsupported_citations as reattribute_unsupported_citations,
    validate_answer_output as validate_answer_output,
)
from .state_evidence import (
    _compact_branch_observation,
    _merge_evidence,
    compact_answer_evidence,
)
from .state_prompts import (
    ANSWER_MAX_TOKENS,
    ANSWER_PREFIX as ANSWER_PREFIX,
    ANSWER_STOPS,
    ANSWER_SUFFIX as ANSWER_SUFFIX,
    BRANCH_MISSIONS,
    TOOL_CALL_PREFIX as TOOL_CALL_PREFIX,
    reconstruct_tool_call,
    render_answer_fallback_prompt as render_answer_fallback_prompt,
    render_branch_step,
    render_compact_answer_prompt,
    render_root_final_input,
    render_root_prompt,
)
from rwkv_search.pipeline.answer_policy import AnswerPolicy
from rwkv_search.semantic_selection import PairScorer


def coordinate_search_query(
    question: str,
    generated_query: str,
    *,
    branch_index: int,
    round_index: int,
    observation: dict[str, Any] | None,
    used_queries: set[str],
) -> tuple[str, str]:
    """Compatibility wrapper around the structured Query View Coordinator."""

    view = QueryCoordinator().coordinate(
        question,
        generated_query,
        branch_index=branch_index,
        round_index=round_index,
        observation=observation,
        used_queries=used_queries,
    )
    return view.query, view.strategy


class StateNativeSearchAgent:
    """Fork recurrent states, run bounded search rounds, then resume the root."""

    def __init__(
        self,
        *,
        state_model: Any,
        parse_tool_call: Callable[[str], dict[str, Any]],
        execute_tool: Callable[..., dict[str, Any]],
        evidence_scorer: PairScorer | None = None,
        answer_policy: AnswerPolicy | None = None,
        query_coordinator: QueryCoordinator | None = None,
        evidence_admission: EntityEvidenceAdmission | None = None,
        preserve_query_view_evidence: bool = False,
    ) -> None:
        self.state_model = state_model
        self.parse_tool_call = parse_tool_call
        self.execute_tool = execute_tool
        self.evidence_scorer = evidence_scorer
        self.answer_policy = answer_policy or AnswerPolicy()
        self.query_coordinator = query_coordinator or QueryCoordinator()
        self.evidence_admission = evidence_admission or EntityEvidenceAdmission()
        self.preserve_query_view_evidence = bool(preserve_query_view_evidence)

    def run(
        self,
        question: str,
        *,
        session_id: str,
        branch_width: int = 4,
        max_rounds: int = 2,
    ) -> dict[str, Any]:
        if not 1 <= int(branch_width) <= len(BRANCH_MISSIONS):
            raise ValueError("branch_width must be between 1 and 4")
        if not 1 <= int(max_rounds) <= 3:
            raise ValueError("max_rounds must be between 1 and 3")
        started = time.perf_counter()
        owner_id = f"turn-{session_id[:48]}-{uuid.uuid4().hex}"
        root: dict[str, Any] | None = None
        branches: list[dict[str, Any]] = []
        all_state_ids: list[str] = []
        home_url = ""
        round_traces: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        previous_by_state: dict[str, dict[str, Any]] = {}
        used_queries: set[str] = set()
        release_trace: dict[str, Any] | None = None
        response: dict[str, Any] | None = None
        try:
            root = self.state_model.state_prefill(
                owner_id=owner_id,
                prompt=render_root_prompt(question),
            )
            home_url = str(root["home_url"])
            root_id = str(root["state_id"])
            all_state_ids.append(root_id)
            missions = list(BRANCH_MISSIONS[: int(branch_width)])
            branches = self.state_model.state_fork(
                home_url=home_url,
                owner_id=owner_id,
                parent_state_id=root_id,
                branches=[f"branch-{index + 1}" for index in range(len(missions))],
            )
            all_state_ids.extend(str(value["state_id"]) for value in branches)

            for round_index in range(1, int(max_rounds) + 1):
                inputs = [
                    {
                        "state_id": str(branch["state_id"]),
                        "input": render_branch_step(
                            question=question,
                            mission=missions[index],
                            round_index=round_index,
                            observation=previous_by_state.get(str(branch["state_id"])),
                        ),
                    }
                    for index, branch in enumerate(branches)
                ]
                completions = self.state_model.state_batch_continue(
                    home_url=home_url,
                    owner_id=owner_id,
                    items=inputs,
                    stops=["</tool_call>"],
                    max_tokens=96,
                )
                parsed_rows = []
                for index, completion in enumerate(completions):
                    raw = reconstruct_tool_call(completion)
                    parsed = self.parse_tool_call(raw)
                    if parsed.get("tool") != "web_search":
                        parsed = {
                            "strict": False,
                            "tool": "",
                            "arguments": {},
                            "error": "state search branch must call web_search",
                        }
                    elif parsed.get("strict"):
                        model_arguments = dict(parsed.get("arguments") or {})
                        query_view = self.query_coordinator.coordinate(
                            question,
                            str(model_arguments.get("query") or ""),
                            branch_index=index,
                            round_index=round_index,
                            observation=previous_by_state.get(
                                str(completion["state_id"])
                            ),
                            used_queries=used_queries,
                        )
                        if query_view.accepted:
                            parsed = {
                                **parsed,
                                "model_arguments": model_arguments,
                                "arguments": {"query": query_view.query},
                                "query_strategy": query_view.strategy,
                                "query_view": query_view.to_trace(),
                            }
                        else:
                            parsed = {
                                **parsed,
                                "strict": False,
                                "model_arguments": model_arguments,
                                "arguments": {},
                                "query_strategy": query_view.strategy,
                                "query_view": query_view.to_trace(),
                                "error": "query coordinator rejected all views",
                            }
                    parsed_rows.append((completion, parsed, raw))

                def execute(row: tuple[dict[str, Any], dict[str, Any], str]):
                    completion, parsed, _raw = row
                    if not parsed.get("strict"):
                        return completion, {
                            "status": "invalid",
                            "evidence": [],
                            "message": str(parsed.get("error") or "route error"),
                        }
                    result = self.execute_tool(
                        "web_search",
                        parsed["arguments"],
                        session_id=session_id,
                    )
                    if result.get("status") != "ok":
                        return completion, result
                    admitted, admission_trace = self.evidence_admission.admit(
                        question,
                        list(result.get("evidence") or []),
                    )
                    result = {
                        **result,
                        "status": "ok" if admitted else "empty",
                        "evidence": admitted,
                        "evidence_admission": admission_trace.to_dict(),
                    }
                    return completion, result

                with ThreadPoolExecutor(
                    max_workers=len(parsed_rows),
                    thread_name_prefix="rwkv-state-search",
                ) as executor:
                    executed = list(executor.map(execute, parsed_rows))

                branch_traces = []
                for (completion, parsed, raw), (_same, result) in zip(
                    parsed_rows,
                    executed,
                    strict=True,
                ):
                    state_id = str(completion["state_id"])
                    previous_by_state[state_id] = _compact_branch_observation(result)
                    if result.get("status") == "ok":
                        tool_results.append(result)
                    branch_traces.append(
                        {
                            "state_id": state_id,
                            "branch": completion.get("branch"),
                            "raw": raw,
                            "route": parsed,
                            "tool_status": result.get("status"),
                            "evidence_count": len(result.get("evidence") or []),
                            "evidence_admission": result.get(
                                "evidence_admission"
                            ),
                            "effective_query": result.get("effective_query"),
                            "scope_root": result.get("scope_root"),
                            "seen_tokens": completion.get("seen_tokens"),
                        }
                    )
                round_traces.append({"round": round_index, "branches": branch_traces})

            evidence = compact_answer_evidence(
                question,
                _merge_evidence(
                    tool_results,
                    question=question,
                    limit=8,
                    scorer=self.evidence_scorer,
                    preserve_query_views=self.preserve_query_view_evidence,
                ),
                max_chars_per_source=900,
            )
            answer_completion: dict[str, Any] | None = None
            fallback_completion: dict[str, Any] | None = None
            primary_validation: dict[str, Any] | None = None
            fallback_validation: dict[str, Any] | None = None
            fallback_error = ""
            response_status = "ok"
            partial_support_notice_appended = False
            if not evidence:
                answer = self.answer_policy.no_evidence_answer(question)
            else:
                answer_completion = self.state_model.state_batch_continue(
                    home_url=home_url,
                    owner_id=owner_id,
                    items=[
                        {
                            "state_id": root_id,
                            "input": render_root_final_input(question, evidence),
                        }
                    ],
                    stops=list(ANSWER_STOPS),
                    max_tokens=ANSWER_MAX_TOKENS,
                )[0]
                primary_validation = coordinate_answer_output(
                    str(answer_completion.get("text") or ""),
                    evidence,
                    scorer=self.evidence_scorer,
                    question=question,
                )
                if (
                    not primary_validation["valid"]
                    or primary_validation.get("partial_answer")
                ):
                    try:
                        fallback_completion = self.state_model.complete(
                            render_compact_answer_prompt(question, evidence),
                            max_tokens=ANSWER_MAX_TOKENS,
                            stops=list(ANSWER_STOPS),
                        )
                        fallback_validation = coordinate_answer_output(
                            str(fallback_completion.get("raw") or ""),
                            evidence,
                            scorer=self.evidence_scorer,
                            question=question,
                        )
                    except Exception as exc:
                        fallback_error = f"{type(exc).__name__}: {exc}"[:300]
                        fallback_validation = {
                            "valid": False,
                            "answer": "",
                            "errors": ["fallback_runtime_error"],
                            "citations": [],
                            "invalid_citations": [],
                        }
                chosen_validation = max(
                    (primary_validation, fallback_validation),
                    key=_answer_validation_rank,
                )
                if chosen_validation and chosen_validation.get("valid"):
                    answer = str(chosen_validation["answer"])
                    if chosen_validation.get("partial_answer"):
                        answer += "\n" + self.answer_policy.partial_support_notice(
                            question
                        )
                        partial_support_notice_appended = True
                else:
                    support_failure = any(
                        any(
                            error in {"unsupported_claim", "irrelevant_claim"}
                            for error in list((value or {}).get("errors") or [])
                        )
                        for value in (
                            primary_validation,
                            fallback_validation,
                        )
                    )
                    if support_failure:
                        response_status = "insufficient_evidence"
                        answer = self.answer_policy.insufficient_support_answer(
                            question
                        )
                    else:
                        response_status = "answer_error"
                        answer = self.answer_policy.generation_failure(question)
            response = {
                "status": response_status,
                "session_id": session_id,
                "message": question,
                "route": {
                    "mode": "state_parallel_search",
                    "tool": "web_search",
                    "branch_width": len(branches),
                    "rounds": int(max_rounds),
                },
                "tool_result": {
                    "status": "ok" if evidence else "empty",
                    "tool": "web_search",
                    "evidence": evidence,
                },
                "answer": answer,
                "trace": {
                    "state_runtime": {
                        "home_url": home_url,
                        "root_prefill_once": True,
                        "forked_states": len(branches),
                        "tensor_state_merge": False,
                        "semantic_reduce_to_root": True,
                    },
                    "rounds": round_traces,
                    "answer_completion": _completion_metadata(answer_completion),
                    "answer_evidence_profile": {
                        "name": "compact-question-span-v1",
                        "sources": len(evidence),
                        "max_chars_per_source": 900,
                        "selector": (
                            "cross_view_confirmed_append_mmr_v1"
                            if self.preserve_query_view_evidence
                            else "query_view_mmr_v1"
                        ),
                        "scorer_model": str(
                            getattr(self.evidence_scorer, "model_name", "")
                        ),
                    },
                    "answer_protocol": {
                        "format": "answer_envelope_v1",
                        "primary": primary_validation,
                        "fallback_used": fallback_completion is not None
                        or bool(fallback_error),
                        "fallback": fallback_validation,
                        "fallback_error": fallback_error,
                        "fallback_completion": _completion_metadata(
                            fallback_completion
                        ),
                        "policy_notice": (
                            "partial_support"
                            if partial_support_notice_appended
                            else ""
                        ),
                    },
                    "elapsed_ms": round(
                        (time.perf_counter() - started) * 1000.0,
                        3,
                    ),
                },
            }
            return response
        finally:
            if home_url and all_state_ids:
                try:
                    release_trace = self.state_model.state_release(
                        home_url=home_url,
                        owner_id=owner_id,
                        state_ids=all_state_ids,
                    )
                except Exception as exc:
                    release_trace = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            if response is not None:
                response["trace"]["state_runtime"]["release"] = release_trace
