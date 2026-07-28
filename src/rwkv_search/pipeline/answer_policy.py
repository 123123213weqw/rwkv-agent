from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping


_CITATION = re.compile(r"\[([A-Za-z]+\d+)\]")


@dataclass(frozen=True)
class AnswerPolicyResult:
    valid: bool
    answer: str
    citations: tuple[str, ...]
    errors: tuple[str, ...]


class AnswerPolicy:
    """Domain-neutral evidence and output policy.

    Risk/source requirements are inputs to this layer; it never guesses a
    business domain from query words.
    """

    @staticmethod
    def no_evidence_answer(question: str) -> str:
        return (
            "没有找到足以回答该问题的可核查证据。"
            if any("\u3400" <= char <= "\u9fff" for char in str(question))
            else "No sufficient verifiable evidence was found."
        )

    @staticmethod
    def generation_failure(question: str) -> str:
        return (
            "检索已完成，但回答生成未通过输出协议校验。请重试。"
            if any("\u3400" <= char <= "\u9fff" for char in str(question))
            else "Retrieval completed, but answer generation failed validation. Please retry."
        )

    @staticmethod
    def insufficient_support_answer(question: str) -> str:
        return (
            "找到了一些相关信息，但现有证据不足以支持可靠结论。"
            if any("\u3400" <= char <= "\u9fff" for char in str(question))
            else (
                "Related information was found, but the evidence does not "
                "support a reliable conclusion."
            )
        )

    @staticmethod
    def partial_support_notice(question: str) -> str:
        return (
            "其余部分缺少足够证据，未予输出。"
            if any("\u3400" <= char <= "\u9fff" for char in str(question))
            else "Other requested details were omitted because support was insufficient."
        )

    @staticmethod
    def chat_failure(question: str) -> str:
        return (
            "模型本次没有生成可显示的回答，请重试。"
            if any("\u3400" <= char <= "\u9fff" for char in str(question))
            else "The model did not produce a displayable answer. Please retry."
        )

    @staticmethod
    def validate_citations(
        answer: str,
        evidence: Iterable[Mapping[str, Any]],
        *,
        require_when_evidence: bool = True,
    ) -> AnswerPolicyResult:
        text = str(answer or "").strip()
        allowed = {
            str(item.get("id") or "").upper()
            for item in evidence
            if item.get("id")
        }
        citations = tuple(dict.fromkeys(value.upper() for value in _CITATION.findall(text)))
        errors: list[str] = []
        if not text:
            errors.append("empty")
        if require_when_evidence and allowed and not citations:
            errors.append("missing_citation")
        if any(value not in allowed for value in citations):
            errors.append("invalid_citation")
        return AnswerPolicyResult(
            valid=not errors,
            answer=text if not errors else "",
            citations=citations,
            errors=tuple(errors),
        )
