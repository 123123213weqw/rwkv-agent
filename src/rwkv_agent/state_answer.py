"""Answer validation, citation attribution and claim-salvage policy."""

from __future__ import annotations

import json
import re
from typing import Any

from .citations import extract_citation_ids, normalize_citation_groups, strip_citations
from .claim_verifier import (
    claim_question_relevance,
    claim_units,
    verify_answer_claims,
)
from .state_prompts import ANSWER_PREFIX, ANSWER_SUFFIX
from rwkv_search.semantic_selection import PairScorer


PROTOCOL_TAG = re.compile(
    r"</?(?:answer|tool_call|tool_calls|tool_code|tool_result)\b",
    re.I,
)
ROLE_HEADER = re.compile(r"(?:^|\n)\s*(?:System|User|Assistant|Tool):", re.I)
TERM = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]", re.I)
HTTP_URL = re.compile(r"https?://[^\s\]\)）]+", re.I)
MIN_QUESTION_RELEVANCE = 0.25

def validate_answer_output(
    raw: str,
    evidence: list[dict[str, Any]],
    *,
    require_citations: bool = True,
) -> dict[str, Any]:
    """Accept only user-visible prose and valid Web Evidence references."""

    candidate = str(raw or "").strip()
    errors: list[str] = []
    if candidate.startswith(ANSWER_PREFIX):
        candidate = candidate[len(ANSWER_PREFIX) :].lstrip()
    if ANSWER_SUFFIX in candidate:
        candidate, trailing = candidate.split(ANSWER_SUFFIX, 1)
        if trailing.strip():
            errors.append("trailing_after_answer")
    candidate = candidate.strip()
    if not candidate:
        errors.append("empty")
    if PROTOCOL_TAG.search(candidate):
        errors.append("protocol_tag")
    if ROLE_HEADER.search(candidate):
        errors.append("role_header")
    lowered = candidate.lower()
    if '"evidence"' in lowered and '"status"' in lowered:
        errors.append("evidence_payload")
    try:
        decoded = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        decoded = None
    if isinstance(decoded, (dict, list)):
        errors.append("json_payload")

    allowed_ids = {
        str(item.get("id") or "")
        for item in evidence
        if isinstance(item, dict) and item.get("id")
    }
    citations = {
        value.upper()
        for value in extract_citation_ids(candidate, prefixes={"W"})
    }
    if require_citations and evidence and not citations:
        errors.append("missing_citation")
    invalid_citations = sorted(citations - allowed_ids)
    if invalid_citations:
        errors.append("invalid_citation")
    return {
        "valid": not errors,
        "answer": candidate if not errors else "",
        "errors": errors,
        "citations": sorted(citations),
        "invalid_citations": invalid_citations,
    }


def attach_evidence_citations(
    answer: str,
    evidence: list[dict[str, Any]],
    *,
    max_citations_per_sentence: int = 2,
    scorer: PairScorer | None = None,
) -> str:
    """Attach valid evidence IDs to uncited prose using pair attribution.

    This is an output coordinator, not a query-specific rule: the model still
    writes the answer, while deterministic overlap ranking ensures every
    sentence receives only IDs that exist in the current evidence packet.
    Existing citations are preserved.
    """

    usable = [
        item
        for item in evidence
        if isinstance(item, dict) and str(item.get("id") or "").startswith("W")
    ]
    if not usable or not str(answer or "").strip():
        return str(answer or "").strip()

    evidence_terms = []
    evidence_documents: list[str] = []
    for item in usable:
        text = " ".join(
            str(value or "")
            for value in (
                item.get("title"),
                item.get("content"),
                item.get("uri"),
            )
        ).casefold()
        evidence_terms.append((str(item["id"]), set(TERM.findall(text))))
        evidence_documents.append(text)

    repaired: list[str] = []
    for sentence in claim_units(str(answer).strip()):
        if not sentence.strip() or extract_citation_ids(sentence, prefixes={"W"}):
            repaired.append(sentence)
            continue
        claim = strip_citations(sentence).strip()
        if scorer is not None:
            raw_scores = list(scorer.score(claim, evidence_documents))
            if len(raw_scores) != len(evidence_documents):
                raise ValueError("citation scorer returned an unexpected score count")
            ranked = sorted(
                (
                    (float(raw_scores[index]), -index, evidence_id)
                    for index, (evidence_id, _terms) in enumerate(evidence_terms)
                ),
                reverse=True,
            )
            positive = [item for item in ranked if item[0] > 0.0]
        else:
            terms = set(TERM.findall(claim.casefold()))
            claim_urls = {
                value.casefold().rstrip(".,;:!?。；：！？")
                for value in HTTP_URL.findall(claim)
            }
            ranked = sorted(
                (
                    (
                        100 * sum(url in evidence_documents[index] for url in claim_urls)
                        + len(terms & source_terms),
                        -index,
                        evidence_id,
                    )
                    for index, (evidence_id, source_terms) in enumerate(evidence_terms)
                ),
                reverse=True,
            )
            positive = [item for item in ranked if item[0] > 0]
        chosen = positive[: max(1, int(max_citations_per_sentence))]
        # Never attach a merely syntactically valid Evidence ID when no source
        # has positive attribution. Leaving the sentence uncited forces the
        # normal unsupported-claim/insufficient-evidence path instead of
        # laundering an answer through citation repair.
        if not chosen:
            repaired.append(sentence)
            continue
        citation = "".join(f"[{item[2]}]" for item in chosen)
        candidate = sentence.rstrip() + " " + citation
        checks = verify_answer_claims(candidate, evidence)
        if not checks or not checks[0].get("supported"):
            repaired.append(sentence)
            continue
        repaired.append(candidate)
    return "\n".join(repaired).strip()


def reattribute_unsupported_citations(
    answer: str,
    evidence: list[dict[str, Any]],
    *,
    scorer: PairScorer | None = None,
) -> tuple[str, int]:
    """Retry only unsupported claim citations against the full Evidence set.

    The model's prose is never changed. A citation replacement is accepted
    only when the same frozen verifier confirms that the claim is supported by
    the newly attributed source; otherwise the original claim remains for the
    normal drop/abstain path.
    """

    output: list[str] = []
    changed = 0
    for unit in claim_units(str(answer or "").strip()):
        checks = verify_answer_claims(unit, evidence)
        if not checks or checks[0].get("supported"):
            output.append(unit)
            continue
        claim = strip_citations(unit).strip()
        candidate = attach_evidence_citations(
            claim,
            evidence,
            scorer=scorer,
        )
        candidate_checks = verify_answer_claims(candidate, evidence)
        if candidate_checks and candidate_checks[0].get("supported"):
            output.append(candidate)
            changed += 1
        else:
            output.append(unit)
    return "\n".join(output).strip(), changed


def coordinate_answer_output(
    raw: str,
    evidence: list[dict[str, Any]],
    *,
    scorer: PairScorer | None = None,
    question: str = "",
) -> dict[str, Any]:
    """Validate prose first, then deterministically repair omitted citations."""

    allowed_ids = {
        str(item.get("id") or "").upper()
        for item in evidence
        if isinstance(item, dict) and item.get("id")
    }
    normalized = normalize_citation_groups(raw, allowed_ids=allowed_ids)
    validation = validate_answer_output(
        normalized,
        evidence,
        require_citations=False,
    )
    validation["citation_repaired"] = False
    validation["citation_reattributed_count"] = 0
    validation["citation_sanitized"] = normalized != str(raw or "")
    if not validation["valid"] or not evidence:
        return validation
    repaired = attach_evidence_citations(
        str(validation["answer"]),
        evidence,
        scorer=scorer,
    )
    reattributed, reattributed_count = reattribute_unsupported_citations(
        repaired,
        evidence,
        scorer=scorer,
    )
    coordinated = validate_answer_output(reattributed, evidence)
    coordinated["citation_repaired"] = reattributed != str(validation["answer"])
    coordinated["citation_reattributed_count"] = reattributed_count
    coordinated["citation_sanitized"] = normalized != str(raw or "")
    claims = verify_answer_claims(reattributed, evidence)
    for claim in claims:
        relevance = (
            claim_question_relevance(question, str(claim.get("text") or ""))
            if str(question or "").strip()
            else 1.0
        )
        claim["question_relevance_score"] = relevance
        claim["question_relevant"] = relevance >= MIN_QUESTION_RELEVANCE
    supported = [
        claim
        for claim in claims
        if claim.get("supported") and claim.get("question_relevant")
    ]
    unsupported = [claim for claim in claims if not claim.get("supported")]
    irrelevant = [
        claim
        for claim in claims
        if claim.get("supported") and not claim.get("question_relevant")
    ]
    rejected = [*unsupported, *irrelevant]
    salvageable: list[dict[str, Any]] = []
    for claim in claims:
        shape = str(claim.get("claim_shape") or "fragment")
        is_supported = bool(
            claim.get("supported") and claim.get("question_relevant")
        )
        # A partial answer may contain only complete, independently verified
        # sentences.  Headings and fragments are layout, not facts; retaining
        # them alone previously produced misleading lines such as
        # "official organization: [W1]" after the following claim was dropped.
        if is_supported and shape == "sentence":
            salvageable.append(claim)
    coordinated["claim_verification"] = claims
    coordinated["unsupported_claim_count"] = len(unsupported)
    coordinated["irrelevant_claim_count"] = len(irrelevant)
    coordinated["dropped_claim_count"] = 0
    coordinated["salvaged_claim_count"] = 0
    coordinated["partial_answer"] = False
    if claims and not supported:
        errors = list(coordinated["errors"])
        if unsupported:
            errors.append("unsupported_claim")
        if irrelevant:
            errors.append("irrelevant_claim")
        coordinated["valid"] = False
        coordinated["answer"] = ""
        coordinated["errors"] = list(dict.fromkeys(errors))
    elif coordinated["valid"] and rejected:
        if salvageable:
            coordinated["answer"] = "\n".join(
                f"{claim['text']} "
                + "".join(f"[{value}]" for value in claim["citations"])
                for claim in salvageable
            )
            coordinated["citations"] = sorted(
                {
                    value
                    for claim in salvageable
                    for value in claim["citations"]
                }
            )
            coordinated["dropped_claim_count"] = len(claims) - len(salvageable)
            coordinated["salvaged_claim_count"] = len(salvageable)
            coordinated["partial_answer"] = True
        else:
            coordinated["valid"] = False
            coordinated["answer"] = ""
            coordinated["errors"] = [
                *coordinated["errors"],
                "unsupported_claim",
            ]
    return coordinated


def _completion_metadata(completion: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep diagnostics without reflecting raw generated protocol payloads."""

    if completion is None:
        return None
    output = {
        key: value
        for key, value in completion.items()
        if key not in {"text", "raw", "token_ids"}
    }
    text = str(completion.get("text") or completion.get("raw") or "")
    output["output_chars"] = len(text)
    token_ids = completion.get("token_ids")
    if isinstance(token_ids, list):
        output["output_tokens"] = len(token_ids)
    return output


def _answer_validation_rank(
    value: dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    if not value or not value.get("valid"):
        return (0, 0, 0, 0)
    claims = list(value.get("claim_verification") or [])
    supported = sum(
        bool(claim.get("supported")) and bool(claim.get("question_relevant", True))
        for claim in claims
    )
    dropped = int(value.get("dropped_claim_count") or 0)
    # Prefer precision before verbosity: a longer retry cannot win merely by
    # emitting many loosely supported claims while dropping more bad ones.
    return (1, -dropped, supported, -len(str(value.get("answer") or "")))
