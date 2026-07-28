"""Bounded State-native parallel Web-search Agent experiment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
from typing import Any, Callable
import uuid

from .citations import extract_citation_ids, normalize_citation_groups, strip_citations
from .claim_verifier import claim_units, verify_answer_claims
from .evidence_admission import EntityEvidenceAdmission
from .query_coordinator import QueryCoordinator
from rwkv_search.pipeline.answer_policy import AnswerPolicy
from rwkv_search.semantic_selection import PairScorer, select_diverse_items
from rwkv_search.text import canonicalize_url


BRANCH_MISSIONS = (
    "Find the primary answer and the strongest directly relevant source.",
    "Prefer official, primary, or first-party sources for the key claims.",
    "Find an independent source that corroborates the likely answer.",
    "Look for missing facts, ambiguity, date issues, or contradictory sources.",
)
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
ANSWER_MAX_TOKENS = 420
PROTOCOL_TAG = re.compile(
    r"</?(?:answer|tool_call|tool_calls|tool_code|tool_result)\b",
    re.I,
)
ROLE_HEADER = re.compile(r"(?:^|\n)\s*(?:System|User|Assistant|Tool):", re.I)
TERM = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]", re.I)
HTTP_URL = re.compile(r"https?://[^\s\]\)）]+", re.I)
PRIMARY_EVIDENCE_SOURCES = frozenset(
    {
        "company_filing",
        "crossref",
        "github",
        "government",
        "mediawiki",
        "official_docs",
        "official_repository",
        "paper",
        "regulator",
    }
)


def render_root_prompt(question: str) -> str:
    return (
        "System: You are a bounded state-native research agent. The Controller "
        "will fork this recurrent state into independent branches. In a branch, "
        "obey only the current branch-step instruction. In the retained root, "
        "answer only at the explicitly marked final-answer stage, only from the "
        "supplied Evidence, and cite every factual claim with its Evidence ID. "
        "Never call a function from the retained root. If Evidence is "
        "insufficient, say so. Do not expose reasoning.\n\n"
        f"User: {question.strip()}"
    )


def render_branch_step(
    *,
    question: str,
    mission: str,
    round_index: int,
    observation: dict[str, Any] | None,
) -> str:
    if round_index == 1:
        return (
            "\n\nUser: Branch mission: "
            + mission
            + "\nOriginal question: "
            + question.strip()
            + "\nProduce one focused web_search call now. The JSON must have "
            'exactly this shape: {"name":"web_search","arguments":'
            '{"query":"..."}}. The arguments object must contain only query.\n\n'
            "Assistant: " + TOOL_CALL_PREFIX
        )
    compact = json.dumps(
        observation or {"status": "empty", "evidence": []},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "\n\nTool: <tool_result>"
        + compact
        + "</tool_result>\n\nUser: Continue the same branch mission. Do not "
        "repeat the previous query. Search for the most important missing fact "
        "or verify the current claim with a better independent source. Output "
        "exactly one web_search tool call. The JSON must have exactly this "
        'shape: {"name":"web_search","arguments":{"query":"..."}}. '
        "The arguments object must contain only query.\n\nAssistant: "
        + TOOL_CALL_PREFIX
    )


def reconstruct_tool_call(result: dict[str, Any]) -> str:
    """Restore the prefix committed in the continuation prompt.

    G1I is greedily continued after ``<tool_call>`` so it cannot spend the
    bounded output budget on a hidden-reasoning preamble. Test doubles and
    older traces may still return the opening tag themselves, which must not
    be duplicated.
    """

    generated = str(result.get("text") or "").lstrip()
    raw = (
        generated
        if generated.startswith(TOOL_CALL_PREFIX)
        else TOOL_CALL_PREFIX + generated
    )
    if result.get("stop_reason") == "</tool_call>":
        raw += "</tool_call>"
    return raw


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


def _compact_branch_observation(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "evidence": [
            {
                "id": item.get("id"),
                "title": str(item.get("title") or "")[:160],
                "content": str(item.get("content") or "")[:700],
                "uri": item.get("uri"),
                "source": item.get("source"),
                "published_at": item.get("published_at"),
                "score": item.get("score"),
                "discovery_stage": item.get("discovery_stage"),
            }
            for item in list(result.get("evidence") or [])[:5]
            if isinstance(item, dict)
        ],
    }


def _merge_evidence(
    results: list[dict[str, Any]],
    *,
    question: str = "",
    limit: int = 12,
    scorer: PairScorer | None = None,
) -> list[dict[str, Any]]:
    """Select complementary evidence from model-generated query views.

    The branch queries are the information-needs representation.  Selection is
    therefore independent of topic words, domains, URL shapes, and fixed source
    records; it uses the shared query-view MMR selector instead.
    """

    candidates: list[dict[str, Any]] = []
    query_views: list[str] = []
    for result in results:
        result_query = " ".join(str(result.get("query") or "").split()).strip()
        if result_query:
            query_views.append(result_query)
        for position, item in enumerate(result.get("evidence") or [], start=1):
            if not isinstance(item, dict):
                continue
            uri = str(item.get("uri") or "").strip()
            content = str(item.get("content") or "").strip()
            if not uri and not content:
                continue
            candidates.append(
                {
                    "title": str(item.get("title") or "")[:240],
                    "content": content[:1800],
                    "uri": uri,
                    "source": str(item.get("source") or ""),
                    "published_at": item.get("published_at"),
                    "discovery_stage": str(item.get("discovery_stage") or ""),
                    "_best_position": position,
                    "_upstream_score": float(item.get("score") or 0.0),
                }
            )
    primary_candidates = [
        item
        for item in candidates
        if str(item.get("source") or "") in PRIMARY_EVIDENCE_SOURCES
    ]
    primary_by_stage: dict[str, dict[str, Any]] = {}
    primary_without_stage: list[dict[str, Any]] = []
    for item in primary_candidates:
        stage = str(item.get("discovery_stage") or "").strip()
        if not stage:
            primary_without_stage.append(item)
            continue
        key = f"{item.get('source') or ''}:{stage}"
        current = primary_by_stage.get(key)
        if current is None or (
            str(item.get("published_at") or ""),
            float(item.get("_upstream_score") or 0.0),
            len(str(item.get("content") or "")),
        ) > (
            str(current.get("published_at") or ""),
            float(current.get("_upstream_score") or 0.0),
            len(str(current.get("content") or "")),
        ):
            primary_by_stage[key] = item
    primary_candidates = [*primary_by_stage.values(), *primary_without_stage]
    reserved = select_diverse_items(
        question,
        query_views,
        primary_candidates,
        # Keep enough primary records to cover compound questions (identity,
        # collection/index and latest event) while leaving half of the normal
        # twelve-item budget for independent corroboration.
        limit=min(6, int(limit), len(primary_candidates)),
        scorer=scorer,
    ).items
    reserved_keys = {
        canonicalize_url(str(item.get("uri") or ""))
        or str(item.get("uri") or "").casefold()
        for item in reserved
    }
    remainder = [
        item
        for item in candidates
        if (
            canonicalize_url(str(item.get("uri") or ""))
            or str(item.get("uri") or "").casefold()
        )
        not in reserved_keys
    ]
    selected = [
        *reserved,
        *select_diverse_items(
            question,
            query_views,
            remainder,
            limit=max(0, int(limit) - len(reserved)),
            scorer=scorer,
        ).items,
    ]
    output = [
        {
            "id": f"W{index}",
            "title": str(item["title"]),
            "content": str(item["content"]),
            "uri": str(item["uri"]),
            **(
                {"source": str(item["source"])}
                if item.get("source")
                else {}
            ),
            **(
                {"published_at": item["published_at"]}
                if item.get("published_at")
                else {}
            ),
            **(
                {"discovery_stage": str(item["discovery_stage"])}
                if item.get("discovery_stage")
                else {}
            ),
        }
        for index, item in enumerate(selected, start=1)
    ]
    return output


def render_root_final_input(
    question: str,
    evidence: list[dict[str, Any]],
) -> str:
    observation = json.dumps(
        {"status": "ok" if evidence else "empty", "evidence": evidence},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "\n\nTool: <tool_result>"
        + observation
        + "</tool_result>\n\nUser: Final answer stage. Answer the original "
        "question directly in its language using only this Evidence. Cite every "
        "factual claim with [W#]. Preserve the exact relation stated by a source: "
        "do not upgrade author, owner, or maintainer into founder or creator. "
        "When the requested relation is not explicit, report the closest verified "
        "relation and say what remains unverified. Put each independently "
        "verifiable fact in its own sentence or bullet; never combine an item "
        "inventory with a date or latest-event claim. Omit unrequested background. "
        "Never invent an Evidence ID. If the Evidence "
        "does not support an answer, explicitly say it is insufficient. Keep the "
        "answer concise. The opening <answer> tag is already supplied. Output only "
        "the user-visible answer text followed by </answer>; never reproduce the "
        "Tool Result, JSON, role labels, or another protocol tag. Original question: "
        + question.strip()
        + "\n\nAssistant: "
        + ANSWER_PREFIX
    )


def render_answer_fallback_prompt(
    question: str,
    evidence: list[dict[str, Any]],
) -> str:
    """Render an independent answer-only retry without repeating Web search."""

    observation = json.dumps(
        {"status": "ok" if evidence else "empty", "evidence": evidence},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "System: You are the final evidence answer stage. Tools are unavailable. "
        "Answer the current question directly in its language using only the "
        "supplied Evidence. Cite factual claims with existing [W#] IDs. Never "
        "upgrade author, owner, or maintainer into founder or creator; report "
        "the closest verified relation instead. Use one independently verifiable "
        "fact per sentence or bullet and omit unrequested background. Never invent an ID. Never output Tool Result, JSON, role labels, a tool call, "
        "or reasoning. The opening <answer> tag is already supplied; output only "
        "the concise user-visible answer text followed by </answer>.\n\n"
        "User: Who maintains ExampleDB?\n\n"
        'Tool: <tool_result>{"status":"ok","evidence":[{"id":"W1",'
        '"title":"ExampleDB","content":"ExampleDB is maintained by '
        'Example Foundation.","uri":"https://example.invalid/db"}]}'
        "</tool_result>\n\n"
        "Assistant: <answer>Example Foundation maintains ExampleDB [W1]."
        "</answer>\n\n"
        "User: 示例系统由谁维护？\n\n"
        'Tool: <tool_result>{"status":"ok","evidence":[{"id":"W1",'
        '"title":"示例系统","content":"示例系统由示例基金会维护。",'
        '"uri":"https://example.invalid/zh"}]}</tool_result>\n\n'
        "Assistant: <answer>示例系统由示例基金会维护 [W1]。</answer>\n\n"
        "User: "
        + question.strip()
        + "\n\nTool: <tool_result>"
        + observation
        + "</tool_result>\n\nAssistant: "
        + ANSWER_PREFIX
    )


def compact_answer_evidence(
    question: str,
    evidence: list[dict[str, Any]],
    *,
    max_chars_per_source: int = 900,
) -> list[dict[str, Any]]:
    """Keep one question-relevant bounded span per evidence source.

    This is a Gold-blind context-budget operation.  It preserves source IDs and
    URIs while ensuring answer-stage training and inference see the same compact
    evidence shape instead of silently left-truncating several whole documents.
    """

    from .tools.long_text import chunk_text, rank_chunks

    if max_chars_per_source < 256:
        raise ValueError("max_chars_per_source must be at least 256")
    output: list[dict[str, Any]] = []
    for item in evidence:
        value = dict(item)
        content = str(value.get("content") or "").strip()
        if content:
            chunks = chunk_text(
                content,
                max_chars=max_chars_per_source,
                overlap_chars=min(80, max_chars_per_source // 4),
            )
            selected = rank_chunks(
                f"{question} {value.get('title') or ''}",
                chunks,
                top_k=1,
            )
            if selected:
                value["content"] = selected[0][1].text[:max_chars_per_source]
            else:
                value["content"] = content[:max_chars_per_source]
        output.append(value)
    return output


def render_compact_answer_prompt(
    question: str,
    evidence: list[dict[str, Any]],
) -> str:
    """Render the production answer protocol without few-shot context overhead."""

    observation = json.dumps(
        {"status": "ok" if evidence else "empty", "evidence": evidence},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "System: Answer the current question directly using only the supplied "
        "Evidence. Cite each factual claim with existing [W#] IDs. Never invent "
        "an ID. Preserve source relation labels; never upgrade author, owner, or "
        "maintainer into founder or creator. Do not call a tool, expose reasoning, "
        "or output JSON or role labels. Use one independently verifiable fact per "
        "sentence or bullet and omit unrequested background. "
        "The opening <answer> tag is already supplied; output only a concise "
        "user-visible answer followed by </answer>.\n\nUser: "
        + question.strip()
        + "\n\nTool: <tool_result>"
        + observation
        + "</tool_result>\n\nAssistant: "
        + ANSWER_PREFIX
    )


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
            positive = ranked
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
        if not chosen:
            chosen = [ranked[0]]
        citation = "".join(f"[{item[2]}]" for item in chosen)
        repaired.append(sentence.rstrip() + " " + citation)
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
    claims = verify_answer_claims(str(coordinated.get("answer") or ""), evidence)
    supported = [claim for claim in claims if claim.get("supported")]
    unsupported = [claim for claim in claims if not claim.get("supported")]
    salvageable: list[dict[str, Any]] = []
    supported_heading = False
    for claim in claims:
        shape = str(claim.get("claim_shape") or "fragment")
        is_supported = bool(claim.get("supported"))
        if shape == "heading":
            supported_heading = is_supported
            if is_supported:
                salvageable.append(claim)
            continue
        if is_supported and (shape == "sentence" or supported_heading):
            salvageable.append(claim)
    coordinated["claim_verification"] = claims
    coordinated["unsupported_claim_count"] = len(unsupported)
    coordinated["dropped_claim_count"] = 0
    coordinated["salvaged_claim_count"] = 0
    coordinated["partial_answer"] = False
    if coordinated["valid"] and unsupported:
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
    supported = sum(bool(claim.get("supported")) for claim in claims)
    dropped = int(value.get("dropped_claim_count") or 0)
    # Prefer precision before verbosity: a longer retry cannot win merely by
    # emitting many loosely supported claims while dropping more bad ones.
    return (1, -dropped, supported, -len(str(value.get("answer") or "")))


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
    ) -> None:
        self.state_model = state_model
        self.parse_tool_call = parse_tool_call
        self.execute_tool = execute_tool
        self.evidence_scorer = evidence_scorer
        self.answer_policy = answer_policy or AnswerPolicy()
        self.query_coordinator = query_coordinator or QueryCoordinator()
        self.evidence_admission = evidence_admission or EntityEvidenceAdmission()

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
                ),
                max_chars_per_source=900,
            )
            answer_completion: dict[str, Any] | None = None
            fallback_completion: dict[str, Any] | None = None
            primary_validation: dict[str, Any] | None = None
            fallback_validation: dict[str, Any] | None = None
            fallback_error = ""
            response_status = "ok"
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
                else:
                    support_failure = any(
                        "unsupported_claim" in list(
                            (value or {}).get("errors") or []
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
                        "selector": "query_view_mmr_v1",
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
