from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, Optional, Sequence

from .g1i_tool_call import (
    P4_SYSTEM_PROMPT as P4_SYSTEM_PROMPT,
    evaluate_web_search_tool_call,
    reconstruct_prefilled_web_search_tool_call,
    render_p4_prompt,
)
from .g1i_types import G1ICompletion
from .search_request import QueryHints, SearchRequest, SearchRequestBuilder


_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_SITE = re.compile(r"(?<!\w)site:", re.I)
_ABSOLUTE_DATE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}(?:[-/.]\d{1,2})(?:[-/.]\d{1,2})?(?!\d)|"
    r"(?<!\d)(?:19|20)\d{2}\s*年\s*\d{1,2}\s*月"
    r"(?:\s*\d{1,2}\s*[日号])?"
)
_VERSION_NUMBER = re.compile(
    r"(?<![\d.])\d+\.\d+(?:\.\d+)*(?![\d.])"
)
_QUARTER = re.compile(
    r"(?<!\w)(?:Q[1-4]|FY\s*(?:19|20)\d{2}\s*Q[1-4])(?!\w)|"
    r"第[一二三四1-4]季度",
    re.I,
)
_FISCAL_YEAR = re.compile(r"(?<!\w)FY\s*(?:19|20)\d{2}(?!\w)", re.I)
_REPAIRABLE_REASONS = frozenset(
    {
        "introduced_date",
        "introduced_fiscal_year",
        "introduced_quarter",
        "introduced_version",
        "introduced_year",
    }
)


def _normalized_matches(pattern: re.Pattern[str], value: str) -> set[str]:
    return {" ".join(match.group(0).casefold().split()) for match in pattern.finditer(value)}


def _introduced_matches(
    pattern: re.Pattern[str], raw_query: str, model_query: str
) -> list[str]:
    raw_values = _normalized_matches(pattern, raw_query)
    return [
        match.group(0)
        for match in pattern.finditer(model_query)
        if " ".join(match.group(0).casefold().split()) not in raw_values
    ]


def _remove_introduced_matches(
    pattern: re.Pattern[str], raw_query: str, value: str, removed: list[str]
) -> str:
    raw_values = _normalized_matches(pattern, raw_query)

    def replace(match: re.Match[str]) -> str:
        normalized = " ".join(match.group(0).casefold().split())
        if normalized in raw_values:
            return match.group(0)
        removed.append(match.group(0))
        return " "

    return pattern.sub(replace, value)


def sanitize_introduced_absolute_terms(
    raw_query: str, model_query: str
) -> tuple[str, tuple[str, ...]]:
    """Remove model-invented dates, versions and quarters without rewriting.

    The raw query remains the authority.  This repair is intentionally narrower
    than another model call: it only deletes absolute tokens absent from the
    user request, preserving useful translation and entity-focused wording.
    """

    value = " ".join(str(model_query or "").split())
    removed: list[str] = []
    for pattern in (
        _ABSOLUTE_DATE,
        _FISCAL_YEAR,
        _VERSION_NUMBER,
        _QUARTER,
        _YEAR,
    ):
        value = _remove_introduced_matches(pattern, raw_query, value, removed)
    value = re.sub(r"(?<!\S)年(?=\s|$)", " ", value)
    value = " ".join(value.split()).strip(" -_/.,，。")
    return value, tuple(dict.fromkeys(removed))


def evaluate_query_constraints(raw_query: str, model_query: str) -> Dict[str, Any]:
    """Validate domain-neutral invariants before executing a model query.

    A query planner may shorten or translate the request, but it must not invent
    an absolute year, expand one requested site into a site-list, or emit an
    unbounded query. Violations fail over to the raw request rather than being
    repaired into another model-authored query.
    """

    raw = str(raw_query or "")
    model = str(model_query or "")
    raw_years = set(_YEAR.findall(raw))
    model_years = set(_YEAR.findall(model))
    introduced_years = sorted(model_years - raw_years)
    introduced_dates = _introduced_matches(_ABSOLUTE_DATE, raw, model)
    introduced_versions = _introduced_matches(_VERSION_NUMBER, raw, model)
    introduced_quarters = _introduced_matches(_QUARTER, raw, model)
    introduced_fiscal_years = _introduced_matches(_FISCAL_YEAR, raw, model)
    raw_site_count = len(_SITE.findall(raw))
    model_site_count = len(_SITE.findall(model))
    allowed_site_count = max(1, raw_site_count)
    clean = " ".join(model.split())
    reasons = []
    if not clean:
        reasons.append("empty_query")
    if introduced_dates:
        reasons.append("introduced_date")
    if introduced_fiscal_years:
        reasons.append("introduced_fiscal_year")
    if introduced_quarters:
        reasons.append("introduced_quarter")
    if introduced_versions:
        reasons.append("introduced_version")
    if introduced_years:
        reasons.append("introduced_year")
    if model_site_count > allowed_site_count:
        reasons.append("excessive_site_operators")
    if len(clean) > 160:
        reasons.append("query_too_long")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "introduced_dates": introduced_dates,
        "introduced_fiscal_years": introduced_fiscal_years,
        "introduced_quarters": introduced_quarters,
        "introduced_versions": introduced_versions,
        "introduced_years": introduced_years,
        "raw_site_count": raw_site_count,
        "model_site_count": model_site_count,
        "allowed_site_count": allowed_site_count,
        "query_characters": len(clean),
    }


def resolve_model_query_constraints(raw_query: str, model_query: str) -> Dict[str, Any]:
    """Resolve a model query through bounded deletion or raw-query fallback."""

    original = evaluate_query_constraints(raw_query, model_query)
    if original["valid"]:
        return {
            "effective_query": " ".join(str(model_query or "").split()),
            "fallback_to_raw": False,
            "fallback_reason": "",
            "constraint_evaluation": {
                **original,
                "repair_applied": False,
                "removed_absolute_terms": [],
            },
        }
    reasons = set(original.get("reasons") or ())
    sanitized, removed = sanitize_introduced_absolute_terms(raw_query, model_query)
    repaired = evaluate_query_constraints(raw_query, sanitized)
    if (
        removed
        and reasons
        and reasons.issubset(_REPAIRABLE_REASONS)
        and repaired["valid"]
    ):
        return {
            "effective_query": sanitized,
            "fallback_to_raw": False,
            "fallback_reason": "",
            "constraint_evaluation": {
                **repaired,
                "repair_applied": True,
                "removed_absolute_terms": list(removed),
                "original_evaluation": original,
            },
        }
    return {
        "effective_query": " ".join(str(raw_query or "").split()),
        "fallback_to_raw": True,
        "fallback_reason": "query_constraint_violation",
        "constraint_evaluation": {
            **original,
            "repair_applied": False,
            "removed_absolute_terms": list(removed),
        },
    }


@dataclass(frozen=True)
class P4Plan:
    raw_output: str
    stop: str
    token_ids: tuple[int, ...]
    elapsed_ms: float
    format_evaluation: Dict[str, Any]
    constraint_evaluation: Dict[str, Any]
    fallback_to_raw: bool
    fallback_reason: str
    search_request: Optional[SearchRequest]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_output": self.raw_output,
            "stop": self.stop,
            "token_ids": list(self.token_ids),
            "token_count": len(self.token_ids),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "format_evaluation": self.format_evaluation,
            "constraint_evaluation": self.constraint_evaluation,
            "fallback_to_raw": self.fallback_to_raw,
            "fallback_reason": self.fallback_reason,
            "search_request": self.search_request.to_dict() if self.search_request else None,
        }


class P4SearchPlanner:
    def __init__(
        self,
        complete: Callable[[str, Sequence[str], int], G1ICompletion],
        *,
        builder: Optional[SearchRequestBuilder] = None,
        max_tokens: int = 96,
        preserve_raw_query: bool = True,
        raw_query_max_characters: int = 256,
    ) -> None:
        self.complete = complete
        self.builder = builder or SearchRequestBuilder()
        self.max_tokens = max_tokens
        self.preserve_raw_query = preserve_raw_query
        self.raw_query_max_characters = max(1, int(raw_query_max_characters))

    def plan(self, user_query: str, *, hints: QueryHints | None = None) -> P4Plan:
        completion = self.complete(
            render_p4_prompt(user_query),
            ("</tool_call>", "</tool_calls>", "</tool_code>", "\n\nUser:", "</s>"),
            self.max_tokens,
        )
        raw = reconstruct_prefilled_web_search_tool_call(
            completion.text,
            completion.stop,
        )
        evaluation = evaluate_web_search_tool_call(raw)
        resolution = resolve_model_query_constraints(
            user_query,
            str(evaluation.get("query") or ""),
        )
        format_valid = bool(evaluation.get("strict_success"))
        fallback_to_raw = not format_valid or bool(resolution["fallback_to_raw"])
        if not format_valid:
            fallback_reason = "format_invalid"
        else:
            fallback_reason = str(resolution["fallback_reason"])
        effective_model_query = (
            "" if fallback_to_raw else str(resolution["effective_query"])
        )
        request = self.builder.build(
            user_query,
            effective_model_query,
            hints=hints,
            preserve_raw_query=self.preserve_raw_query and not fallback_to_raw,
            raw_query_max_characters=self.raw_query_max_characters,
        )
        return P4Plan(
            raw_output=raw,
            stop=completion.stop,
            token_ids=completion.token_ids,
            elapsed_ms=float(completion.elapsed_ms or 0.0),
            format_evaluation=evaluation,
            constraint_evaluation=dict(resolution["constraint_evaluation"]),
            fallback_to_raw=fallback_to_raw,
            fallback_reason=fallback_reason,
            search_request=request,
        )
