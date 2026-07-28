"""Frozen, citation-aware lexical claim verifier for benchmark scoring.

The verifier deliberately has no access to benchmark Gold answers. It only
compares each answer claim with the evidence IDs cited by that claim. This is
not an NLI model; it is a conservative, reproducible support check used to
detect uncited or textually unsupported generations before a blind run.
"""

from __future__ import annotations

from collections import Counter
import re
import unicodedata
from typing import Any, Mapping, Sequence

from .citations import extract_citation_ids, strip_citations


VERIFIER_VERSION = "fitgen-claim-lexical-v1"
_CITATION_AT = re.compile(r"\[[A-Za-z][A-Za-z0-9_.:-]*\]")
_WORD = re.compile(r"[a-z0-9]+(?:[_.+-][a-z0-9]+)*", re.I)
_NUMBER = re.compile(r"(?<![A-Za-z0-9])[+-]?\d+(?:[.,]\d+)*(?:%|‰)?", re.I)
_URL = re.compile(r"https?://[^\s\]\)）]+", re.I)
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")
_BOUNDARY = frozenset("。！？!?.")
_FACTUAL_TEXT = re.compile(r"[A-Za-z\u3400-\u9fff]")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
        "其",
        "及",
        "和",
        "是",
        "由",
        "的",
        "了",
        "在",
        "为",
    }
)


def _normalize(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value or "")).casefold().split()
    )


def _canonical_number(value: str) -> str:
    normalized = _normalize(value)
    suffix = normalized[-1:] if normalized[-1:] in {"%", "‰"} else ""
    body = normalized[:-1] if suffix else normalized
    if re.fullmatch(r"[+-]?\d+", body):
        return f"{int(body)}{suffix}"
    return normalized


def claim_units(answer: str) -> list[str]:
    """Split prose while keeping immediately following citation groups."""

    output: list[str] = []
    for raw_line in str(answer or "").splitlines():
        line = raw_line.strip().lstrip("-*• ")
        if not line:
            continue
        start = 0
        index = 0
        while index < len(line):
            char = line[index]
            # Decimal points and dotted identifiers are not sentence ends.
            if (
                char == "."
                and index > 0
                and index + 1 < len(line)
                and line[index - 1].isalnum()
                and line[index + 1].isalnum()
            ):
                index += 1
                continue
            if char == ".":
                prefix = line[:index]
                suffix = line[index + 1 :]
                repeated_initials = re.search(
                    r"(?:\b[A-Za-z]\.){1,5}[A-Za-z]$",
                    prefix,
                )
                name_initial = re.search(
                    r"\b[A-Z][a-z]+\s+[A-Z]$",
                    prefix,
                ) and re.match(r"\s+[A-Z][a-z]+", suffix)
                if repeated_initials or name_initial:
                    index += 1
                    continue
            if char not in _BOUNDARY:
                index += 1
                continue
            end = index + 1
            while end < len(line) and line[end].isspace():
                end += 1
            while True:
                match = _CITATION_AT.match(line, end)
                if match is None:
                    break
                end = match.end()
                while end < len(line) and line[end].isspace():
                    end += 1
            unit = line[start:end].strip()
            if unit:
                output.append(unit)
            start = end
            index = end
        tail = line[start:].strip()
        if tail:
            output.append(tail)
    return output


def _features(value: str) -> list[str]:
    normalized = _normalize(value)
    features = [
        match.group(0)
        for match in _WORD.finditer(normalized)
        if match.group(0) not in _STOPWORDS
    ]
    for run in _CJK_RUN.findall(normalized):
        if len(run) == 1:
            if run not in _STOPWORDS:
                features.append(run)
            continue
        features.extend(run[index : index + 2] for index in range(len(run) - 1))
    return features


def _support(claim: str, evidence_text: str) -> tuple[bool, float, str]:
    clean = _normalize(strip_citations(claim)).strip(" -*•:：")
    source = _normalize(evidence_text)
    if not clean or not source:
        return False, 0.0, "empty_claim_or_evidence"
    if clean in source:
        return True, 1.0, "verbatim"

    claim_urls = {
        _normalize(value).rstrip(".,;:!?。；：！？")
        for value in _URL.findall(clean)
    }
    source_urls = {
        _normalize(value).rstrip(".,;:!?。；：！？")
        for value in _URL.findall(source)
    }
    if claim_urls - source_urls:
        return False, 0.0, "url_mismatch"

    claim_numbers = {_canonical_number(value) for value in _NUMBER.findall(clean)}
    source_numbers = {_canonical_number(value) for value in _NUMBER.findall(source)}
    if claim_numbers - source_numbers:
        return False, 0.0, "number_mismatch"

    # Structured update records are commonly English identifiers plus an ISO
    # timestamp, while the answer surrounding them may be Chinese. Exact
    # agreement on every number and every ASCII identifier is stronger and
    # more language-independent than bag-of-words overlap in this case.
    claim_identifiers = {
        value
        for value in _WORD.findall(clean)
        if any(char.isalpha() for char in value)
        and value not in _STOPWORDS
        and len(value) >= 3
    }
    source_identifiers = set(_WORD.findall(source))
    if (
        claim_numbers
        and claim_identifiers
        and claim_identifiers <= source_identifiers
    ):
        return True, 1.0, "structured_exact"

    wanted = Counter(_features(clean))
    available = Counter(_features(source))
    if not wanted:
        return False, 0.0, "no_factual_features"
    overlap = sum((wanted & available).values())
    recall = overlap / sum(wanted.values())
    unique_recall = len(set(wanted) & set(available)) / len(set(wanted))
    score = round(min(recall, unique_recall), 6)
    # Short assertions must be fully present. Longer paraphrases may contain
    # grammar words absent from a passage, so require all numbers plus a clear
    # majority of the non-stopword lexical features.
    threshold = 1.0 if len(wanted) <= 3 else 0.55
    if score >= threshold:
        return True, score, "lexical_recall"
    return False, score, "insufficient_lexical_support"


def verify_answer_claims(
    answer: str,
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return benchmark-schema claims judged only against cited Evidence IDs."""

    evidence_by_id = {
        str(item.get("id") or ""): item
        for item in evidence
        if str(item.get("id") or "")
    }
    claims: list[dict[str, Any]] = []
    for unit in claim_units(answer):
        text = strip_citations(unit).strip()
        if not text or not _FACTUAL_TEXT.search(text) or not _features(text):
            continue
        citations = [
            evidence_id
            for evidence_id in extract_citation_ids(unit)
            if evidence_id in evidence_by_id
        ]
        source = "\n".join(
            " ".join(
                (
                    str(evidence_by_id[evidence_id].get("title") or ""),
                    str(evidence_by_id[evidence_id].get("content") or ""),
                    str(evidence_by_id[evidence_id].get("uri") or ""),
                    str(evidence_by_id[evidence_id].get("published_at") or ""),
                )
            )
            for evidence_id in citations
        )
        supported, score, reason = _support(text, source)
        claims.append(
            {
                "text": text,
                "feature_count": len(_features(text)),
                "claim_shape": (
                    "heading"
                    if text.rstrip().endswith((":", "："))
                    else (
                        "sentence"
                        if text.rstrip().endswith(
                            ("。", "！", "？", ".", "!", "?")
                        )
                        else "fragment"
                    )
                ),
                "citations": citations,
                "requires_citation": True,
                "supported": bool(citations) and supported,
                "verifier": VERIFIER_VERSION,
                "support_score": score if citations else 0.0,
                "support_reason": reason if citations else "missing_valid_citation",
            }
        )
    return claims
