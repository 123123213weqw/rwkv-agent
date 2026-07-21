from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


SCHEMA_VERSION = "realtime-retrieval-case.v1"
CATEGORIES = {
    "academic_paper",
    "company_filing",
    "government_policy",
    "newsroom",
    "official_docs",
    "realtime_public_info",
    "repository_release",
    "software_release",
    "statistics",
}
FRESHNESS_VALUES = {"stable", "latest", "realtime"}
SOURCE_POLICIES = {"official_required", "original_required", "primary_required"}
FORBIDDEN_RESULT_TYPES = {
    "dictionary",
    "empty_content",
    "error_page",
    "login_or_captcha",
    "search_homepage",
}
REQUIRED_FIELDS = {
    "schema_version",
    "id",
    "query",
    "language",
    "category",
    "freshness",
    "source_policy",
    "expected_domains_any",
    "target_url_patterns_any",
    "forbidden_result_types",
    "notes",
}
_ID_RE = re.compile(r"^retrieval-(zh|en)-\d{3}$")
_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)


class RetrievalCaseError(ValueError):
    pass


def _string_list(value: Any, field: str, *, nonempty: bool) -> List[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise RetrievalCaseError(f"{field} must be a{' non-empty' if nonempty else ''} list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise RetrievalCaseError(f"{field} must contain non-empty strings")
    cleaned = [item.strip() for item in value]
    if len({item.casefold() for item in cleaned}) != len(cleaned):
        raise RetrievalCaseError(f"{field} contains duplicates")
    return cleaned


def validate_case(value: Mapping[str, Any], *, location: str = "case") -> Dict[str, Any]:
    missing = REQUIRED_FIELDS - set(value)
    if missing:
        raise RetrievalCaseError(f"{location}: missing fields {sorted(missing)}")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RetrievalCaseError(f"{location}: unsupported schema_version")
    identifier = str(value.get("id") or "")
    match = _ID_RE.fullmatch(identifier)
    if not match:
        raise RetrievalCaseError(f"{location}: invalid id {identifier!r}")
    language = str(value.get("language") or "")
    if language not in {"zh", "en"} or language != match.group(1):
        raise RetrievalCaseError(f"{location}: language does not match id")
    query = str(value.get("query") or "").strip()
    if len(query) < 8 or len(query) > 500:
        raise RetrievalCaseError(f"{location}: query length is outside 8..500")
    if value.get("category") not in CATEGORIES:
        raise RetrievalCaseError(f"{location}: invalid category")
    if value.get("freshness") not in FRESHNESS_VALUES:
        raise RetrievalCaseError(f"{location}: invalid freshness")
    if value.get("source_policy") not in SOURCE_POLICIES:
        raise RetrievalCaseError(f"{location}: invalid source_policy")

    domains = _string_list(value.get("expected_domains_any"), "expected_domains_any", nonempty=True)
    for domain in domains:
        if domain != domain.casefold() or not _DOMAIN_RE.fullmatch(domain):
            raise RetrievalCaseError(f"{location}: invalid normalized domain {domain!r}")
    patterns = _string_list(
        value.get("target_url_patterns_any"), "target_url_patterns_any", nonempty=False
    )
    for pattern in patterns:
        if not pattern.startswith("/") or "://" in pattern or any(ch.isspace() for ch in pattern):
            raise RetrievalCaseError(f"{location}: invalid URL path pattern {pattern!r}")
    forbidden = _string_list(
        value.get("forbidden_result_types"), "forbidden_result_types", nonempty=False
    )
    unknown = set(forbidden) - FORBIDDEN_RESULT_TYPES
    if unknown:
        raise RetrievalCaseError(f"{location}: unknown forbidden result types {sorted(unknown)}")
    if not str(value.get("notes") or "").strip():
        raise RetrievalCaseError(f"{location}: notes must be non-empty")
    return dict(value)


def validate_cases(values: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ids = set()
    queries = set()
    for index, value in enumerate(values, 1):
        row = validate_case(value, location=f"row {index}")
        identifier = row["id"]
        query_key = " ".join(str(row["query"]).casefold().split())
        if identifier in ids:
            raise RetrievalCaseError(f"row {index}: duplicate id {identifier!r}")
        if query_key in queries:
            raise RetrievalCaseError(f"row {index}: duplicate query")
        ids.add(identifier)
        queries.add(query_key)
        rows.append(row)
    return rows


def load_cases(path: Path) -> List[Dict[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RetrievalCaseError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise RetrievalCaseError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return validate_cases(rows)
