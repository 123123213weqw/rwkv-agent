from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from urllib.parse import urlsplit

SCHEMA_VERSION = "web-extraction-case.v1"
LANGUAGES = {"zh", "en"}
PAGE_TYPES = {
    "article",
    "documentation",
    "release",
    "repository",
    "government",
    "filing",
    "paper",
    "table",
    "code",
    "json",
    "plain_text",
    "js_app",
}
CONTENT_KINDS = {"html", "pdf", "json", "markdown", "plain_text"}
STATIC_OUTCOMES = {"usable", "unsupported", "js_required", "blocked"}
_ID_RE = re.compile(r"^extract-(?:zh|en)-\d{3}$")


class WebExtractionSchemaError(ValueError):
    pass


def _string_list(value: object, *, field: str, required: bool = False) -> List[str]:
    if value is None:
        values: List[str] = []
    elif isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise WebExtractionSchemaError(f"{field} must be a list")
    if required and not values:
        raise WebExtractionSchemaError(f"{field} must not be empty")
    if len(values) != len(set(values)):
        raise WebExtractionSchemaError(f"{field} contains duplicates")
    return values


def validate_case(value: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(value)
    if row.get("schema_version") != SCHEMA_VERSION:
        raise WebExtractionSchemaError("unsupported schema_version")
    case_id = str(row.get("id") or "").strip()
    if not _ID_RE.fullmatch(case_id):
        raise WebExtractionSchemaError(f"invalid id: {case_id!r}")
    language = str(row.get("language") or "").strip()
    if language not in LANGUAGES or not case_id.startswith(f"extract-{language}-"):
        raise WebExtractionSchemaError("language and id prefix do not match")
    url = str(row.get("url") or "").strip()
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise WebExtractionSchemaError("url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise WebExtractionSchemaError("url must not contain credentials")
    page_type = str(row.get("page_type") or "").strip()
    if page_type not in PAGE_TYPES:
        raise WebExtractionSchemaError(f"invalid page_type: {page_type!r}")
    content_kind = str(row.get("content_kind") or "").strip()
    if content_kind not in CONTENT_KINDS:
        raise WebExtractionSchemaError(f"invalid content_kind: {content_kind!r}")
    expected_static_outcome = str(row.get("expected_static_outcome") or "").strip()
    if expected_static_outcome not in STATIC_OUTCOMES:
        raise WebExtractionSchemaError(
            f"invalid expected_static_outcome: {expected_static_outcome!r}"
        )
    title_contains_any = _string_list(
        row.get("title_contains_any"), field="title_contains_any"
    )
    content_contains_any = _string_list(
        row.get("content_contains_any"), field="content_contains_any"
    )
    forbidden_content_any = _string_list(
        row.get("forbidden_content_any"), field="forbidden_content_any"
    )
    table_text_any = _string_list(row.get("table_text_any"), field="table_text_any")
    code_text_any = _string_list(row.get("code_text_any"), field="code_text_any")
    min_text_chars = int(row.get("min_text_chars", 80))
    if min_text_chars < 0 or min_text_chars > 1_000_000:
        raise WebExtractionSchemaError("min_text_chars is out of range")
    require_title = bool(row.get("require_title", content_kind == "html"))
    if expected_static_outcome == "usable":
        if require_title and not title_contains_any:
            raise WebExtractionSchemaError(
                "usable cases that require a title need title_contains_any"
            )
        if not content_contains_any:
            raise WebExtractionSchemaError("usable cases require content_contains_any")
        if min_text_chars < 80:
            raise WebExtractionSchemaError("usable cases require min_text_chars >= 80")
    if bool(row.get("require_table")) and not table_text_any:
        raise WebExtractionSchemaError("require_table cases require table_text_any")
    if bool(row.get("require_code")) and not code_text_any:
        raise WebExtractionSchemaError("require_code cases require code_text_any")
    row.update(
        {
            "id": case_id,
            "url": url,
            "language": language,
            "page_type": page_type,
            "content_kind": content_kind,
            "expected_static_outcome": expected_static_outcome,
            "title_contains_any": title_contains_any,
            "content_contains_any": content_contains_any,
            "forbidden_content_any": forbidden_content_any,
            "table_text_any": table_text_any,
            "code_text_any": code_text_any,
            "require_author": bool(row.get("require_author", False)),
            "require_published_at": bool(row.get("require_published_at", False)),
            "require_title": require_title,
            "require_table": bool(row.get("require_table", False)),
            "require_code": bool(row.get("require_code", False)),
            "min_text_chars": min_text_chars,
            "notes": str(row.get("notes") or "").strip(),
        }
    )
    return row


def validate_cases(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_urls = set()
    for index, value in enumerate(rows, 1):
        try:
            row = validate_case(value)
        except WebExtractionSchemaError as exc:
            raise WebExtractionSchemaError(f"case {index}: {exc}") from exc
        if row["id"] in seen_ids:
            raise WebExtractionSchemaError(f"duplicate id: {row['id']}")
        if row["url"] in seen_urls:
            raise WebExtractionSchemaError(f"duplicate url: {row['url']}")
        seen_ids.add(row["id"])
        seen_urls.add(row["url"])
        output.append(row)
    if not output:
        raise WebExtractionSchemaError("dataset is empty")
    return output


def load_cases(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise WebExtractionSchemaError(
                f"line {line_number}: invalid JSON: {exc.msg}"
            ) from exc
    return validate_cases(rows)
