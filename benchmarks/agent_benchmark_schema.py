from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


CASE_SCHEMA_VERSION = "rwkv-agent-benchmark-case.v1"
RESULT_SCHEMA_VERSION = "rwkv-agent-benchmark-result.v1"
TRACKS = {
    "tool_protocol",
    "web_research",
    "citation_grounding",
    "long_text",
    "memory",
    "end_to_end",
}
RESULT_STATUSES = {"ok", "error", "insufficient", "timeout"}


def _require_string(value: Mapping[str, Any], key: str, *, label: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return item


def _validate_string_list(
    value: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> None:
    if key not in value:
        return
    items = value[key]
    if not isinstance(items, list) or not all(
        isinstance(item, str) and item.strip() for item in items
    ):
        raise ValueError(f"{label}.{key} must be a list of non-empty strings")


def _validate_tool_calls(value: Any, *, label: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    for index, call in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(call, Mapping):
            raise ValueError(f"{item_label} must be an object")
        _require_string(call, "name", label=item_label)
        if not isinstance(call.get("arguments"), Mapping):
            raise ValueError(f"{item_label}.arguments must be an object")
        group = call.get("parallel_group")
        if group is not None and not isinstance(group, (str, int)):
            raise ValueError(
                f"{item_label}.parallel_group must be a string or integer"
            )


def validate_case(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("benchmark case must be an object")
    if value.get("schema_version") != CASE_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark case schema_version")
    case_id = _require_string(value, "id", label="case")
    _require_string(value, "dataset", label=f"case {case_id}")
    _require_string(value, "split", label=f"case {case_id}")
    track = _require_string(value, "track", label=f"case {case_id}")
    if track not in TRACKS:
        raise ValueError(f"case {case_id}.track is unsupported: {track}")
    _require_string(value, "language", label=f"case {case_id}")
    _require_string(value, "prompt", label=f"case {case_id}")

    gold = value.get("gold")
    if not isinstance(gold, Mapping):
        raise ValueError(f"case {case_id}.gold must be an object")
    for key in ("answers", "source_uris", "evidence_ids"):
        _validate_string_list(gold, key, label=f"case {case_id}.gold")
    for key in ("answerable", "requires_citations", "should_call_tools"):
        if key in gold and not isinstance(gold[key], bool):
            raise ValueError(f"case {case_id}.gold.{key} must be boolean")
    if "tool_calls" in gold:
        _validate_tool_calls(
            gold["tool_calls"],
            label=f"case {case_id}.gold.tool_calls",
        )
    if track == "tool_protocol":
        if not isinstance(gold.get("should_call_tools"), bool):
            raise ValueError(
                f"case {case_id} tool_protocol requires gold.should_call_tools"
            )
        if "tool_calls" not in gold:
            raise ValueError(
                f"case {case_id} tool_protocol requires gold.tool_calls"
            )

    limits = value.get("limits", {})
    if not isinstance(limits, Mapping):
        raise ValueError(f"case {case_id}.limits must be an object")
    for key in ("max_rounds", "max_requests", "max_latency_ms"):
        if key in limits:
            number = limits[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f"case {case_id}.limits.{key} must be numeric")
            if number < 0:
                raise ValueError(
                    f"case {case_id}.limits.{key} must be non-negative"
                )
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"case {case_id}.metadata must be an object")
    return dict(value)


def validate_result(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("benchmark result must be an object")
    if value.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark result schema_version")
    case_id = _require_string(value, "case_id", label="result")
    status = _require_string(value, "status", label=f"result {case_id}")
    if status not in RESULT_STATUSES:
        raise ValueError(f"result {case_id}.status is unsupported: {status}")
    if not isinstance(value.get("answer", ""), str):
        raise ValueError(f"result {case_id}.answer must be a string")
    if "abstained" in value and not isinstance(value["abstained"], bool):
        raise ValueError(f"result {case_id}.abstained must be boolean")
    _validate_tool_calls(
        value.get("tool_calls", []),
        label=f"result {case_id}.tool_calls",
    )

    evidence = value.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError(f"result {case_id}.evidence must be a list")
    evidence_ids: list[str] = []
    for index, item in enumerate(evidence):
        label = f"result {case_id}.evidence[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} must be an object")
        evidence_ids.append(_require_string(item, "id", label=label))
        for key in ("uri", "gold_id"):
            if key in item and not isinstance(item[key], str):
                raise ValueError(f"{label}.{key} must be a string")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError(f"result {case_id}.evidence IDs must be unique")

    for key in ("trace", "resources", "protocol"):
        if not isinstance(value.get(key, {}), Mapping):
            raise ValueError(f"result {case_id}.{key} must be an object")

    claims = value.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError(f"result {case_id}.claims must be a list")
    for index, claim in enumerate(claims):
        label = f"result {case_id}.claims[{index}]"
        if not isinstance(claim, Mapping):
            raise ValueError(f"{label} must be an object")
        _require_string(claim, "text", label=label)
        _validate_string_list(claim, "citations", label=label)
        for key in ("requires_citation", "supported"):
            if key in claim and not isinstance(claim[key], bool):
                raise ValueError(f"{label}.{key} must be boolean")
    return dict(value)


def load_jsonl(path: str | Path, *, kind: str) -> list[dict[str, Any]]:
    source = Path(path)
    validator = validate_case if kind == "case" else validate_result
    if kind not in {"case", "result"}:
        raise ValueError("kind must be 'case' or 'result'")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
            rows.append(validator(decoded))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"{source} contains no {kind} rows")
    id_key = "id" if kind == "case" else "case_id"
    ids = [str(row[id_key]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source} contains duplicate {id_key} values")
    return rows
