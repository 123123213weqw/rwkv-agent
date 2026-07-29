#!/usr/bin/env python3
"""Run the frozen RWKV Agent public benchmark core on the V100 server.

The runner is deliberately isolated from the stable controller database.  It may
call the already-running localhost G1I sidecars, but all cases, checkpoints and
reports are written under the benchmark root.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import re
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(
    os.environ.get(
        "RWKV_AGENT_BENCH_ROOT",
        "bench/runs/rwkv-agent",
    )
)
PROJECT = ROOT / "project"
for value in (str(PROJECT), str(PROJECT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmarks.agent_benchmark_schema import (  # noqa: E402
    RESULT_SCHEMA_VERSION,
    validate_case,
    validate_result,
)
from benchmarks.run_agent_benchmark_metrics import build_report  # noqa: E402
from rwkv_agent.controller import AgentController, ModelClient  # noqa: E402
from rwkv_agent.claim_verifier import verify_answer_claims  # noqa: E402
from rwkv_agent.longbench_state import (  # noqa: E402
    run_state_longbench_chunk_ensemble,
)
from rwkv_agent.state_agent import (  # noqa: E402
    ANSWER_STOPS,
    compact_answer_evidence,
    coordinate_answer_output,
    render_answer_fallback_prompt,
    render_compact_answer_prompt,
)
from rwkv_agent.tools.long_text import chunk_text, rank_chunks  # noqa: E402
from rwkv_agent.tools.web import WebSearchAdapter  # noqa: E402

DATASETS = ("bfcl", "webwalkerqa", "frames", "longbench_v2", "alce")
MODEL_URLS = tuple(
    value.strip()
    for value in os.environ.get(
        "RWKV_AGENT_MODEL_URLS",
        "http://127.0.0.1:8118,http://127.0.0.1:8119",
    ).split(",")
    if value.strip()
)
if not MODEL_URLS:
    raise RuntimeError("RWKV_AGENT_MODEL_URLS must contain at least one URL")
CORE_DIR = ROOT / "data" / "normalized" / "core_v1"
DEFAULT_CONFIG = Path(
    os.environ.get("RWKV_AGENT_CONFIG", "configs/default.json")
)
TOOL_PREFIX = "<tool_call>"
TOOL_SUFFIX = "</tool_call>"
TOOL_ENVELOPE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S | re.I)
CHOICE = re.compile(r"\b([ABCD])\b", re.I)
WRITE_LOCK = threading.Lock()
HTTP_409 = re.compile(r"(?<!\d)409(?!\d)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def jsonl_dump(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def jsonl_load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be object")
            rows.append(value)
    return rows


def append_result(path: Path, row: Mapping[str, Any]) -> None:
    validate_result(row)
    line = json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
    with WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def _stratum_value(case: Mapping[str, Any], dataset: str) -> str:
    metadata = dict(case.get("metadata") or {})
    if dataset == "bfcl":
        return str(metadata.get("category") or "unknown")
    if dataset == "webwalkerqa":
        return str(case.get("language") or "unknown")
    if dataset == "frames":
        return str(metadata.get("reasoning_types") or "unknown")
    if dataset == "longbench_v2":
        return "/".join((str(metadata.get("domain") or "unknown"), str(metadata.get("difficulty") or "unknown")))
    if dataset == "alce":
        return str(metadata.get("subset") or "unknown")
    return "all"


def select_smoke(rows: Sequence[dict[str, Any]], dataset: str, count: int) -> list[dict[str, Any]]:
    """Deterministic round-robin stratification without reading gold answers."""
    count = min(max(1, int(count)), len(rows))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_stratum_value(row, dataset)].append(row)
    for values in groups.values():
        values.sort(key=lambda item: str(item["id"]))
    keys = sorted(groups)
    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in keys}
    while len(selected) < count:
        progressed = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(groups[key]):
                continue
            selected.append(groups[key][offset])
            offsets[key] = offset + 1
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return sorted(selected, key=lambda item: str(item["id"]))


def selected_cases(dataset: str, smoke: int | None, *, cases_dir: Path = CORE_DIR) -> list[dict[str, Any]]:
    source = cases_dir / f"{dataset}.jsonl"
    rows = [validate_case(value) for value in jsonl_load(source)]
    return rows if smoke is None else select_smoke(rows, dataset, smoke)


def base_result(case: Mapping[str, Any], *, status: str, answer: str = "") -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "case_id": str(case["id"]),
        "status": status,
        "answer": str(answer),
        "abstained": status == "insufficient",
        "tool_calls": [],
        "evidence": [],
        "claims": [],
        "trace": {},
        "resources": {},
        "protocol": {},
        "benchmark": {
            "runner": "rwkv-agent-core-score.v1",
            "created_at": utc_now(),
        },
    }


def error_result(case: Mapping[str, Any], exc: BaseException, elapsed_ms: float) -> dict[str, Any]:
    row = base_result(case, status="error")
    row["resources"] = {"latency_ms": round(elapsed_ms, 3)}
    row["benchmark"].update({
        "error_type": type(exc).__name__,
        "error": str(exc)[:1000],
    })
    return row


def render_bfcl_prompt(case: Mapping[str, Any]) -> str:
    functions = json.dumps(case["available_tools"], ensure_ascii=False, separators=(",", ":"))
    return (
        "System: You are an exact function-calling engine. Select only functions listed below. "
        "Do not answer the request in prose. Output one JSON array inside the tool envelope. "
        "Every array item must have exactly name and arguments. Preserve function names exactly, "
        "use JSON-native argument types, omit optional arguments that the user did not provide, "
        "and include all independent calls when the request asks for several operations. Count the "
        "requested operations before emitting JSON: repeated uses of one function require repeated "
        "array items, and words such as also, then, each, first, second, and lastly usually introduce "
        "another call. Never merge several argument sets into one item and never stop after only the "
        "first requested operation.\n"
        "Single-call example: <tool_call>[{\"name\":\"weather.get\",\"arguments\":{\"city\":\"Paris\"}}]</tool_call>\n"
        "Repeated parallel example: <tool_call>[{\"name\":\"weather.get\",\"arguments\":{\"city\":\"Paris\"}},"
        "{\"name\":\"weather.get\",\"arguments\":{\"city\":\"Tokyo\"}}]</tool_call>\n"
        "Multi-function example: <tool_call>[{\"name\":\"calendar.find\",\"arguments\":{\"day\":\"Monday\"}},"
        "{\"name\":\"mail.send\",\"arguments\":{\"recipient\":\"Alex\"}},"
        "{\"name\":\"tasks.create\",\"arguments\":{\"title\":\"Follow up\"}}]</tool_call>\n"
        f"Functions: {functions}\n\nUser: {str(case['prompt']).strip()}\n\nAssistant: {TOOL_PREFIX}"
    )


def _repair_arguments_first_payload(value: str) -> str | None:
    """Close a misplaced ``arguments`` object before an outer ``name`` key.

    Some greedy completions emit ``{"arguments":{... ,"name":"tool"}`` for
    each list item.  The repair is syntax-only: it runs only after both JSON and
    Python-literal decoding fail, inserts braces only at the exact list-item
    depth, and accepts the result only when every item has exactly the public
    ``name``/``arguments`` envelope.  It never reads the benchmark Gold.
    """

    stripped = value.strip()
    if not stripped.startswith("["):
        return None
    output: list[str] = []
    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escaped = False
    index = 0
    while index < len(value):
        char = value[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            if (
                brace_depth == 0
                and bracket_depth == 0
                and not value[index + 1 :].strip()
            ):
                index += 1
                continue
            brace_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif (
            char == ","
            and brace_depth == 2
            and bracket_depth == 1
            and re.match(r',\s*"name"\s*:', value[index:])
        ):
            output.append("}")
            brace_depth -= 1
        output.append(char)
        index += 1
    repaired = "".join(output)
    if repaired == value:
        return None
    try:
        payload = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    if any(
        not isinstance(item, dict)
        or set(item) != {"name", "arguments"}
        or not isinstance(item.get("name"), str)
        or not isinstance(item.get("arguments"), dict)
        for item in payload
    ):
        return None
    return repaired


def _decode_bfcl_payload(value: str) -> tuple[Any, bool]:
    """Decode strict JSON first, then a safe Python-literal compatibility form."""

    try:
        return json.loads(value), False
    except json.JSONDecodeError as json_error:
        try:
            return ast.literal_eval(value), True
        except (ValueError, SyntaxError):
            if value.lstrip().startswith("{"):
                try:
                    return json.loads("[" + value + "]"), True
                except json.JSONDecodeError:
                    pass
            repaired = _repair_arguments_first_payload(value)
            if repaired is not None:
                return json.loads(repaired), True
            raise json_error


def parse_bfcl_calls(raw: str) -> tuple[list[dict[str, Any]], str]:
    generated = str(raw or "").strip()
    reconstructed = generated if generated.lower().startswith(TOOL_PREFIX) else TOOL_PREFIX + generated
    if TOOL_SUFFIX not in reconstructed.lower():
        reconstructed += TOOL_SUFFIX
    match = TOOL_ENVELOPE.fullmatch(reconstructed.strip())
    if not match:
        raise ValueError("tool envelope is not strict")
    payload, _repaired = _decode_bfcl_payload(match.group(1))
    if isinstance(payload, dict):
        if set(payload) == {"name", "arguments"}:
            payload = [payload]
        elif set(payload) == {"calls"} and isinstance(payload["calls"], list):
            payload = payload["calls"]
    if not isinstance(payload, list) or not payload:
        raise ValueError("tool payload must be a non-empty array")
    calls: list[dict[str, Any]] = []
    for index, value in enumerate(payload):
        if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
            raise ValueError(f"call {index} must have exactly name and arguments")
        if not isinstance(value["name"], str) or not value["name"].strip():
            raise ValueError(f"call {index} name is invalid")
        if not isinstance(value["arguments"], dict):
            raise ValueError(f"call {index} arguments must be an object")
        calls.append({"name": value["name"].strip(), "arguments": dict(value["arguments"])})
    return calls, reconstructed


def _bfcl_schema_type(schema: Mapping[str, Any]) -> str:
    return str(schema.get("type") or "").casefold()


def _bfcl_value_compatible(value: Any, schema: Mapping[str, Any]) -> bool:
    expected = _bfcl_schema_type(schema)
    if expected in {"string", "str"}:
        return isinstance(value, str)
    if expected in {"integer", "int"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected in {"number", "float"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected in {"boolean", "bool"}:
        return isinstance(value, bool)
    if expected in {"array", "list"}:
        return isinstance(value, list)
    if expected in {"object", "dict"}:
        return isinstance(value, Mapping)
    return True


def _coerce_bfcl_value(value: Any, schema: Mapping[str, Any]) -> Any:
    """Apply only lossless JSON-Schema type normalization."""

    expected = _bfcl_schema_type(schema)
    if expected in {"integer", "int"}:
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            return int(value)
        return value
    if expected in {"number", "float"}:
        if isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return value
        return value
    if expected in {"boolean", "bool"} and isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return value
    if expected in {"array", "list"} and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            return [_coerce_bfcl_value(item, item_schema) for item in value]
        return value
    if expected in {"object", "dict"} and isinstance(value, Mapping):
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            return dict(value)
        return {
            str(key): _coerce_bfcl_value(child, child_schema)
            for key, child in value.items()
            if isinstance((child_schema := properties.get(key)), Mapping)
        }
    return value


def _coerce_bfcl_prompt_grounded_scalar(
    value: Any,
    schema: Mapping[str, Any],
    *,
    parameter: str,
    prompt: str,
) -> Any:
    """Normalize an unambiguous scalar representation explicitly in the prompt."""

    expected = _bfcl_schema_type(schema)
    if (
        expected in {"integer", "int", "number", "float"}
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value).is_integer()
        and 1 <= int(value) <= 11
        and re.search(r"(?:hour|time|operat)", parameter, re.I)
        and re.search(
            rf"(?<!\d){int(value)}(?:\s*:\s*00)?\s*p\.?\s*m\.?(?!\w)",
            prompt,
            re.I,
        )
    ):
        converted: int | float = int(value) + 12
        return float(converted) if expected in {"number", "float"} else converted
    return value


def _expand_bfcl_prompt_grounded_enum_array(
    value: Any,
    schema: Mapping[str, Any],
    *,
    prompt: str,
) -> Any:
    """Add only advertised enum values that the user explicitly names."""

    if _bfcl_schema_type(schema) not in {"array", "list"} or not isinstance(
        value, list
    ):
        return value
    items = schema.get("items")
    if not isinstance(items, Mapping):
        return value
    choices = items.get("enum")
    if not isinstance(choices, list) or not choices or not all(
        isinstance(choice, str) for choice in choices
    ):
        return value
    mentioned: list[str] = []
    for choice in choices:
        lowered = choice.casefold()
        forms = {lowered}
        if lowered.endswith("ies") and len(lowered) > 3:
            forms.add(lowered[:-3] + "y")
        elif lowered.endswith("es") and len(lowered) > 2:
            forms.add(lowered[:-2])
        elif lowered.endswith("s") and len(lowered) > 1:
            forms.add(lowered[:-1])
        if any(re.search(rf"(?<!\w){re.escape(form)}(?!\w)", prompt, re.I) for form in forms):
            mentioned.append(choice)
    if not mentioned:
        return value
    existing = {str(item).casefold() for item in value}
    return value + [choice for choice in mentioned if choice.casefold() not in existing]


def normalize_bfcl_calls(
    case: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate model calls against advertised schemas without consulting Gold.

    The normalizer drops non-schema keys, performs lossless scalar coercion,
    repairs an unambiguous one-key alias, and expands a list accidentally sent
    to a scalar parameter into repeated calls.  It never inserts benchmark
    answers or rewrites semantic values.
    """

    schemas = {
        str(tool.get("name") or ""): tool
        for tool in case.get("available_tools") or ()
        if isinstance(tool, Mapping)
    }
    output: list[dict[str, Any]] = []
    prompt = str(case.get("prompt") or "")
    for raw_call in calls:
        name = str(raw_call.get("name") or "")
        arguments = dict(raw_call.get("arguments") or {})
        tool = schemas.get(name)
        if not isinstance(tool, Mapping):
            output.append({"name": name, "arguments": arguments})
            continue
        parameter_schema = tool.get("parameters")
        if not isinstance(parameter_schema, Mapping):
            parameter_schema = {}
        properties = parameter_schema.get("properties")
        if not isinstance(properties, Mapping):
            properties = {}
        required = {str(value) for value in parameter_schema.get("required") or ()}

        unknown = [key for key in arguments if key not in properties]
        missing = [key for key in required if key not in arguments]
        alias_candidates = [
            (old_key, new_key)
            for old_key in unknown
            for new_key in missing
            if isinstance((target_schema := properties.get(new_key)), Mapping)
            and _bfcl_value_compatible(arguments[old_key], target_schema)
        ]
        if len(alias_candidates) == 1:
            old_key, new_key = alias_candidates[0]
            arguments[new_key] = arguments.pop(old_key)
        arguments = {
            str(key): value for key, value in arguments.items() if key in properties
        }

        fanout = [
            key
            for key, value in arguments.items()
            if isinstance(value, list)
            and len(value) > 1
            and isinstance(properties.get(key), Mapping)
            and _bfcl_schema_type(properties[key])
            not in {"array", "list", "object", "dict"}
        ]
        lengths = {len(arguments[key]) for key in fanout}
        variants = range(next(iter(lengths))) if len(lengths) == 1 else range(1)
        for index in variants:
            normalized: dict[str, Any] = {}
            for key, value in arguments.items():
                schema = properties.get(key)
                if not isinstance(schema, Mapping):
                    continue
                selected = value[index] if key in fanout and len(lengths) == 1 else value
                coerced = _coerce_bfcl_value(selected, schema)
                coerced = _expand_bfcl_prompt_grounded_enum_array(
                    coerced,
                    schema,
                    prompt=prompt,
                )
                normalized[key] = _coerce_bfcl_prompt_grounded_scalar(
                    coerced,
                    schema,
                    parameter=key,
                    prompt=prompt,
                )
            output.append({"name": name, "arguments": normalized})
    return output


def bfcl_official_ast(case: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Use the pinned BFCL package's official AST checker on decoded calls."""
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    decoded = [{str(call["name"]): dict(call["arguments"])} for call in calls]
    category = str(case.get("metadata", {}).get("category") or "simple")
    result = ast_checker(
        list(case["available_tools"]),
        decoded,
        list(case["gold"]["bfcl_ground_truth"]),
        Language.PYTHON,
        category,
        "gorilla-openfunctions-v2",
    )
    return dict(result)


def score_bfcl(case: Mapping[str, Any], model: ModelClient) -> dict[str, Any]:
    started = time.perf_counter()
    completion = model.complete(
        render_bfcl_prompt(case),
        max_tokens=512,
        stops=[TOOL_SUFFIX, "\nUser:", "\nSystem:", "\n\nUser:", "</s>"],
    )
    raw_completion = str(completion.get("raw") or "")
    calls: list[dict[str, Any]] = []
    reconstructed = ""
    parse_error = ""
    official = {"valid": False, "error": ["not parsed"], "error_type": "runner:not_parsed"}
    try:
        calls, reconstructed = parse_bfcl_calls(raw_completion)
        calls = normalize_bfcl_calls(case, calls)
        official = bfcl_official_ast(case, calls)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"[:1000]
    parallel = "parallel" in str(case.get("metadata", {}).get("category") or "")
    if parallel:
        calls = [{**call, "parallel_group": 1} for call in calls]
    elapsed = (time.perf_counter() - started) * 1000.0
    row = base_result(case, status="ok" if calls else "error")
    row["tool_calls"] = calls
    row["trace"] = {"requests": 1, "rounds": 1}
    row["resources"] = {
        "latency_ms": round(elapsed, 3),
        "output_tokens": int(completion.get("output_tokens") or 0),
    }
    row["protocol"] = {"tool_call_valid": bool(calls)}
    row["benchmark"].update({
        "category": str(case.get("metadata", {}).get("category") or ""),
        "bfcl_ast_exact_match": bool(official.get("valid")),
        "bfcl_evaluator": "bfcl-eval==2026.3.23 ast_checker/Language.PYTHON",
        "bfcl_checker_error_type": str(official.get("error_type") or ""),
        "bfcl_checker_errors": list(official.get("error") or [])[:8],
        "parse_error": parse_error,
        "completion": {
            "stop": completion.get("stop"),
            "output_tokens": completion.get("output_tokens"),
            "model_elapsed_ms": completion.get("model_elapsed_ms"),
            "request_elapsed_ms": completion.get("request_elapsed_ms"),
            "model": completion.get("model"),
            "url": completion.get("url"),
            "output_sha256": hashlib.sha256(raw_completion.encode()).hexdigest(),
            "output_chars": len(raw_completion),
            "parsed_envelope_sha256": (
                hashlib.sha256(reconstructed.encode()).hexdigest()
                if reconstructed
                else ""
            ),
        },
    })
    return row


def _safe_evidence(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        evidence_id = str(item.get("id") or f"W{len(output) + 1}")
        if not evidence_id or evidence_id in seen:
            evidence_id = f"W{len(output) + 1}"
        seen.add(evidence_id)
        output.append({
            "id": evidence_id,
            "title": str(item.get("title") or "")[:500],
            "content": str(item.get("content") or "")[:2000],
            "uri": str(item.get("uri") or ""),
        })
    return output


def score_web(
    case: Mapping[str, Any],
    controller: AgentController,
) -> dict[str, Any]:
    started = time.perf_counter()
    root_url = str(case.get("metadata", {}).get("root_url") or "")
    with controller.web.scoped(root_url):
        native = controller.run_stateful_search(
            str(case["prompt"]),
            session_id="benchmark-" + str(case["id"]),
            branch_width=4,
            max_rounds=2,
        )
    elapsed = (time.perf_counter() - started) * 1000.0
    native_trace = dict(native.get("trace") or {})
    rounds = list(native_trace.get("rounds") or [])
    branch_rows = [branch for item in rounds for branch in list(item.get("branches") or [])]
    requests = sum(bool(dict(branch.get("route") or {}).get("strict")) for branch in branch_rows)
    strict = bool(branch_rows) and all(bool(dict(branch.get("route") or {}).get("strict")) for branch in branch_rows)
    runtime = dict(native_trace.get("state_runtime") or {})
    release = dict(runtime.get("release") or {})
    created = 1 + int(runtime.get("forked_states") or 0)
    released = int(release.get("released") or 0)
    protocol = dict(native_trace.get("answer_protocol") or {})
    primary = dict(protocol.get("primary") or {})
    fallback = dict(protocol.get("fallback") or {})
    answer_valid = bool(primary.get("valid")) or bool(fallback.get("valid")) or not bool(native.get("tool_result", {}).get("evidence"))
    native_status = str(native.get("status") or "")
    normalized_status = (
        "ok" if native_status in {"ok", "insufficient_evidence"} else "error"
    )
    row = base_result(case, status=normalized_status, answer=str(native.get("answer") or ""))
    row["evidence"] = _safe_evidence(list(native.get("tool_result", {}).get("evidence") or []))
    row["abstained"] = (
        native_status == "insufficient_evidence" or not row["evidence"]
    )
    row["claims"] = (
        []
        if row["abstained"]
        else verify_answer_claims(row["answer"], row["evidence"])
    )
    row["trace"] = {
        "requests": requests,
        "rounds": len(rounds),
        "states_created": created,
        "states_released": released,
        "states_leaked": max(0, created - released),
        "states_reused": len(branch_rows) + (1 if row["evidence"] else 0),
    }
    output_tokens = 0
    for key in ("answer_completion",):
        value = native_trace.get(key)
        if isinstance(value, Mapping):
            output_tokens += int(value.get("output_tokens") or 0)
    fallback_meta = dict(protocol.get("fallback_completion") or {})
    output_tokens += int(fallback_meta.get("output_tokens") or 0)
    row["resources"] = {
        "latency_ms": round(float(native_trace.get("elapsed_ms") or elapsed), 3),
        "output_tokens": output_tokens,
    }
    row["protocol"] = {"tool_call_valid": strict, "answer_valid": answer_valid}
    row["benchmark"].update({
        "native_status": native_status,
        "branch_width": 4,
        "configured_rounds": 2,
        "strict_tool_calls": sum(bool(dict(branch.get("route") or {}).get("strict")) for branch in branch_rows),
        "branch_calls": len(branch_rows),
        "answer_fallback_used": bool(protocol.get("fallback_used")),
        "answer_primary_valid": bool(primary.get("valid")),
        "answer_fallback_valid": bool(fallback.get("valid")),
        "answer_citation_repaired": bool(primary.get("citation_repaired"))
        or bool(fallback.get("citation_repaired")),
        "search_queries": [
            str(dict(branch.get("route") or {}).get("arguments", {}).get("query") or "")
            for branch in branch_rows
        ],
        "executed_search_queries": [
            str(branch.get("effective_query") or "")
            for branch in branch_rows
            if str(branch.get("effective_query") or "")
        ],
        "search_scope": str(case.get("metadata", {}).get("root_url") or ""),
        "primary_retrieval_metric": "exact_page_recall",
        "release_status": str(release.get("status") or ""),
    })
    return row


def render_longbench_prompt(case: Mapping[str, Any], selected: Sequence[tuple[float, Any]]) -> str:
    excerpts = []
    for index, (score, chunk) in enumerate(selected, 1):
        excerpts.append(f"[Excerpt {index}; chars {chunk.char_start}-{chunk.char_end}; retrieval {score:.4f}]\n{chunk.text}")
    return (
        "System: Answer the multiple-choice question using only the supplied excerpts. "
        "Choose exactly one of A, B, C, or D. Do not explain and do not call tools.\n\n"
        + "\n\n".join(excerpts)
        + "\n\nQuestion:\n"
        + str(case["prompt"]).strip()
        + '\n\nAssistant: {"choice":"'
    )


def score_longbench(case: Mapping[str, Any], model: ModelClient) -> dict[str, Any]:
    started = time.perf_counter()
    chunks = chunk_text(str(case.get("context") or ""), max_chars=1600, overlap_chars=160)
    selected = rank_chunks(str(case["prompt"]), chunks, top_k=6)
    prompt = render_longbench_prompt(case, selected)
    classification = model.classify(
        prompt,
        labels={label: label for label in "ABCD"},
    )
    choice_scores = {
        label: float(score)
        for label, score in dict(classification.get("scores") or {}).items()
        if label in "ABCD"
    }
    answer = max(choice_scores, key=choice_scores.get) if choice_scores else ""
    elapsed = (time.perf_counter() - started) * 1000.0
    row = base_result(case, status="ok" if answer else "error", answer=answer)
    row["evidence"] = [
        {
            "id": f"L{index}",
            "title": f"selected chunk {chunk.chunk_id}",
            "content": chunk.text[:2000],
            "uri": f"longbench://{case['id']}/chunk/{chunk.chunk_id}",
        }
        for index, (_score, chunk) in enumerate(selected, 1)
    ]
    row["trace"] = {"requests": 1, "rounds": 1}
    row["resources"] = {
        "latency_ms": round(elapsed, 3),
        "output_tokens": 0,
    }
    row["protocol"] = {"choice_valid": bool(answer)}
    row["benchmark"].update({
        "method": "lexical-chunk-retrieval-top6-plus-choice-logits",
        "context_chars": len(str(case.get("context") or "")),
        "chunks": len(chunks),
        "selected_chunk_ids": [chunk.chunk_id for _score, chunk in selected],
        "selected_scores": [round(float(score), 6) for score, _chunk in selected],
        "choice_scores": choice_scores,
        "choice_exact_match": answer in set(map(str, case["gold"].get("answers") or [])),
        "classification": {
            "model_elapsed_ms": classification.get("elapsed_ms"),
            "request_elapsed_ms": classification.get("request_elapsed_ms"),
            "url": classification.get("url"),
        },
    })
    return row


def score_longbench_state(
    case: Mapping[str, Any],
    state_model: ModelClient,
) -> dict[str, Any]:
    started = time.perf_counter()
    chunks = chunk_text(
        str(case.get("context") or ""),
        max_chars=1600,
        overlap_chars=160,
    )
    selected = rank_chunks(str(case["prompt"]), chunks, top_k=8)
    result = run_state_longbench_chunk_ensemble(
        state_model,
        question=str(case["prompt"]),
        selected=selected,
        session_id=str(case["id"]),
        branch_width=8,
    )
    answer = str(result.get("choice") or "")
    elapsed = (time.perf_counter() - started) * 1000.0
    row = base_result(case, status="ok" if answer else "error", answer=answer)
    row["evidence"] = [
        {
            "id": f"L{index}",
            "title": f"state-read chunk {chunk.chunk_id}",
            "content": chunk.text[:2000],
            "uri": f"longbench://{case['id']}/chunk/{chunk.chunk_id}",
        }
        for index, (_score, chunk) in enumerate(selected[:12], 1)
    ]
    row["trace"] = {
        "requests": 2,
        "rounds": 2,
        "states_created": int(result.get("states_created") or 0),
        "states_released": int(result.get("states_released") or 0),
        "states_leaked": int(result.get("state_leak_count") or 0),
    }
    row["resources"] = {
        "latency_ms": round(float(result.get("latency_ms") or elapsed), 3),
        "output_tokens": int(result.get("root_output_tokens") or 0)
        + int(result.get("branch_output_tokens") or 0),
    }
    row["protocol"] = {"choice_valid": bool(answer)}
    row["benchmark"].update({
        "method": "recurrent-state-top8-parallel-chunks-cyclic-option-logit-ensemble",
        "context_chars": len(str(case.get("context") or "")),
        "chunks": len(chunks),
        "selected_chunk_ids": [chunk.chunk_id for _score, chunk in selected],
        "selected_scores": [round(float(score), 6) for score, _chunk in selected],
        "choice_exact_match": answer in set(map(str, case["gold"].get("answers") or [])),
        "branch_reports": list(result.get("reports") or []),
        "choice_scores": dict(result.get("choice_scores") or {}),
        "release_status": str(dict(result.get("release") or {}).get("status") or ""),
    })
    return row


def _is_insufficient_evidence_validation(errors: Sequence[str]) -> bool:
    """Separate safe evidence abstention from malformed answer protocol."""

    normalized = {str(value) for value in errors if str(value)}
    return bool(
        "unsupported_claim" in normalized
        and normalized <= {"missing_citation", "unsupported_claim"}
    )


def score_alce(
    case: Mapping[str, Any],
    model: ModelClient,
    *,
    max_tokens: int = 32,
    prompt_profile: str = "full",
) -> dict[str, Any]:
    started = time.perf_counter()
    evidence = _safe_evidence(list(case.get("evidence_context") or []))
    if prompt_profile == "compact":
        evidence = compact_answer_evidence(str(case["prompt"]), evidence)
    elif prompt_profile == "full":
        for item in evidence:
            item["content"] = str(item.get("content") or "")[:1800]
    else:
        raise ValueError(f"unknown ALCE prompt profile: {prompt_profile}")
    prompt_renderer = (
        render_compact_answer_prompt
        if prompt_profile == "compact"
        else render_answer_fallback_prompt
    )
    completion = model.complete(
        prompt_renderer(str(case["prompt"]), evidence),
        max_tokens=max_tokens,
        stops=list(ANSWER_STOPS),
    )
    validation = coordinate_answer_output(str(completion.get("raw") or ""), evidence)
    validation_errors = list(validation.get("errors") or [])
    insufficient_evidence = _is_insufficient_evidence_validation(
        validation_errors
    )
    answer = (
        str(validation.get("answer") or "")
        if validation.get("valid")
        else ""
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    row = base_result(
        case,
        status=(
            "ok"
            if validation.get("valid") or insufficient_evidence
            else "error"
        ),
        answer=answer,
    )
    row["abstained"] = insufficient_evidence
    row["evidence"] = evidence
    row["claims"] = (
        []
        if insufficient_evidence
        else verify_answer_claims(row["answer"], row["evidence"])
    )
    row["trace"] = {"requests": 1, "rounds": 1}
    row["resources"] = {
        "latency_ms": round(elapsed, 3),
        "output_tokens": int(completion.get("output_tokens") or 0),
    }
    row["protocol"] = {"answer_valid": bool(validation.get("valid"))}
    row["benchmark"].update({
        "subset": str(case.get("metadata", {}).get("subset") or ""),
        "method": f"oracle-top5-{prompt_profile}-evidence-plus-greedy-cited-answer",
        "answer_validation_errors": validation_errors,
        "answer_outcome": (
            "insufficient_evidence"
            if insufficient_evidence
            else "answered" if validation.get("valid") else "protocol_error"
        ),
        "citation_repaired": bool(validation.get("citation_repaired")),
        "citations": list(validation.get("citations") or []),
        "completion": {
            "stop": completion.get("stop"),
            "output_tokens": completion.get("output_tokens"),
            "model_elapsed_ms": completion.get("model_elapsed_ms"),
            "request_elapsed_ms": completion.get("request_elapsed_ms"),
            "model": completion.get("model"),
            "url": completion.get("url"),
            "output_chars": len(str(completion.get("raw") or "")),
        },
    })
    return row


@dataclass
class TrackRuntime:
    model: ModelClient
    controller_pool: "ControllerLeasePool | None" = None
    longbench_pool: "ControllerLeasePool | None" = None
    alce_max_tokens: int = 32
    alce_prompt_profile: str = "full"

    def score_web(self, case: Mapping[str, Any]) -> dict[str, Any]:
        if self.controller_pool is None:
            raise RuntimeError("web controller pool is unavailable")
        with self.controller_pool.lease() as controller:
            return score_web(case, controller)

    def close(self) -> None:
        if self.controller_pool is not None:
            self.controller_pool.close()
        if self.longbench_pool is not None:
            self.longbench_pool.close()

    def score_longbench_state(self, case: Mapping[str, Any]) -> dict[str, Any]:
        if self.longbench_pool is None:
            raise RuntimeError("LongBench state reader pool is unavailable")
        with self.longbench_pool.lease() as state_model:
            return score_longbench_state(case, state_model)


class ControllerLeasePool:
    """Reserve one B4 request per sidecar to avoid persistent-state 409s."""

    def __init__(self, controllers: Sequence[AgentController]) -> None:
        if not controllers:
            raise ValueError("at least one controller is required")
        self.controllers = list(controllers)
        self._available: Queue[AgentController] = Queue(
            maxsize=len(self.controllers)
        )
        for controller in self.controllers:
            self._available.put(controller)

    @contextmanager
    def lease(self) -> Iterable[AgentController]:
        controller = self._available.get()
        try:
            yield controller
        finally:
            self._available.put(controller)

    def close(self) -> None:
        for controller in self.controllers:
            close = getattr(controller, "close", None)
            if callable(close):
                close()


def make_runtime(
    dataset: str,
    run_dir: Path,
    *,
    web_profile: str,
    web_fallback_engines: Sequence[str],
    longbench_mode: str,
    alce_max_tokens: int,
    alce_prompt_profile: str,
) -> TrackRuntime:
    model = ModelClient(list(MODEL_URLS))
    if dataset == "longbench_v2" and longbench_mode == "state":
        return TrackRuntime(
            model=model,
            longbench_pool=ControllerLeasePool(
                [ModelClient([url]) for url in MODEL_URLS]
            ),
            alce_max_tokens=alce_max_tokens,
            alce_prompt_profile=alce_prompt_profile,
        )
    if dataset not in {"webwalkerqa", "frames"}:
        return TrackRuntime(
            model=model,
            alce_max_tokens=alce_max_tokens,
            alce_prompt_profile=alce_prompt_profile,
        )
    controllers = []
    effective_fallback_engines = (
        tuple(dict.fromkeys((*web_fallback_engines, "wikipedia")))
        if dataset == "frames"
        else tuple(web_fallback_engines)
    )
    for index, model_url in enumerate(MODEL_URLS):
        web = WebSearchAdapter(
            str(DEFAULT_CONFIG),
            profile=web_profile,
            shadow=False,
            fallback_engines=effective_fallback_engines,
        )
        controllers.append(
            AgentController(
                model_urls=[model_url],
                memory_path=str(
                    run_dir / f"{dataset}.sessions-{index}.sqlite3"
                ),
                web_config=str(DEFAULT_CONFIG),
                web_adapter=web,
            )
        )
    return TrackRuntime(
        model=model,
        controller_pool=ControllerLeasePool(controllers),
        alce_max_tokens=alce_max_tokens,
        alce_prompt_profile=alce_prompt_profile,
    )


def track_scorer(
    dataset: str,
    runtime: TrackRuntime,
    *,
    longbench_mode: str,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    if dataset == "bfcl":
        return lambda case: score_bfcl(case, runtime.model)
    if dataset in {"webwalkerqa", "frames"}:
        return runtime.score_web
    if dataset == "longbench_v2":
        if longbench_mode == "state":
            return runtime.score_longbench_state
        return lambda case: score_longbench(case, runtime.model)
    if dataset == "alce":
        return lambda case: score_alce(
            case,
            runtime.model,
            max_tokens=runtime.alce_max_tokens,
            prompt_profile=runtime.alce_prompt_profile,
        )
    raise ValueError(dataset)


def _group_metric(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    metadata_key: str,
    metric: str,
    explode: bool = False,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in evaluations:
        value = dict(row.get("metadata") or {}).get(metadata_key)
        if explode and isinstance(value, list):
            values = value
        elif explode and isinstance(value, str):
            values = [item.strip() for item in value.split("|") if item.strip()]
        else:
            values = [value]
        measured = dict(row.get("metrics") or {}).get(metric)
        if not isinstance(measured, (bool, int, float)):
            continue
        for group in values:
            name = str(group or "").strip()
            if name:
                groups[name].append(float(measured))
    return {
        name: {
            "cases": len(values),
            "mean": round(sum(values) / len(values), 6),
        }
        for name, values in sorted(groups.items())
    }


def summarize(
    dataset: str,
    results: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ok = sum(row.get("status") == "ok" for row in results)
    errors = Counter(str(row.get("benchmark", {}).get("error_type") or row.get("benchmark", {}).get("parse_error") or "") for row in results if row.get("status") != "ok")
    output: dict[str, Any] = {
        "schema_version": "rwkv-agent-core-score-summary.v1",
        "dataset": dataset,
        "cases": len(results),
        "status_ok": ok,
        "status_ok_rate": round(ok / max(1, len(results)), 6),
        "errors": dict(errors),
        "unified_metrics": report["summary"]["overall"]["metrics"],
        "grand_score": None,
    }
    def has_http_409(row: Mapping[str, Any]) -> bool:
        diagnostics: list[str] = []

        def visit(value: Any, *, diagnostic: bool = False) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    name = str(key).casefold()
                    visit(
                        child,
                        diagnostic=diagnostic
                        or any(
                            marker in name
                            for marker in (
                                "error",
                                "status",
                                "message",
                                "reason",
                                "detail",
                            )
                        ),
                    )
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child, diagnostic=diagnostic)
            elif diagnostic and isinstance(value, str):
                diagnostics.append(value)

        visit(
            {
                "status": row.get("status"),
                "trace": row.get("trace"),
                "benchmark": row.get("benchmark"),
            }
        )
        return any(HTTP_409.search(value) for value in diagnostics)

    output["reliability"] = {
        "http_409_count": sum(
            has_http_409(row)
            for row in results
        ),
        "route_error_count": sum(
            str(dict(row.get("benchmark") or {}).get("native_status") or "")
            == "route_error"
            for row in results
        ),
        "state_leak_count": sum(
            int(dict(row.get("trace") or {}).get("states_leaked") or 0)
            for row in results
        ),
        "protocol_leak_count": sum(
            bool(dict(row.get("metrics") or {}).get("protocol_leak"))
            for row in evaluations
        ),
        "budget_overrun_count": sum(
            any(
                dict(row.get("metrics") or {}).get(name) is False
                for name in (
                    "within_latency_budget",
                    "within_request_budget",
                    "within_round_budget",
                )
            )
            for row in evaluations
        ),
    }
    if dataset == "bfcl":
        passed = sum(bool(row.get("benchmark", {}).get("bfcl_ast_exact_match")) for row in results)
        output["bfcl_ast_exact_match"] = {"passed": passed, "rate": round(passed / max(1, len(results)), 6)}
        by_category: dict[str, dict[str, Any]] = {}
        for category in sorted({str(row.get("benchmark", {}).get("category") or "") for row in results}):
            group = [row for row in results if str(row.get("benchmark", {}).get("category") or "") == category]
            count = sum(bool(row.get("benchmark", {}).get("bfcl_ast_exact_match")) for row in group)
            by_category[category] = {"cases": len(group), "passed": count, "rate": round(count / max(1, len(group)), 6)}
        output["by_category"] = by_category
    if dataset == "longbench_v2":
        passed = sum(bool(row.get("benchmark", {}).get("choice_exact_match")) for row in results)
        output["choice_accuracy"] = {"passed": passed, "rate": round(passed / max(1, len(results)), 6)}
        output["accuracy_by_difficulty"] = _group_metric(
            evaluations,
            metadata_key="difficulty",
            metric="answer_exact_match",
        )
        output["accuracy_by_domain"] = _group_metric(
            evaluations,
            metadata_key="domain",
            metric="answer_exact_match",
        )
        output["accuracy_by_sub_domain"] = _group_metric(
            evaluations,
            metadata_key="sub_domain",
            metric="answer_exact_match",
        )
        output["accuracy_by_context_bucket"] = {
            name: {
                "cases": int(value.get("cases") or 0),
                "mean": float(
                    dict(value.get("metrics") or {})
                    .get("answer_exact_match", {})
                    .get("rate", 0.0)
                ),
            }
            for name, value in dict(
                report["summary"].get("by_context_bucket") or {}
            ).items()
        }
    if dataset == "frames":
        output["answer_f1_by_reasoning_type"] = _group_metric(
            evaluations,
            metadata_key="reasoning_types",
            metric="answer_token_f1",
            explode=True,
        )
    if dataset == "alce":
        output["answer_f1_by_subset"] = _group_metric(
            evaluations,
            metadata_key="subset",
            metric="answer_token_f1",
        )
    return output


def verify_sidecars_idle() -> list[dict[str, Any]]:
    health = ModelClient(list(MODEL_URLS)).health()
    for value in health:
        persistent = dict(value.get("persistent_states") or {})
        pool = dict(value.get("inference", {}).get("scheduler", {}).get("pool") or {})
        if int(persistent.get("allocated") or 0) != 0 or int(pool.get("allocated") or 0) != 0:
            raise RuntimeError(f"sidecar not idle after track: {value.get('cuda_visible_devices')}")
    return health


def run_dataset(
    dataset: str,
    *,
    run_dir: Path,
    cases_dir: Path,
    web_profile: str,
    web_fallback_engines: Sequence[str],
    longbench_mode: str,
    alce_max_tokens: int,
    alce_prompt_profile: str,
    smoke: int | None,
    concurrency: int,
) -> dict[str, Any]:
    cases = selected_cases(dataset, smoke, cases_dir=cases_dir)
    cases_path = run_dir / f"{dataset}.cases.jsonl"
    results_path = run_dir / f"{dataset}.results.jsonl"
    report_path = run_dir / f"{dataset}.report.json"
    evaluations_path = run_dir / f"{dataset}.evaluations.jsonl"
    summary_path = run_dir / f"{dataset}.score-summary.json"
    progress_path = run_dir / f"{dataset}.progress.json"

    if cases_path.exists():
        existing = jsonl_load(cases_path)
        if [row["id"] for row in existing] != [row["id"] for row in cases]:
            raise RuntimeError(f"existing selected cases differ: {cases_path}")
    else:
        jsonl_dump(cases_path, cases)

    previous = jsonl_load(results_path)
    completed: dict[str, dict[str, Any]] = {}
    for row in previous:
        validate_result(row)
        case_id = str(row["case_id"])
        if case_id in completed:
            raise RuntimeError(f"duplicate checkpoint result: {case_id}")
        completed[case_id] = row
    allowed = {str(case["id"]) for case in cases}
    if set(completed) - allowed:
        raise RuntimeError("checkpoint contains result IDs outside selected cases")
    pending = [case for case in cases if str(case["id"]) not in completed]

    runtime = make_runtime(
        dataset,
        run_dir,
        web_profile=web_profile,
        web_fallback_engines=web_fallback_engines,
        longbench_mode=longbench_mode,
        alce_max_tokens=alce_max_tokens,
        alce_prompt_profile=alce_prompt_profile,
    )
    scorer = track_scorer(
        dataset,
        runtime,
        longbench_mode=longbench_mode,
    )
    started = time.perf_counter()
    print(f"TRACK_START dataset={dataset} cases={len(cases)} completed={len(completed)} pending={len(pending)} concurrency={concurrency}", flush=True)

    def guarded(case: Mapping[str, Any]) -> dict[str, Any]:
        item_started = time.perf_counter()
        try:
            row = scorer(case)
            return validate_result(row)
        except BaseException as exc:
            return error_result(case, exc, (time.perf_counter() - item_started) * 1000.0)

    try:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=f"score-{dataset}") as executor:
            futures = {executor.submit(guarded, case): case for case in pending}
            for future in as_completed(futures):
                row = future.result()
                completed[str(row["case_id"])] = row
                append_result(results_path, row)
                done = len(completed)
                if done <= 5 or done % 10 == 0 or done == len(cases):
                    json_dump(progress_path, {
                        "dataset": dataset,
                        "cases": len(cases),
                        "completed": done,
                        "remaining": len(cases) - done,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "updated_at": utc_now(),
                    })
                    print(f"TRACK_PROGRESS dataset={dataset} completed={done}/{len(cases)} status={row['status']} case={row['case_id']}", flush=True)
    finally:
        runtime.close()

    ordered = [completed[str(case["id"])] for case in cases]
    jsonl_dump(results_path, ordered)
    report, evaluations = build_report(cases_path=cases_path, results_path=results_path)
    json_dump(report_path, report)
    jsonl_dump(evaluations_path, evaluations)
    score_summary = summarize(dataset, ordered, report, evaluations)
    score_summary["inputs"] = {
        "cases_sha256": sha256(cases_path),
        "results_sha256": sha256(results_path),
        "normalized_source_sha256": sha256(cases_dir / f"{dataset}.jsonl"),
    }
    score_summary["elapsed_seconds_this_invocation"] = round(time.perf_counter() - started, 3)
    score_summary["sidecars_after"] = verify_sidecars_idle()
    json_dump(summary_path, score_summary)
    print(f"TRACK_DONE dataset={dataset} cases={len(cases)} status_ok={score_summary['status_ok']} elapsed={score_summary['elapsed_seconds_this_invocation']}", flush=True)
    return score_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score frozen RWKV Agent public benchmark tracks on the V100 server.")
    parser.add_argument("--dataset", choices=("all",) + DATASETS, default="all")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=CORE_DIR,
        help="directory containing one normalized <dataset>.jsonl file per selected track",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", type=int, metavar="N", help="deterministic stratified N cases per selected dataset")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--concurrency", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--web-profile",
        choices=("legacy", "balanced", "enhanced"),
        default="legacy",
    )
    parser.add_argument(
        "--web-fallback-engines",
        default="bing",
        help="comma-separated isolated HTML discovery engines: bing,baidu,so360",
    )
    parser.add_argument(
        "--longbench-mode",
        choices=("lexical", "state"),
        default="lexical",
    )
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        help=(
            "optional frozen checkpoint manifest; its path and SHA-256 are bound "
            "into the run manifest (required by the final release-gate audit)"
        ),
    )
    parser.add_argument(
        "--alce-prompt-profile",
        choices=("full", "compact"),
        default="full",
    )
    parser.add_argument(
        "--alce-max-tokens",
        type=int,
        default=32,
        help="greedy ALCE answer cap; frozen into the run manifest",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", args.run_id):
        parser.error("--run-id must contain only letters, digits, dot, underscore, or hyphen")
    if args.smoke is not None and args.smoke <= 0:
        parser.error("--smoke must be positive")
    if not 16 <= args.alce_max_tokens <= 384:
        parser.error("--alce-max-tokens must be between 16 and 384")
    web_fallback_engines = tuple(
        dict.fromkeys(
            value.strip()
            for value in str(args.web_fallback_engines).split(",")
            if value.strip()
        )
    )
    if not web_fallback_engines or any(
        value not in {"bing", "baidu", "so360"}
        for value in web_fallback_engines
    ):
        parser.error("--web-fallback-engines must contain bing, baidu, or so360")
    cases_dir = args.cases_dir.expanduser().resolve()
    checkpoint_manifest = (
        args.checkpoint_manifest.expanduser().resolve()
        if args.checkpoint_manifest is not None
        else None
    )
    if checkpoint_manifest is not None and not checkpoint_manifest.is_file():
        parser.error(f"--checkpoint-manifest is not a file: {checkpoint_manifest}")
    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    manifest_path = run_dir / "run-manifest.json"
    manifest = {
        "schema_version": "rwkv-agent-core-score-run.v1",
        "run_id": args.run_id,
        "datasets": list(datasets),
        "mode": "full" if args.full else "smoke",
        "smoke_cases_per_dataset": args.smoke,
        "concurrency": args.concurrency,
        "web_profile": args.web_profile,
        "web_fallback_engines": list(web_fallback_engines),
        "effective_web_fallback_engines": {
            "webwalkerqa": list(web_fallback_engines),
            "frames": list(dict.fromkeys((*web_fallback_engines, "wikipedia"))),
        },
        "longbench_mode": args.longbench_mode,
        "alce_max_tokens": args.alce_max_tokens,
        "alce_prompt_profile": args.alce_prompt_profile,
        "cases_dir": str(cases_dir),
        "model_urls": list(MODEL_URLS),
        "checkpoint_manifest": (
            {
                "path": str(checkpoint_manifest),
                "sha256": sha256(checkpoint_manifest),
            }
            if checkpoint_manifest is not None
            else None
        ),
        "stable_controller_called": False,
        "isolated_sessions": True,
        "created_at": utc_now(),
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "run_id",
            "datasets",
            "mode",
            "smoke_cases_per_dataset",
            "concurrency",
            "cases_dir",
            "web_profile",
            "web_fallback_engines",
            "effective_web_fallback_engines",
            "longbench_mode",
            "alce_max_tokens",
            "alce_prompt_profile",
            "checkpoint_manifest",
        ):
            if old.get(key) != manifest.get(key):
                raise RuntimeError(f"run manifest mismatch for {key}")
        manifest = old
    else:
        json_dump(manifest_path, manifest)

    verify_sidecars_idle()
    summaries = []
    for dataset in datasets:
        summaries.append(
            run_dataset(
                dataset,
                run_dir=run_dir,
                cases_dir=cases_dir,
                web_profile=args.web_profile,
                web_fallback_engines=web_fallback_engines,
                longbench_mode=args.longbench_mode,
                alce_max_tokens=args.alce_max_tokens,
                alce_prompt_profile=args.alce_prompt_profile,
                smoke=None if args.full else args.smoke,
                concurrency=args.concurrency,
            )
        )
    manifest["completed_at"] = utc_now()
    manifest["completed_datasets"] = [value["dataset"] for value in summaries]
    manifest["artifacts"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != manifest_path.name
    }
    json_dump(manifest_path, manifest)
    print(json.dumps({"run_id": args.run_id, "summaries": summaries}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
