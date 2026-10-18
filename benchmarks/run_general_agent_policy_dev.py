#!/usr/bin/env python3
"""Evaluate frozen procedural Policy Dev turns against an HF eval sidecar.

The dataset is the held-out procedural Dev split created independently from
GeneralAgent-Loop-v1.  This runner checks the deployed response-prefix contract,
strict envelope validity, action type, stopping behaviour, and latency without
reading any frozen GeneralAgent benchmark Gold or failure trace.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from http.client import HTTPResponse
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable
from urllib.request import Request, urlopen


SCHEMA = "rwkv-agent-policy-dev-eval.v1"
PROTOCOL_MARKERS = ("<think>", "</think>", "System:", "User:", "Assistant:", "Tool:")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: row must be an object")
        required = {"id", "trajectory_id", "family", "language", "task", "prompt", "response"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"line {line_number}: missing keys {sorted(missing)}")
        row_id = str(value["id"])
        if not row_id or row_id in ids:
            raise ValueError(f"line {line_number}: duplicate or empty id {row_id!r}")
        task = str(value["task"])
        if task not in {"initial_tool", "continue_tool", "final_answer", "budget_answer"}:
            raise ValueError(f"line {line_number}: unsupported task {task!r}")
        if not str(value["prompt"]):
            raise ValueError(f"line {line_number}: prompt must not be empty")
        ids.add(row_id)
        rows.append(value)
    if not rows:
        raise ValueError("Policy Dev dataset is empty")
    return rows


def _post_json(endpoint: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return _read_json(response)


def _read_json(response: HTTPResponse) -> dict[str, Any]:
    value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("HTTP response must be a JSON object")
    return value


def _json_tool(value: str, *, opening_supplied: bool) -> tuple[bool, dict[str, Any] | None]:
    text = value.strip()
    if opening_supplied:
        if text.startswith("<tool_call>") or not text.endswith("</tool_call>"):
            return False, None
        payload = text[: -len("</tool_call>")]
    else:
        opening, closing = "<tool_call>", "</tool_call>"
        if not text.startswith(opening) or not text.endswith(closing):
            return False, None
        payload = text[len(opening) : -len(closing)]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return False, None
    valid = bool(
        isinstance(parsed, dict)
        and set(parsed) == {"name", "arguments"}
        and parsed.get("name") == "run_command"
        and isinstance(parsed.get("arguments"), dict)
        and set(parsed["arguments"]) == {"command"}
        and isinstance(parsed["arguments"].get("command"), str)
        and parsed["arguments"]["command"].strip()
    )
    return valid, parsed if isinstance(parsed, dict) else None


def classify(task: str, text: str, stop_reason: str, *, max_tokens: int, token_count: int) -> dict[str, Any]:
    stripped = text.strip()
    if task == "initial_tool":
        strict, parsed = _json_tool(stripped, opening_supplied=True)
        actual_action = "tool" if strict else "invalid"
        expected_action = "tool"
    elif task == "continue_tool":
        strict, parsed = _json_tool(stripped, opening_supplied=False)
        actual_action = "tool" if strict else "invalid"
        expected_action = "tool"
    elif task == "final_answer":
        parsed = None
        strict = bool(
            stripped.startswith("<answer>")
            and stripped.endswith("</answer>")
            and stripped[len("<answer>") : -len("</answer>")].strip()
            and "<tool_call>" not in stripped
        )
        actual_action = "answer" if strict else "invalid"
        expected_action = "answer"
    else:
        parsed = None
        strict = bool(
            stripped
            and not stripped.startswith("<answer>")
            and stripped.endswith("</answer>")
            and stripped[: -len("</answer>")].strip()
            and "<tool_call>" not in stripped
        )
        actual_action = "answer" if strict else "invalid"
        expected_action = "answer"
    return {
        "strict_envelope": strict,
        "expected_action": actual_action == expected_action,
        "actual_action": actual_action,
        "parsed_tool": parsed,
        "nonempty": bool(stripped),
        "no_reasoning_or_role_leak": not any(marker in stripped for marker in PROTOCOL_MARKERS),
        "stopped_before_limit": stop_reason != "max_tokens" and token_count < max_tokens,
    }


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return float(ordered[index])


def _slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    fields = (
        "strict_envelope",
        "expected_action",
        "nonempty",
        "no_reasoning_or_role_leak",
        "stopped_before_limit",
        "exact_response",
    )
    result: dict[str, Any] = {"total": total}
    for field in fields:
        count = sum(bool(row[field]) for row in rows)
        result[field] = count
        result[field + "_rate"] = round(count / total if total else 0.0, 6)
    latencies = [float(row["elapsed_ms"]) for row in rows]
    result["mean_elapsed_ms"] = round(sum(latencies) / total if total else 0.0, 3)
    result["p50_elapsed_ms"] = round(_percentile(latencies, 0.50), 3)
    result["p95_elapsed_ms"] = round(_percentile(latencies, 0.95), 3)
    result["stop_reason_counts"] = dict(sorted(Counter(str(row["stop_reason"]) for row in rows).items()))
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = _slice(rows)
    for key in ("task", "family", "language"):
        result[key + "s"] = {
            value: _slice([row for row in rows if str(row[key]) == value])
            for value in sorted({str(row[key]) for row in rows})
        }
    return result


def _result(
    *,
    dataset: Path,
    dataset_rows: int,
    endpoint: str,
    health: dict[str, Any],
    max_tokens: int,
    predictions: list[dict[str, Any]],
    elapsed_seconds: float,
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "isolated_policy_dev_not_deployed" if complete else "running_checkpoint",
        "complete": complete,
        "dataset": {"path": str(dataset), "sha256": sha256(dataset), "rows": dataset_rows},
        "runner_sha256": sha256(Path(__file__).resolve()),
        "endpoint": endpoint,
        "model": health.get("model"),
        "health_before": health,
        "max_tokens": max_tokens,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "summary": summarize(predictions),
        "rows": predictions,
    }


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_resume_rows(
    output: Path,
    *,
    dataset: Path,
    rows: list[dict[str, Any]],
    endpoint: str,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], float]:
    if not output.exists():
        return [], 0.0
    value = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("resume output has an incompatible schema")
    metadata = value.get("dataset") if isinstance(value.get("dataset"), dict) else {}
    if metadata.get("sha256") != sha256(dataset) or metadata.get("rows") != len(rows):
        raise ValueError("resume output dataset does not match")
    if value.get("endpoint") != endpoint or value.get("max_tokens") != max_tokens:
        raise ValueError("resume output inference contract does not match")
    predictions = value.get("rows")
    if not isinstance(predictions, list) or any(not isinstance(row, dict) for row in predictions):
        raise ValueError("resume output rows are invalid")
    expected_ids = [str(row["id"]) for row in rows[: len(predictions)]]
    actual_ids = [str(row.get("id", "")) for row in predictions]
    if actual_ids != expected_ids:
        raise ValueError("resume output is not a contiguous dataset prefix")
    return list(predictions), float(value.get("elapsed_seconds") or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8317")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    rows = load_rows(dataset)
    if args.limit:
        rows = rows[: args.limit]
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    with urlopen(args.endpoint.rstrip("/") + "/health", timeout=args.timeout) as response:
        health = _read_json(response)

    predictions, previous_elapsed = (
        load_resume_rows(
            output,
            dataset=dataset,
            rows=rows,
            endpoint=args.endpoint,
            max_tokens=args.max_tokens,
        )
        if args.resume
        else ([], 0.0)
    )
    started = time.perf_counter()
    for index, row in enumerate(rows[len(predictions) :], len(predictions) + 1):
        request_started = time.perf_counter()
        response = _post_json(
            args.endpoint,
            "/v1/completions",
            {
                "prompt": str(row["prompt"]),
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stop": ["\n\nSystem:", "\n\nUser:", "\n\nAssistant:", "\n\nTool:"],
            },
            args.timeout,
        )
        g1i = response.get("g1i") if isinstance(response.get("g1i"), dict) else {}
        choices = response.get("choices") if isinstance(response.get("choices"), list) else []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        text = str(g1i.get("text") if "text" in g1i else choice.get("text") or "")
        token_ids = g1i.get("token_ids") if isinstance(g1i.get("token_ids"), list) else []
        stop_reason = str(g1i.get("stop_reason") or choice.get("finish_reason") or "")
        checks = classify(
            str(row["task"]), text, stop_reason,
            max_tokens=args.max_tokens, token_count=len(token_ids),
        )
        prediction = {
            "id": str(row["id"]),
            "trajectory_id": str(row["trajectory_id"]),
            "family": str(row["family"]),
            "language": str(row["language"]),
            "task": str(row["task"]),
            "expected_response": str(row["response"]),
            "prediction": text,
            "token_ids": token_ids,
            "stop_reason": stop_reason,
            "elapsed_ms": round((time.perf_counter() - request_started) * 1000.0, 3),
            "exact_response": text.strip() == str(row["response"]).strip(),
            **checks,
        }
        predictions.append(prediction)
        elapsed_seconds = previous_elapsed + time.perf_counter() - started
        if index % args.checkpoint_every == 0 or index == len(rows):
            _write_atomic(
                output,
                _result(
                    dataset=dataset,
                    dataset_rows=len(rows),
                    endpoint=args.endpoint,
                    health=health,
                    max_tokens=args.max_tokens,
                    predictions=predictions,
                    elapsed_seconds=elapsed_seconds,
                    complete=False,
                ),
            )
        print(
            f"[{index}/{len(rows)}] {prediction['id']} "
            f"strict={prediction['strict_envelope']} stop={stop_reason} "
            f"elapsed={prediction['elapsed_ms']:.0f}ms",
            flush=True,
        )

    result = _result(
        dataset=dataset,
        dataset_rows=len(rows),
        endpoint=args.endpoint,
        health=health,
        max_tokens=args.max_tokens,
        predictions=predictions,
        elapsed_seconds=previous_elapsed + time.perf_counter() - started,
        complete=True,
    )
    _write_atomic(output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"output={output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
