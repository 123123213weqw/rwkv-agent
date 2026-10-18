#!/usr/bin/env python3
"""Frozen long-horizon Agent benchmark through the Rust TaskSpec/TaskLedger path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


SCHEMA = "rwkv-agent-long-horizon.v1"
PROTOCOL_MARKERS = ("<tool_call>", "</tool_call>", "<tool_result>", "</tool_result>")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def load_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    required = {"id", "category", "language", "objective", "fixtures", "flat", "stages", "expect"}
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"line {line_no}: invalid top-level keys")
        case_id = str(row["id"])
        if case_id in ids:
            raise ValueError(f"line {line_no}: duplicate id {case_id}")
        if row["language"] not in {"zh", "en"} or "{run_dir}" not in row["objective"]:
            raise ValueError(f"line {line_no}: invalid language/objective")
        for relative, content in row["fixtures"].items():
            safe_relative(str(relative))
            if not isinstance(content, str):
                raise ValueError(f"line {line_no}: fixture must be text")
        stages = row["stages"]
        if not isinstance(stages, list) or len(stages) < 3:
            raise ValueError(f"line {line_no}: at least three stages required")
        ids.add(case_id)
        rows.append(row)
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def request_json(method: str, endpoint: str, path: str, payload: dict[str, Any] | None, timeout: float) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = Request(endpoint.rstrip("/") + path, data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            value = json.load(response)
            return int(response.status), dict(value)
    except HTTPError as exc:
        try:
            value = json.load(exc)
        except Exception:
            value = {"status": "http_error", "error": str(exc)}
        return int(exc.code), dict(value)


def prepare(case: dict[str, Any], root: Path) -> dict[str, str]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    original: dict[str, str] = {}
    for relative, content in case["fixtures"].items():
        target = root / safe_relative(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        original[relative] = hashlib.sha256(content.encode()).hexdigest()
    return original


def stage_responses(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    task = ledger.get("task") if isinstance(ledger.get("task"), dict) else {}
    stages = task.get("stages") if isinstance(task.get("stages"), list) else []
    return [stage for stage in stages if isinstance(stage, dict)]


def tool_rows(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in stages:
        response = stage.get("response") if isinstance(stage.get("response"), dict) else {}
        values = response.get("trace", {}).get("agent", {}).get("tool_steps", [])
        if isinstance(values, list):
            rows.extend(dict(value) for value in values if isinstance(value, dict))
    return rows


def event_rows(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in stages:
        response = stage.get("response") if isinstance(stage.get("response"), dict) else {}
        values = response.get("trace", {}).get("agent", {}).get("events", [])
        if isinstance(values, list):
            rows.extend(dict(value) for value in values if isinstance(value, dict))
    return rows


def verify(case: dict[str, Any], root: Path, original: dict[str, str], http_status: int, response: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    expect = case["expect"]
    stages = stage_responses(ledger)
    tools = tool_rows(stages)
    events = event_rows(stages)
    results = [row.get("result") for row in tools if isinstance(row.get("result"), dict)]
    stdout = "\n".join(str(value.get("stdout") or "") for value in results)
    artifacts_ok = True
    for relative, expected in expect.get("files_exact", {}).items():
        target = root / safe_relative(relative)
        actual = target.read_text() if target.is_file() else None
        if actual != expected:
            artifacts_ok = False
            errors.append(f"file_exact:{relative}")
    for relative, values in expect.get("files_contains", {}).items():
        target = root / safe_relative(relative)
        actual = target.read_text() if target.is_file() else ""
        if not all(str(value) in actual for value in values):
            artifacts_ok = False
            errors.append(f"file_contains:{relative}")
    for relative, expected in expect.get("json_equals", {}).items():
        target = root / safe_relative(relative)
        try:
            actual = json.loads(target.read_text())
        except Exception:
            actual = None
        if actual != expected:
            artifacts_ok = False
            errors.append(f"json_equals:{relative}")
    protected_ok = True
    for relative in expect.get("forbidden_changes", []):
        target = root / safe_relative(relative)
        actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else "missing"
        if actual != original.get(relative):
            protected_ok = False
            errors.append(f"protected_changed:{relative}")
    stdout_ok = all(str(value) in stdout for value in expect.get("stdout_contains", []))
    task = ledger.get("task") if isinstance(ledger.get("task"), dict) else {}
    attempted_stages = [stage for stage in stages if int(stage.get("attempts") or 0) > 0]
    attempts = [int(stage.get("attempts") or 0) for stage in stages]
    attempted_responses = [
        stage.get("response")
        for stage in attempted_stages
        if isinstance(stage.get("response"), dict)
    ]
    protocol_valid = bool(
        http_status == 200
        and len(attempted_responses) == len(attempted_stages)
        and attempted_responses
        and all(
            value.get("error_code") != "protocol_error"
            and (value.get("route") or {}).get("mode") == "tool_loop"
            and (value.get("route") or {}).get("strict") is True
            for value in attempted_responses
        )
    )
    release_events = [event for event in events if event.get("type") == "state_released"]
    state_releases = sum(event.get("success") is True for event in release_events)
    release_failures = sum(event.get("success") is not True for event in release_events)
    # A root-backed workspace stage normally releases both its immutable root
    # and task worker; recovery refreshes can legitimately add more releases.
    # Equality with stage count therefore mislabels a clean run as a leak.
    lifecycle_ok = (
        bool(attempted_stages)
        and state_releases >= len(attempted_stages)
        and release_failures == 0
    )
    answer_text = "\n".join(str((stage.get("response") or {}).get("answer") or "") for stage in stages)
    protocol_leak_free = not any(marker in answer_text for marker in PROTOCOL_MARKERS)
    commands = []
    for row in tools:
        name = str(row.get("name") or "")
        arguments = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
        if name == "run_command":
            commands.append(str(arguments.get("command") or ""))
        else:
            commands.append(
                f"{name}:"
                + json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
    normalized = [" ".join(command.split()) for command in commands]
    adjacent_repeats = sum(left == right for left, right in zip(normalized, normalized[1:]))
    action_count = len(tools)
    action_budget_ok = int(expect["min_actions"]) <= action_count <= int(expect["max_actions"])
    checks = {
        "artifacts_ok": artifacts_ok,
        "protected_inputs_ok": protected_ok,
        "stdout_ok": stdout_ok,
        "protocol_valid": protocol_valid,
        "lifecycle_ok": lifecycle_ok,
        "protocol_leak_free": protocol_leak_free,
        "action_budget_ok": action_budget_ok,
        "no_adjacent_repeat_loop": adjacent_repeats == 0,
        "ledger_succeeded": task.get("status") == "succeeded",
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    return {
        **checks,
        "passed": all(checks.values()),
        "errors": errors,
        "action_count": action_count,
        "adjacent_repeats": adjacent_repeats,
        "state_releases": state_releases,
        "state_release_failures": release_failures,
        "stage_count": len(stages),
        "stage_attempts": attempts,
        "commands": commands,
        "tool_results": results,
    }


def task_spec(case: dict[str, Any], arm: str, run_dir: str) -> dict[str, Any]:
    base = {
        "schema_version": 1,
        "objective": case["objective"].format(run_dir=run_dir),
        "working_directory": run_dir,
    }
    if arm == "flat":
        base.update(case["flat"])
        base["stages"] = []
    else:
        base.update({"acceptance_criteria": case["flat"]["acceptance_criteria"], "constraints": case["flat"]["constraints"], "verification_commands": case["flat"]["verification_commands"], "requires_mutation": case["flat"]["requires_mutation"], "stages": case["stages"]})
    return base


def run_case(endpoint: str, case: dict[str, Any], arm: str, workspace_base: Path, timeout: float) -> dict[str, Any]:
    run_dir = f"long-horizon/{arm}/{case['id']}"
    root = workspace_base / run_dir
    original = prepare(case, root)
    task_id = f"lh-v1-{arm}-{case['id']}"
    started = time.perf_counter()
    try:
        status, response = request_json("POST", endpoint, "/v1/agent/run", {"session_id": task_id, "task_id": task_id, "task_spec": task_spec(case, arm, run_dir)}, timeout)
    except Exception as exc:
        status, response = 0, {"status": "exception", "error": f"{type(exc).__name__}: {exc}"}
    try:
        _, ledger = request_json("GET", endpoint, f"/v1/task-ledger/{task_id}", None, 20)
    except Exception as exc:
        ledger = {"status": "exception", "error": str(exc)}
    checks = verify(case, root, original, status, response, ledger)
    return {"case_id": case["id"], "category": case["category"], "language": case["language"], "arm": arm, "run_dir": run_dir, "task_id": task_id, "http_status": status, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), **checks, "response": response, "ledger": ledger}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        elapsed = sorted(float(row["elapsed_ms"]) for row in selected)
        result[arm] = {
            "cases": len(selected),
            "passed": sum(row["passed"] for row in selected),
            "success_rate": round(sum(row["passed"] for row in selected) / len(selected), 6),
            "artifact_rate": round(sum(row["artifacts_ok"] for row in selected) / len(selected), 6),
            "protocol_rate": round(sum(row["protocol_valid"] for row in selected) / len(selected), 6),
            "lifecycle_rate": round(sum(row["lifecycle_ok"] for row in selected) / len(selected), 6),
            "mean_actions": round(sum(row["action_count"] for row in selected) / len(selected), 3),
            "mean_elapsed_ms": round(sum(elapsed) / len(elapsed), 3),
            "p95_elapsed_ms": elapsed[max(0, math.ceil(len(elapsed) * .95) - 1)],
            "adjacent_repeats": sum(row["adjacent_repeats"] for row in selected),
        }
    if {"flat", "staged"} <= set(result):
        result["comparison"] = {
            "success_rate_delta_staged_minus_flat": round(result["staged"]["success_rate"] - result["flat"]["success_rate"], 6),
            "mean_actions_delta_staged_minus_flat": round(result["staged"]["mean_actions"] - result["flat"]["mean_actions"], 3),
            "lifecycle_no_regression": result["staged"]["lifecycle_rate"] >= result["flat"]["lifecycle_rate"],
            "protocol_no_regression": result["staged"]["protocol_rate"] >= result["flat"]["protocol_rate"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:19120")
    parser.add_argument("--dataset", default="benchmarks/long_horizon_agent_v1.jsonl")
    parser.add_argument("--workspace-base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arms", default="flat,staged")
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    dataset, workspace, output = Path(args.dataset).resolve(), Path(args.workspace_base).resolve(), Path(args.output).resolve()
    cases = load_cases(dataset)
    arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    if not arms or any(arm not in {"flat", "staged"} for arm in arms):
        raise ValueError("arms must be flat and/or staged")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for arm in arms:
        for case in cases:
            row = run_case(args.endpoint, case, arm, workspace, args.timeout)
            rows.append(row)
            print(f"[{arm}] {case['id']}: {'PASS' if row['passed'] else 'FAIL'} actions={row['action_count']} stages={row['stage_count']} elapsed={row['elapsed_ms']:.0f}ms", flush=True)
    result = {"schema": SCHEMA, "status": "isolated_benchmark_only_not_deployed", "dataset": {"path": str(dataset), "sha256": sha256(dataset), "cases": len(cases)}, "runner_sha256": sha256(Path(__file__).resolve()), "endpoint": args.endpoint, "arms": arms, "elapsed_seconds": round(time.perf_counter() - started, 3), "summary": summarize(rows), "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"output={output}\nsha256={sha256(output)}")


if __name__ == "__main__":
    main()
