#!/usr/bin/env python3
"""Run the frozen GeneralAgent-Loop-v1 suite through the Rust control plane.

The runner prepares deterministic per-case workspaces, sends each task through
``/v1/agent/run``, validates real filesystem effects and command observations,
and compares sequential C1 with mixed C4 execution.  Commands themselves are
executed only by the Rust Bubblewrap executor configured for the benchmark.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
from http.client import HTTPResponse
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


SCHEMA = "rwkv-agent-general-agent-loop.v1"
PROTOCOL_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<tool_result>",
    "</tool_result>",
    "<think>",
    "</think>",
)


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    language: str
    task: str
    fixtures: dict[str, str]
    expect: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def load_cases(path: Path) -> list[Case]:
    rows: list[Case] = []
    ids: set[str] = set()
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: case must be an object")
        required = {"id", "category", "language", "task", "fixtures", "expect"}
        if set(value) != required:
            raise ValueError(f"line {line_number}: expected keys {sorted(required)}")
        case_id = str(value["id"])
        if case_id in ids:
            raise ValueError(f"line {line_number}: duplicate id {case_id}")
        if value["language"] not in {"zh", "en"}:
            raise ValueError(f"line {line_number}: invalid language")
        if "{run_dir}" not in str(value["task"]):
            raise ValueError(f"line {line_number}: task must contain {{run_dir}}")
        fixtures = value["fixtures"]
        expect = value["expect"]
        if not isinstance(fixtures, dict) or not isinstance(expect, dict):
            raise ValueError(f"line {line_number}: fixtures and expect must be objects")
        for relative, content in fixtures.items():
            _relative_path(str(relative))
            if not isinstance(content, str):
                raise ValueError(f"line {line_number}: fixture content must be text")
        for group in ("files_exact", "files_contains", "json_equals"):
            values = expect.get(group, {})
            if not isinstance(values, dict):
                raise ValueError(f"line {line_number}: {group} must be an object")
            for relative in values:
                _relative_path(str(relative))
        ids.add(case_id)
        rows.append(
            Case(
                id=case_id,
                category=str(value["category"]),
                language=str(value["language"]),
                task=str(value["task"]),
                fixtures={str(key): str(content) for key, content in fixtures.items()},
                expect=dict(expect),
            )
        )
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def prepare_workspace(case: Case, *, workspace_base: Path, run_dir: str) -> Path:
    relative = _relative_path(run_dir)
    root = workspace_base / relative
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for name, content in case.fixtures.items():
        target = root / _relative_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return root


def _read_response(response: HTTPResponse) -> tuple[int, dict[str, Any]]:
    payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("HTTP response must be a JSON object")
    return int(response.status), payload


def post_json(
    endpoint: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    request = Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return _read_response(response)
    except HTTPError as exc:
        try:
            value = json.load(exc)
        except (ValueError, json.JSONDecodeError):
            value = {"status": "http_error", "error": str(exc)}
        if not isinstance(value, dict):
            value = {"status": "http_error", "error": str(value)}
        return int(exc.code), value


def _tool_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    trace = response.get("trace")
    agent = trace.get("agent") if isinstance(trace, dict) else None
    rows = agent.get("tool_steps") if isinstance(agent, dict) else None
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _events(response: dict[str, Any]) -> list[dict[str, Any]]:
    trace = response.get("trace")
    agent = trace.get("agent") if isinstance(trace, dict) else None
    events = agent.get("events") if isinstance(agent, dict) else None
    if not isinstance(events, list):
        return []
    return [dict(event) for event in events if isinstance(event, dict)]


def _verify_case(
    case: Case,
    *,
    root: Path,
    response: dict[str, Any],
    http_status: int,
    max_steps: int,
) -> dict[str, Any]:
    expect = case.expect
    answer = str(response.get("answer") or "")
    tools = _tool_rows(response)
    events = _events(response)
    errors: list[str] = []
    artifacts_ok = True

    for relative, expected in dict(expect.get("files_exact", {})).items():
        target = root / _relative_path(str(relative))
        actual = target.read_text() if target.is_file() else None
        if actual != expected:
            artifacts_ok = False
            errors.append(f"file_exact:{relative}")
    for relative, required in dict(expect.get("files_contains", {})).items():
        target = root / _relative_path(str(relative))
        actual = target.read_text() if target.is_file() else ""
        values = required if isinstance(required, list) else [required]
        if not all(str(value) in actual for value in values):
            artifacts_ok = False
            errors.append(f"file_contains:{relative}")
    for relative, expected in dict(expect.get("json_equals", {})).items():
        target = root / _relative_path(str(relative))
        try:
            actual = json.loads(target.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            actual = object()
        if actual != expected:
            artifacts_ok = False
            errors.append(f"json_equals:{relative}")

    folded_answer = answer.casefold()
    for value in expect.get("answer_contains", []):
        if str(value).casefold() not in folded_answer:
            artifacts_ok = False
            errors.append(f"answer_missing:{value}")

    results = [row.get("result") for row in tools]
    results = [dict(value) for value in results if isinstance(value, dict)]
    combined_stdout = "\n".join(str(value.get("stdout") or "") for value in results)
    for value in expect.get("stdout_contains", []):
        if str(value) not in combined_stdout:
            artifacts_ok = False
            errors.append(f"stdout_missing:{value}")

    if bool(expect.get("must_fail_before_success", False)):
        failure_indices = [
            index
            for index, value in enumerate(results)
            if value.get("status") != "ok" or value.get("exit_code") not in {0, None}
        ]
        success_indices = [
            index
            for index, value in enumerate(results)
            if value.get("status") == "ok" and value.get("exit_code") in {0, None}
        ]
        recovered = bool(
            failure_indices
            and success_indices
            and min(failure_indices) < max(success_indices)
        )
        if not recovered:
            artifacts_ok = False
            errors.append("missing_failure_recovery")

    route = response.get("route")
    route = route if isinstance(route, dict) else {}
    protocol_valid = bool(
        http_status == 200
        and response.get("status") == "ok"
        and route.get("mode") == "tool_loop"
        and route.get("strict") is True
        and isinstance(route.get("steps"), int)
    )
    tool_only = bool(tools) and all(row.get("name") == "run_command" for row in tools)
    state_released = any(
        event.get("type") == "state_released" and event.get("success") is True
        for event in events
    )
    no_protocol_leak = not any(marker in answer for marker in PROTOCOL_MARKERS)
    minimum_actions = int(expect.get("min_steps", 1))
    action_count = len(tools)
    step_budget_ok = action_count <= max_steps
    minimum_actions_met = action_count >= minimum_actions
    checks = {
        "protocol_valid": protocol_valid,
        "tool_only": tool_only,
        "state_released": state_released,
        "no_protocol_leak": no_protocol_leak,
        "artifacts_verified": artifacts_ok,
        "minimum_actions_met": minimum_actions_met,
        "step_budget_ok": step_budget_ok,
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return {
        **checks,
        "passed": all(checks.values()),
        "answer": answer,
        "action_count": action_count,
        "minimum_actions": minimum_actions,
        "tool_names": [row.get("name") for row in tools],
        "commands": [
            str((row.get("arguments") or {}).get("command") or "")
            if isinstance(row.get("arguments"), dict)
            else ""
            for row in tools
        ],
        "tool_results": results,
        "errors": errors,
    }


def run_case(
    endpoint: str,
    case: Case,
    *,
    profile: str,
    workspace_base: Path,
    timeout: float,
    max_steps: int,
) -> dict[str, Any]:
    run_dir = f"runs/{profile}/{case.id}"
    root = prepare_workspace(case, workspace_base=workspace_base, run_dir=run_dir)
    task = case.task.format(run_dir=run_dir)
    session_id = f"general-agent-loop-v1-{profile}-{case.id}"
    started = time.perf_counter()
    try:
        http_status, response = post_json(
            endpoint,
            "/v1/agent/run",
            {
                "message": task,
                "session_id": session_id,
                "working_directory": run_dir,
            },
            timeout=timeout,
        )
        exception = ""
    except Exception as exc:
        http_status = 0
        response = {"status": "exception", "error": f"{type(exc).__name__}: {exc}"}
        exception = str(response["error"])
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    verification = _verify_case(
        case,
        root=root,
        response=response,
        http_status=http_status,
        max_steps=max_steps,
    )
    return {
        "case_id": case.id,
        "category": case.category,
        "language": case.language,
        "profile": profile,
        "run_dir": run_dir,
        "task": task,
        "session_id": session_id,
        "http_status": http_status,
        "elapsed_ms": round(elapsed_ms, 3),
        "exception": exception,
        **verification,
        "response": response,
    }


def run_profile(
    endpoint: str,
    cases: list[Case],
    *,
    profile: str,
    concurrency: int,
    workspace_base: Path,
    timeout: float,
    max_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                run_case,
                endpoint,
                case,
                profile=profile,
                workspace_base=workspace_base,
                timeout=timeout,
                max_steps=max_steps,
            )
            for case in cases
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"[{profile}] {row['case_id']}: "
                f"{'PASS' if row['passed'] else 'FAIL'} "
                f"steps={row['action_count']} elapsed={row['elapsed_ms']:.0f}ms",
                flush=True,
            )
    return sorted(rows, key=lambda row: str(row["case_id"]))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _slice_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    fields = (
        "passed",
        "protocol_valid",
        "tool_only",
        "state_released",
        "no_protocol_leak",
        "artifacts_verified",
        "minimum_actions_met",
        "step_budget_ok",
    )
    result: dict[str, Any] = {"total": total}
    for field in fields:
        count = sum(bool(row[field]) for row in rows)
        result[field] = count
        result[field + "_rate"] = round(count / total if total else 0.0, 6)
    latencies = [float(row["elapsed_ms"]) for row in rows]
    result["mean_elapsed_ms"] = round(sum(latencies) / total if total else 0.0, 3)
    result["p95_elapsed_ms"] = round(_p95(latencies), 3)
    result["mean_actions"] = round(
        sum(int(row["action_count"]) for row in rows) / total if total else 0.0,
        6,
    )
    result["http_status_counts"] = {
        str(status): sum(int(row["http_status"]) == status for row in rows)
        for status in sorted({int(row["http_status"]) for row in rows})
    }
    return result


def _normalize_commands(row: dict[str, Any]) -> tuple[str, ...]:
    run_dir = str(row["run_dir"])
    return tuple(str(command).replace(run_dir, "{run_dir}") for command in row["commands"])


def summarize(rows: list[dict[str, Any]], *, max_p95_ms: float) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for profile in sorted({str(row["profile"]) for row in rows}):
        selected = [row for row in rows if row["profile"] == profile]
        summary = _slice_summary(selected)
        summary["categories"] = {
            category: _slice_summary(
                [row for row in selected if row["category"] == category]
            )
            for category in sorted({str(row["category"]) for row in selected})
        }
        summary["languages"] = {
            language: _slice_summary(
                [row for row in selected if row["language"] == language]
            )
            for language in sorted({str(row["language"]) for row in selected})
        }
        fix_rate = summary["categories"].get("fix_test", {}).get("passed_rate", 0.0)
        reliability = all(
            summary[field + "_rate"] == 1.0
            for field in (
                "protocol_valid",
                "tool_only",
                "state_released",
                "no_protocol_leak",
                "step_budget_ok",
            )
        )
        summary["mvp_gate"] = {
            "task_success_at_least_80pct": summary["passed_rate"] >= 0.80,
            "fix_test_at_least_80pct": fix_rate >= 0.80,
            "protocol_and_lifecycle_100pct": reliability,
            "p95_within_budget": summary["p95_elapsed_ms"] <= max_p95_ms,
        }
        summary["mvp_pass"] = all(summary["mvp_gate"].values())
        summary["candidate_pass"] = bool(
            summary["mvp_pass"] and summary["passed_rate"] >= 0.90
        )
        profiles[profile] = summary

    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), {})[str(row["profile"])] = row
    comparable = [pair for pair in by_case.values() if {"c1", "c4"} <= set(pair)]
    trajectory_exact = sum(
        _normalize_commands(pair["c1"]) == _normalize_commands(pair["c4"])
        and pair["c1"]["answer"] == pair["c4"]["answer"]
        for pair in comparable
    )
    c1_rate = float(profiles.get("c1", {}).get("passed_rate", 0.0))
    c4_rate = float(profiles.get("c4", {}).get("passed_rate", 0.0))
    drop = c1_rate - c4_rate
    cross_profile = {
        "comparable_cases": len(comparable),
        "command_answer_exact": trajectory_exact,
        "command_answer_exact_rate": round(
            trajectory_exact / len(comparable) if comparable else 0.0, 6
        ),
        "c1_success_rate": c1_rate,
        "c4_success_rate": c4_rate,
        "c1_to_c4_drop": round(drop, 6),
        "drop_within_5pp": drop <= 0.05,
    }
    return {
        "profiles": profiles,
        "cross_profile": cross_profile,
        "mvp_pass": bool(
            profiles
            and all(profile["mvp_pass"] for profile in profiles.values())
            and cross_profile["drop_within_5pp"]
        ),
        "candidate_pass": bool(
            profiles
            and all(profile["candidate_pass"] for profile in profiles.values())
            and cross_profile["drop_within_5pp"]
        ),
    }


def run_security_probes(
    endpoint: str,
    *,
    workspace_base: Path,
    timeout: float,
) -> dict[str, Any]:
    sentinel = workspace_base.parent / "host-secret.txt"
    sentinel_value = "HOST_SECRET_MUST_NOT_BE_VISIBLE"
    sentinel.write_text(sentinel_value + "\n")
    probes = {
        "pwd": "pwd",
        "parent_escape": "cat ../host-secret.txt",
        "network": (
            "python3 -c 'import socket; "
            'socket.create_connection(("1.1.1.1",53),timeout=1)\''
        ),
    }
    results: dict[str, Any] = {}
    for name, command in probes.items():
        status, response = post_json(
            endpoint,
            "/v1/tools/call",
            {
                "name": "run_command",
                "arguments": {"command": command},
                "session_id": f"general-agent-loop-v1-security-{name}",
            },
            timeout=timeout,
        )
        results[name] = {"http_status": status, "command": command, "response": response}
    pwd = results["pwd"]["response"]
    escape = results["parent_escape"]["response"]
    network = results["network"]["response"]
    checks = {
        "pwd_is_workspace": pwd.get("status") == "ok"
        and str(pwd.get("stdout") or "").strip() == "/workspace",
        "parent_escape_blocked": escape.get("status") != "ok"
        and sentinel_value not in str(escape.get("stdout") or ""),
        "network_blocked": network.get("status") != "ok",
    }
    return {"checks": checks, "passed": all(checks.values()), "probes": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8432")
    parser.add_argument(
        "--dataset",
        default="benchmarks/general_agent_loop_v1.jsonl",
    )
    parser.add_argument("--workspace-base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--timeout", type=float, default=220.0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--c4", type=int, default=4)
    parser.add_argument("--max-p95-ms", type=float, default=30000.0)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    workspace_base = Path(args.workspace_base).resolve()
    output = Path(args.output).resolve()
    workspace_base.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = load_cases(dataset)
    if len(cases) != 30:
        raise ValueError(f"GeneralAgent-Loop-v1 must contain exactly 30 cases, got {len(cases)}")

    started = time.perf_counter()
    c1 = run_profile(
        args.endpoint,
        cases,
        profile="c1",
        concurrency=1,
        workspace_base=workspace_base,
        timeout=args.timeout,
        max_steps=args.max_steps,
    )
    c4 = run_profile(
        args.endpoint,
        cases,
        profile="c4",
        concurrency=max(1, args.c4),
        workspace_base=workspace_base,
        timeout=args.timeout,
        max_steps=args.max_steps,
    )
    rows = c1 + c4
    security = run_security_probes(
        args.endpoint,
        workspace_base=workspace_base,
        timeout=args.timeout,
    )
    summary = summarize(rows, max_p95_ms=args.max_p95_ms)
    result = {
        "schema": SCHEMA,
        "status": "isolated_benchmark_only_not_deployed",
        "source_commit": args.source_commit,
        "dataset": {
            "path": str(dataset),
            "sha256": _sha256(dataset),
            "cases": len(cases),
            "languages": {
                language: sum(case.language == language for case in cases)
                for language in sorted({case.language for case in cases})
            },
            "categories": {
                category: sum(case.category == category for case in cases)
                for category in sorted({case.category for case in cases})
            },
        },
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "endpoint": args.endpoint,
        "profiles": {"c1": 1, "c4": max(1, args.c4)},
        "max_steps": args.max_steps,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "summary": summary,
        "security": security,
        "rows": rows,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": summary, "security": security["checks"]}, ensure_ascii=False, indent=2))
    print(f"output={output}")
    print(f"sha256={_sha256(output)}")


if __name__ == "__main__":
    main()
