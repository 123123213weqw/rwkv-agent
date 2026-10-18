#!/usr/bin/env python3
"""Probe which flat execution Tool Calls a greedy G1I model already prefers.

The benchmark never executes a generated command.  It only compares strict
Tool Call generation under the same tasks, prompt, decoder and token budget.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from typing import Any
from urllib.request import Request, urlopen


TOOL_ENVELOPE = re.compile(r"\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*", re.S)


@dataclass(frozen=True)
class Arm:
    name: str
    signatures: tuple[str, ...]
    tools: tuple[str, ...]
    argument_keys: dict[str, frozenset[str]]


ARMS = (
    Arm(
        name="native_trio",
        signatures=(
            "read_file(path): read a file or list a directory",
            "write_file(path, content): write exact content to a file",
            "run_command(command): run one terminal command",
        ),
        tools=("read_file", "write_file", "run_command"),
        argument_keys={
            "read_file": frozenset({"path"}),
            "write_file": frozenset({"path", "content"}),
            "run_command": frozenset({"command"}),
        },
    ),
    Arm(
        name="run_command_only",
        signatures=("run_command(command): run one terminal command",),
        tools=("run_command",),
        argument_keys={"run_command": frozenset({"command"})},
    ),
    Arm(
        name="execute_command_only",
        signatures=("execute_command(command): execute one terminal command",),
        tools=("execute_command",),
        argument_keys={"execute_command": frozenset({"command"})},
    ),
    Arm(
        name="shell_only",
        signatures=("shell(command): run one shell command",),
        tools=("shell",),
        argument_keys={"shell": frozenset({"command"})},
    ),
    Arm(
        name="bash_only",
        signatures=("bash(command): run one Bash command",),
        tools=("bash",),
        argument_keys={"bash": frozenset({"command"})},
    ),
    Arm(
        name="python_only",
        signatures=("python(code): execute Python code",),
        tools=("python",),
        argument_keys={"python": frozenset({"code"})},
    ),
    Arm(
        name="workspace_only",
        signatures=(
            "workspace(op, input): operate on the workspace; op is read, run, or patch",
        ),
        tools=("workspace",),
        argument_keys={"workspace": frozenset({"op", "input"})},
    ),
)


CASES = (
    {
        "id": "read_en_readme",
        "kind": "read",
        "message": "Read README.md.",
        "path": "README.md",
    },
    {
        "id": "read_zh_controller",
        "kind": "read",
        "message": "读取 src/rwkv_agent/controller.py。",
        "path": "src/rwkv_agent/controller.py",
    },
    {
        "id": "read_en_space",
        "kind": "read",
        "message": "Read the file named docs/Quick Start.md.",
        "path": "docs/Quick Start.md",
    },
    {
        "id": "read_zh_head",
        "kind": "read",
        "message": "查看 pyproject.toml 的前20行。",
        "path": "pyproject.toml",
        "required_terms": ("20",),
    },
    {
        "id": "write_en_note",
        "kind": "write",
        "message": "Write exactly hello to note.txt.",
        "path": "note.txt",
        "content": "hello",
    },
    {
        "id": "write_zh_preference",
        "kind": "write",
        "message": "把“回答尽量简短”写入 prefs.txt。",
        "path": "prefs.txt",
        "content": "回答尽量简短",
    },
    {
        "id": "write_en_json",
        "kind": "write",
        "message": 'Write exactly {"enabled":true} to config.json.',
        "path": "config.json",
        "content": '{"enabled":true}',
    },
    {
        "id": "write_zh_space",
        "kind": "write",
        "message": "把 completed 写入 logs/run result.txt。",
        "path": "logs/run result.txt",
        "content": "completed",
    },
    {
        "id": "run_en_pytest",
        "kind": "run",
        "message": "Run pytest -q.",
        "required_terms": ("pytest", "-q"),
    },
    {
        "id": "run_zh_git",
        "kind": "run",
        "message": "运行 git status --short。",
        "required_terms": ("git", "status", "--short"),
    },
    {
        "id": "run_en_rg",
        "kind": "run",
        "message": 'Run rg -n "AgentController" src.',
        "required_terms": ("rg", "-n", "AgentController", "src"),
    },
    {
        "id": "run_zh_compile",
        "kind": "run",
        "message": "运行 python -m compileall -q src。",
        "required_terms": ("python", "compileall", "-q", "src"),
    },
)


CONTINUATION_PROMPTS = (
    'System: {"',
    'System: {"tools":[{"type":"function","function":{"name":"',
    'System: {"tools":[{"type":"function","function":{"name":"read',
    'System: {"tools":[{"type":"function","function":{"name":"write',
    'System: {"tools":[{"type":"function","function":{"name":"run_',
    'System: {"tools":[{"type":"function","function":{"name":"execute_',
)


def render_prompt(arm: Arm, case: dict[str, Any]) -> str:
    functions = "\n".join(f"- {value}" for value in arm.signatures)
    return (
        "System: Call exactly one function and output only "
        '<tool_call>{"name":...,"arguments":...}</tool_call>.\n'
        f"Functions:\n{functions}\n"
        "Do not answer. Do not emit reasoning or any other text.\n"
        f"User: {case['message']}\n\nAssistant:"
    )


def parse_tool_call(raw: str) -> dict[str, Any]:
    match = TOOL_ENVELOPE.fullmatch(str(raw or ""))
    if not match:
        return {"strict": False, "tool": "", "arguments": {}, "error": "envelope"}
    try:
        payload = json.loads(match.group(1))
        if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
            raise ValueError("payload keys")
        if not isinstance(payload["name"], str) or not payload["name"].strip():
            raise ValueError("tool name")
        if not isinstance(payload["arguments"], dict):
            raise ValueError("arguments")
        return {
            "strict": True,
            "tool": payload["name"].strip(),
            "arguments": payload["arguments"],
            "error": "",
        }
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"strict": False, "tool": "", "arguments": {}, "error": str(exc)}


def expected_tool(arm: Arm, case: dict[str, Any]) -> str:
    if arm.name == "native_trio":
        return {
            "read": "read_file",
            "write": "write_file",
            "run": "run_command",
        }[str(case["kind"])]
    return arm.tools[0]


def _folded_contains(value: str, expected: str) -> bool:
    return expected.casefold() in value.casefold()


def _generic_payload(parsed: dict[str, Any]) -> str:
    arguments = parsed.get("arguments") or {}
    return " ".join(str(value) for value in arguments.values())


def semantic_valid(arm: Arm, case: dict[str, Any], parsed: dict[str, Any]) -> bool:
    if not parsed.get("strict"):
        return False
    tool = str(parsed.get("tool") or "")
    arguments = parsed.get("arguments") or {}
    kind = str(case["kind"])
    if arm.name == "native_trio":
        if kind == "read":
            return _folded_contains(str(arguments.get("path") or ""), str(case["path"]))
        if kind == "write":
            return (
                _folded_contains(str(arguments.get("path") or ""), str(case["path"]))
                and str(case["content"]) in str(arguments.get("content") or "")
            )
    if arm.name == "workspace_only":
        expected_op = {"read": "read", "write": "patch", "run": "run"}[kind]
        op = str(arguments.get("op") or "").casefold()
        payload = str(arguments.get("input") or "")
        if op != expected_op:
            return False
    else:
        payload = _generic_payload(parsed)
    if kind in {"read", "write"} and not _folded_contains(payload, str(case["path"])):
        return False
    if kind == "write" and str(case["content"]) not in payload:
        return False
    for term in case.get("required_terms", ()):
        if not _folded_contains(payload, str(term)):
            return False
    if kind == "read" and tool not in {"read_file", "workspace"}:
        read_markers = ("cat", "head", "sed", "open(", "read_text", "read(")
        if not any(marker in payload.casefold() for marker in read_markers):
            return False
    if kind == "write" and tool not in {"write_file", "workspace"}:
        write_markers = (">", "tee", "write_text", "write(", "open(", "printf", "echo")
        if not any(marker in payload.casefold() for marker in write_markers):
            return False
    return True


def evaluate(arm: Arm, case: dict[str, Any], parsed: dict[str, Any]) -> dict[str, bool]:
    strict = bool(parsed.get("strict"))
    tool_correct = strict and parsed.get("tool") == expected_tool(arm, case)
    expected_keys = arm.argument_keys.get(str(parsed.get("tool") or ""), frozenset())
    schema_valid = tool_correct and frozenset((parsed.get("arguments") or {}).keys()) == expected_keys
    semantics = tool_correct and schema_valid and semantic_valid(arm, case, parsed)
    return {
        "strict": strict,
        "tool_correct": tool_correct,
        "schema_valid": schema_valid,
        "semantic_valid": semantics,
        "passed": strict and tool_correct and schema_valid and semantics,
    }


def post_completion(
    model_url: str,
    *,
    prompt: str,
    prefix_token_ids: list[int],
    max_tokens: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "prompt": prompt,
            "prefix_token_ids": prefix_token_ids,
            "max_tokens": max_tokens,
            "stop": ["</tool_call>", "</s>", "\nUser:", "\nSystem:"],
        },
        ensure_ascii=False,
    ).encode()
    request = Request(
        model_url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urlopen(request, timeout=180) as response:
        data = json.load(response)
    g1i = data["g1i"]
    stop = str(g1i.get("stop_reason") or "")
    raw = str(g1i.get("text") or "")
    if stop.startswith("</tool"):
        raw += stop
    return {
        "raw": raw,
        "stop": stop,
        "output_tokens": len(g1i.get("token_ids") or []),
        "model_elapsed_ms": float(g1i.get("elapsed_ms") or 0.0),
        "request_elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "model": data.get("model"),
    }


def run_case(
    model_url: str,
    *,
    arm: Arm,
    case: dict[str, Any],
    prefix_mode: str,
    repeat: int,
    max_tokens: int,
) -> dict[str, Any]:
    prefix = [0] if prefix_mode == "token0" else []
    completion = post_completion(
        model_url,
        prompt=render_prompt(arm, case),
        prefix_token_ids=prefix,
        max_tokens=max_tokens,
    )
    parsed = parse_tool_call(completion["raw"])
    return {
        "arm": arm.name,
        "case_id": case["id"],
        "kind": case["kind"],
        "prefix_mode": prefix_mode,
        "repeat": repeat,
        "expected_tool": expected_tool(arm, case),
        "message": case["message"],
        **completion,
        "parsed": parsed,
        "evaluation": evaluate(arm, case, parsed),
    }


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, ((95 * len(ordered) + 99) // 100) - 1)
    return ordered[index]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    arms: dict[str, Any] = {}
    for arm in (value.name for value in ARMS):
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        arm_passed = 0
        arm_total = 0
        stable_groups = 0
        stable_exact = 0
        for prefix_mode in ("none", "token0"):
            selected = [row for row in arm_rows if row["prefix_mode"] == prefix_mode]
            if not selected:
                continue
            key = f"{arm}/{prefix_mode}"
            total = len(selected)
            repeat_groups: dict[str, list[str]] = {}
            for row in selected:
                repeat_groups.setdefault(row["case_id"], []).append(row["raw"])
            repeated = [values for values in repeat_groups.values() if len(values) > 1]
            exact = sum(len(set(values)) == 1 for values in repeated)
            passed = sum(row["evaluation"]["passed"] for row in selected)
            groups[key] = {
                "total": total,
                "strict": sum(row["evaluation"]["strict"] for row in selected),
                "tool_correct": sum(row["evaluation"]["tool_correct"] for row in selected),
                "schema_valid": sum(row["evaluation"]["schema_valid"] for row in selected),
                "semantic_valid": sum(row["evaluation"]["semantic_valid"] for row in selected),
                "passed": passed,
                "pass_rate": round(passed / total, 6),
                "repeat_groups": len(repeated),
                "repeat_raw_exact": exact,
                "repeat_raw_exact_rate": round(exact / len(repeated), 6) if repeated else None,
                "mean_output_tokens": round(
                    statistics.fmean(row["output_tokens"] for row in selected), 6
                ),
                "mean_model_elapsed_ms": round(
                    statistics.fmean(row["model_elapsed_ms"] for row in selected), 6
                ),
                "p95_model_elapsed_ms": round(
                    _p95([row["model_elapsed_ms"] for row in selected]), 6
                ),
            }
            arm_passed += passed
            arm_total += total
            stable_groups += len(repeated)
            stable_exact += exact
        arms[arm] = {
            "total": arm_total,
            "passed": arm_passed,
            "pass_rate": round(arm_passed / arm_total, 6),
            "repeat_groups": stable_groups,
            "repeat_raw_exact": stable_exact,
            "repeat_raw_exact_rate": (
                round(stable_exact / stable_groups, 6) if stable_groups else None
            ),
        }
    ranking = sorted(
        arms,
        key=lambda arm: (
            arms[arm]["pass_rate"],
            arms[arm]["repeat_raw_exact_rate"] or 0.0,
            arm == "native_trio",
        ),
        reverse=True,
    )
    return {"groups": groups, "arms": arms, "ranking": ranking}


def run_continuation_probes(model_url: str, *, max_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for prompt in CONTINUATION_PROMPTS:
        completion = post_completion(
            model_url,
            prompt=prompt,
            prefix_token_ids=[0],
            max_tokens=max_tokens,
        )
        rows.append({"prompt": prompt, **completion})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url", default="http://127.0.0.1:8417")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--output",
        default="benchmarks/native_tool_prior_probe_preview4922_v1.json",
    )
    args = parser.parse_args()
    jobs = [
        (arm, case, prefix_mode, repeat)
        for arm in ARMS
        for case in CASES
        for prefix_mode in ("none", "token0")
        for repeat in range(max(1, args.repeats))
    ]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                run_case,
                args.model_url,
                arm=arm,
                case=case,
                prefix_mode=prefix_mode,
                repeat=repeat,
                max_tokens=args.max_tokens,
            )
            for arm, case, prefix_mode, repeat in jobs
        ]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(
        key=lambda row: (
            row["arm"],
            row["prefix_mode"],
            row["case_id"],
            row["repeat"],
        )
    )
    continuation = run_continuation_probes(
        args.model_url,
        max_tokens=args.max_tokens,
    )
    payload = {
        "schema": "rwkv-agent-native-tool-prior-probe.v1",
        "created_unix": time.time(),
        "model_mode": "greedy_argmax",
        "execution_enabled": False,
        "model_url": args.model_url,
        "cases": len(CASES),
        "arms": [arm.name for arm in ARMS],
        "prefix_modes": ["none", "token0"],
        "repeats": max(1, args.repeats),
        "rows": len(rows),
        "elapsed_s": round(time.perf_counter() - started, 6),
        "summary": summarize(rows),
        "continuation_probes": continuation,
        "results": rows,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload["sha256_without_sha_field"] = hashlib.sha256(canonical).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(output.resolve())


if __name__ == "__main__":
    main()
