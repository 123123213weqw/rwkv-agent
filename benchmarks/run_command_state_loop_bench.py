#!/usr/bin/env python3
"""Execute a bounded ``run_command`` loop against one recurrent RWKV State.

Every command runs in a fresh fixture directory.  The benchmark rejects path
escape and network/process-control commands, applies fixed time/output budgets,
and always releases the persistent model State.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable
from urllib.request import Request, urlopen
import uuid


TOOL_CALL = re.compile(r"\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*", re.S)
ANSWER = re.compile(r"\s*<answer>\s*(.*?)\s*</answer>\s*", re.S)
FORBIDDEN_COMMAND = re.compile(
    r"(?:^|[;&|\s])(?:sudo|su|ssh|scp|sftp|curl|wget|nc|ncat|telnet|"
    r"systemctl|service|kill|pkill|killall|mount|umount|docker|podman|"
    r"apt|apt-get|yum|dnf|pacman|brew|pip|pip3|npm|pnpm|yarn)(?:\s|$)",
    re.I,
)
ABSOLUTE_PATH = re.compile(r"(?:^|[\s><=:'\"])/(?!/)")


@dataclass(frozen=True)
class Case:
    id: str
    task: str
    fixtures: dict[str, str]
    minimum_actions: int
    verify: Callable[[Path, str, list[dict[str, Any]]], tuple[bool, str]]


def _answer_has(value: str) -> Callable[[Path, str, list[dict[str, Any]]], tuple[bool, str]]:
    def verify(_root: Path, answer: str, _steps: list[dict[str, Any]]) -> tuple[bool, str]:
        passed = value.casefold() in answer.casefold()
        return passed, "answer_contains" if passed else f"answer_missing:{value}"

    return verify


def _verify_compare(_root: Path, answer: str, _steps: list[dict[str, Any]]) -> tuple[bool, str]:
    folded = answer.casefold()
    passed = "beta" in folded and ("larger" in folded or "更大" in answer)
    return passed, "beta_larger" if passed else "comparison_missing"


def _verify_release(root: Path, answer: str, steps: list[dict[str, Any]]) -> tuple[bool, str]:
    content = (root / "config.ini").read_text().strip()
    observed = any("mode=release" in str(step.get("stdout") or "") for step in steps)
    passed = content == "mode=release" and observed and "release" in answer.casefold()
    return passed, "modified_and_verified" if passed else "release_not_verified"


def _verify_calc(root: Path, answer: str, steps: list[dict[str, Any]]) -> tuple[bool, str]:
    source = (root / "calc.py").read_text()
    final_test = any(
        "test_calc.py" in str(step.get("command") or "")
        and step.get("exit_code") == 0
        and "PASS" in str(step.get("stdout") or "")
        for step in steps
    )
    passed = "return a + b" in source and final_test and "pass" in answer.casefold()
    return passed, "fixed_and_tested" if passed else "calc_not_fixed_and_tested"


def _verify_summary(root: Path, answer: str, steps: list[dict[str, Any]]) -> tuple[bool, str]:
    target = root / "summary.txt"
    content = target.read_text().strip() if target.exists() else ""
    observed = any("count=3,total=12" in str(step.get("stdout") or "") for step in steps)
    passed = content == "count=3,total=12" and observed and "summary.txt" in answer.casefold()
    return passed, "created_and_verified" if passed else "summary_not_verified"


CASES = (
    Case(
        id="read_value",
        task=(
            "Inspect note.txt with a command, then report the exact value of alpha. "
            "Do not guess from the task text."
        ),
        fixtures={"note.txt": "alpha=7\n"},
        minimum_actions=1,
        verify=_answer_has("7"),
    ),
    Case(
        id="compare_files",
        task=(
            "分别使用独立命令查看 a.txt 和 b.txt，然后报告哪个命名值更大。"
            "回答前必须查看两个文件。"
        ),
        fixtures={"a.txt": "alpha=11\n", "b.txt": "beta=19\n"},
        minimum_actions=2,
        verify=_verify_compare,
    ),
    Case(
        id="calculate_sum",
        task=(
            "先查看 numbers.txt，再使用另一个独立命令计算其中整数的总和。"
            "两个命令完成后才能报告数字结果。"
        ),
        fixtures={"numbers.txt": "13\n29\n"},
        minimum_actions=2,
        verify=_answer_has("42"),
    ),
    Case(
        id="modify_and_verify",
        task=(
            "使用独立命令依次查看 config.ini、将 mode=debug 修改为 mode=release，"
            "再读取文件验证修改，然后报告结果。"
        ),
        fixtures={"config.ini": "mode=debug\n"},
        minimum_actions=3,
        verify=_verify_release,
    ),
    Case(
        id="fix_and_test",
        task=(
            "Inspect calc.py and test_calc.py, run the test, fix calc.py, then run the test "
            "again. Use separate commands and finish only after the test prints PASS."
        ),
        fixtures={
            "calc.py": "def add(a, b):\n    return a - b\n",
            "test_calc.py": (
                "from calc import add\n"
                "assert add(2, 3) == 5, add(2, 3)\n"
                "print('PASS')\n"
            ),
        },
        minimum_actions=4,
        verify=_verify_calc,
    ),
    Case(
        id="create_and_verify",
        task=(
            "使用一个命令创建 summary.txt，内容必须严格为 count=3,total=12。"
            "再使用另一个独立命令读回文件，并报告 summary.txt 已验证。"
        ),
        fixtures={},
        minimum_actions=2,
        verify=_verify_summary,
    ),
)


def render_root_prompt(case: Case, *, style: str) -> str:
    prompt = (
        "System: You are a bounded command agent operating in an isolated current "
        "directory. The only function is run_command(command). Use relative paths. "
        "On every step output exactly one strict tool call or one final answer. A tool "
        "call has exactly this shape: <tool_call>{\"name\":\"run_command\","
        "\"arguments\":{\"command\":\"...\"}}</tool_call>. A final answer has "
        "exactly this shape: <answer>...</answer>. Never emit reasoning, Markdown, "
        "role labels, or protocol text outside those envelopes. Treat Tool Result as "
        "the only observation of command execution. Do not claim success until the "
        "required verification command succeeds."
    )
    if style == "one_loop_example":
        prompt += (
            "\n\nUser: Inspect demo.txt and report its value.\n\nAssistant: "
            '<tool_call>{"name":"run_command","arguments":{"command":'
            '"cat demo.txt"}}</tool_call>\n\nTool: <tool_result>'
            '{"status":"ok","exit_code":0,"stdout":"demo=ready\\n",'
            '"stderr":""}</tool_result>\n\nUser: Continue the original task. '
            "Call run_command again only if more work is required; otherwise return "
            "the final answer.\n\nAssistant: <answer>demo is ready.</answer>"
        )
    elif style != "instruction_only":
        raise ValueError(f"unknown style: {style}")
    return prompt + "\n\nUser: " + case.task


def parse_action(raw: str) -> dict[str, Any]:
    tool = TOOL_CALL.fullmatch(str(raw or ""))
    if tool:
        try:
            value = json.loads(tool.group(1))
            if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
                raise ValueError("payload keys")
            if value["name"] != "run_command":
                raise ValueError("tool name")
            arguments = value["arguments"]
            if not isinstance(arguments, dict) or set(arguments) != {"command"}:
                raise ValueError("argument keys")
            command = arguments["command"]
            if not isinstance(command, str) or not command.strip():
                raise ValueError("command")
            return {"kind": "tool", "command": command.strip(), "error": ""}
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return {"kind": "invalid", "command": "", "answer": "", "error": str(exc)}
    answer = ANSWER.fullmatch(str(raw or ""))
    if answer and answer.group(1).strip():
        return {
            "kind": "answer",
            "command": "",
            "answer": answer.group(1).strip(),
            "error": "",
        }
    return {"kind": "invalid", "command": "", "answer": "", "error": "envelope"}


def validate_command(command: str) -> str:
    value = str(command or "").strip()
    if not value:
        raise ValueError("empty command")
    if len(value) > 2000:
        raise ValueError("command too long")
    if "\x00" in value or ".." in value:
        raise ValueError("path escape")
    if FORBIDDEN_COMMAND.search(value):
        raise ValueError("forbidden command")
    if ABSOLUTE_PATH.search(value):
        raise ValueError("absolute path")
    return value


def execute_command(command: str, *, cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        clean = validate_command(command)
    except ValueError as exc:
        return {
            "status": "rejected",
            "exit_code": None,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(cwd),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": str(cwd),
    }
    try:
        result = subprocess.run(
            ["/bin/bash", "-lc", clean],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=8,
            check=False,
        )
        stdout = result.stdout[:8000]
        stderr = result.stderr[:4000]
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "exit_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "exit_code": None,
            "stdout": str(exc.stdout or "")[:8000],
            "stderr": str(exc.stderr or "")[:4000],
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }


class StateClient:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.url + path,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=180) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise RuntimeError("non-object response")
        return value

    def prefill(self, *, owner_id: str, prompt: str) -> dict[str, Any]:
        result = self.post(
            "/v1/states/prefill",
            {"owner_id": owner_id, "prompt": prompt, "branch": "root"},
        )
        return dict(result["state"])

    def continue_state(
        self,
        *,
        owner_id: str,
        state_id: str,
        input_text: str,
        max_tokens: int = 160,
    ) -> dict[str, Any]:
        result = self.post(
            "/v1/states/batch_continue",
            {
                "owner_id": owner_id,
                "items": [{"state_id": state_id, "input": input_text}],
                "stop": ["</tool_call>", "</answer>"],
                "max_tokens": max_tokens,
            },
        )
        rows = result["results"]
        if not isinstance(rows, list) or len(rows) != 1:
            raise RuntimeError("invalid continuation row count")
        return dict(rows[0])

    def release(self, *, owner_id: str, state_id: str) -> dict[str, Any]:
        return self.post(
            "/v1/states/release",
            {"owner_id": owner_id, "state_ids": [state_id]},
        )


def reconstruct_completion(row: dict[str, Any], *, supplied_prefix: str = "") -> str:
    text = str(row.get("text") or "")
    if supplied_prefix and not text.lstrip().startswith(supplied_prefix):
        text = supplied_prefix + text.lstrip()
    stop = str(row.get("stop_reason") or "")
    if stop in {"</tool_call>", "</answer>"}:
        text += stop
    return text


def render_observation(result: dict[str, Any]) -> str:
    compact = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    return (
        "\n\nTool: <tool_result>"
        + compact
        + "</tool_result>\n\nUser: Continue the original task. Call run_command "
        "again only if more work or verification is required; otherwise output the "
        "final answer. Output exactly one protocol envelope and no reasoning.\n\nAssistant:"
    )


def create_workspace(case: Case, *, base: Path, run_name: str) -> Path:
    root = base / run_name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for relative, content in case.fixtures.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return root


def run_case(
    client: StateClient,
    *,
    case: Case,
    style: str,
    repeat: int,
    workspace_base: Path,
    max_steps: int,
) -> dict[str, Any]:
    run_name = f"{style}-{case.id}-r{repeat}"
    workspace = create_workspace(case, base=workspace_base, run_name=run_name)
    owner_id = "command-loop-" + uuid.uuid4().hex
    state_id = ""
    steps: list[dict[str, Any]] = []
    state_ids: list[str] = []
    answer = ""
    terminal = ""
    released = False
    started = time.perf_counter()
    try:
        state = client.prefill(
            owner_id=owner_id,
            prompt=render_root_prompt(case, style=style),
        )
        state_id = str(state["state_id"])
        initial = client.continue_state(
            owner_id=owner_id,
            state_id=state_id,
            input_text="\n\nAssistant: <tool_call>",
        )
        state_ids.append(str(initial.get("state_id") or ""))
        raw = reconstruct_completion(initial, supplied_prefix="<tool_call>")
        for index in range(max_steps + 1):
            parsed = parse_action(raw)
            if parsed["kind"] == "answer":
                answer = parsed["answer"]
                terminal = "answer"
                break
            if parsed["kind"] != "tool":
                terminal = "invalid_protocol"
                steps.append({"step": index + 1, "raw": raw, "parsed": parsed})
                break
            if index >= max_steps:
                terminal = "step_limit"
                break
            command = parsed["command"]
            execution = execute_command(command, cwd=workspace)
            step = {
                "step": index + 1,
                "raw": raw,
                "command": command,
                **execution,
            }
            steps.append(step)
            continuation = client.continue_state(
                owner_id=owner_id,
                state_id=state_id,
                input_text=render_observation(execution),
            )
            state_ids.append(str(continuation.get("state_id") or ""))
            raw = reconstruct_completion(continuation)
        else:
            terminal = "step_limit"
    except Exception as exc:
        terminal = "exception"
        steps.append({"step": len(steps) + 1, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if state_id:
            try:
                client.release(owner_id=owner_id, state_id=state_id)
            except Exception as exc:
                steps.append({"release_error": f"{type(exc).__name__}: {exc}"})
            else:
                released = True
    verified, verify_reason = case.verify(workspace, answer, steps)
    state_constant = bool(state_ids) and set(state_ids) == {state_id}
    minimum_actions_met = len([step for step in steps if "command" in step]) >= case.minimum_actions
    protocol_valid = terminal == "answer"
    passed = (
        protocol_valid
        and verified
        and minimum_actions_met
        and state_constant
        and released
    )
    return {
        "case_id": case.id,
        "style": style,
        "repeat": repeat,
        "task": case.task,
        "workspace": run_name,
        "state_id": state_id,
        "state_ids": state_ids,
        "state_constant": state_constant,
        "released": released,
        "steps": steps,
        "action_count": len([step for step in steps if "command" in step]),
        "minimum_actions": case.minimum_actions,
        "minimum_actions_met": minimum_actions_met,
        "terminal": terminal,
        "protocol_valid": protocol_valid,
        "answer": answer,
        "verified": verified,
        "verify_reason": verify_reason,
        "passed": passed,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    styles: dict[str, Any] = {}
    for style in sorted({row["style"] for row in rows}):
        selected = [row for row in rows if row["style"] == style]
        repeat_groups: dict[str, list[dict[str, Any]]] = {}
        for row in selected:
            repeat_groups.setdefault(row["case_id"], []).append(row)
        exact = 0
        for group in repeat_groups.values():
            signatures = [
                (
                    tuple(step.get("command") for step in row["steps"] if "command" in step),
                    row["answer"],
                    row["terminal"],
                )
                for row in group
            ]
            exact += int(len(group) > 1 and len(set(signatures)) == 1)
        styles[style] = {
            "total": len(selected),
            "passed": sum(row["passed"] for row in selected),
            "protocol_valid": sum(row["protocol_valid"] for row in selected),
            "task_verified": sum(row["verified"] for row in selected),
            "minimum_actions_met": sum(row["minimum_actions_met"] for row in selected),
            "state_constant": sum(row["state_constant"] for row in selected),
            "released": sum(row["released"] for row in selected),
            "repeat_groups": len(repeat_groups),
            "repeat_command_answer_exact": exact,
            "mean_actions": round(
                sum(row["action_count"] for row in selected) / len(selected), 6
            ),
            "mean_elapsed_ms": round(
                sum(row["elapsed_ms"] for row in selected) / len(selected), 6
            ),
            "terminal_counts": {
                terminal: sum(row["terminal"] == terminal for row in selected)
                for terminal in sorted({row["terminal"] for row in selected})
            },
        }
    ranking = sorted(
        styles,
        key=lambda style: (
            styles[style]["passed"],
            styles[style]["protocol_valid"],
            styles[style]["repeat_command_answer_exact"],
            -styles[style]["mean_actions"],
        ),
        reverse=True,
    )
    return {"styles": styles, "ranking": ranking}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url", default="http://127.0.0.1:8417")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument(
        "--workspace-base",
        default="workspaces",
    )
    parser.add_argument("--output", default="result.json")
    args = parser.parse_args()
    workspace_base = Path(args.workspace_base).resolve()
    workspace_base.mkdir(parents=True, exist_ok=True)
    client = StateClient(args.model_url)
    jobs = [
        (case, style, repeat)
        for case in CASES
        for style in ("instruction_only", "one_loop_example")
        for repeat in range(max(1, args.repeats))
    ]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                run_case,
                client,
                case=case,
                style=style,
                repeat=repeat,
                workspace_base=workspace_base,
                max_steps=max(1, args.max_steps),
            )
            for case, style, repeat in jobs
        ]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row["style"], row["case_id"], row["repeat"]))
    payload = {
        "schema": "rwkv-agent-run-command-state-loop.v1",
        "created_unix": time.time(),
        "model_mode": "greedy_argmax",
        "model_url": args.model_url,
        "cases": len(CASES),
        "styles": ["instruction_only", "one_loop_example"],
        "repeats": max(1, args.repeats),
        "runs": len(rows),
        "max_steps": max(1, args.max_steps),
        "elapsed_s": round(time.perf_counter() - started, 6),
        "summary": summarize(rows),
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
