#!/usr/bin/env python3
"""Generate randomized, benchmark-independent workspace Agent Policy SFT data.

The frozen GeneralAgent benchmark is intentionally not an input.  Every
trajectory is synthesized from a seed and teaches the deployed recurrent
protocol: one Task Worker, one command per turn, real Tool observations, error
recovery, exact file contracts, verification, and an explicit final answer.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import shlex
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "rwkv-agent-policy-sft.v1"
DATASET = "general_agent_policy"
MAX_TOOL_STEPS = 8
FAMILIES = (
    "inspect_rank",
    "create_exact_json",
    "sum_exact_file",
    "update_two_files",
    "repair_delimiter",
    "repair_filter",
    "recover_runtime",
)

SYSTEM_PROMPT = (
    "System: You are a bounded workspace agent. Complete the user's task "
    "autonomously with run_command and inspect every Tool Result before choosing "
    "the next action. At each turn output exactly one strict envelope: "
    '<tool_call>{"name":"run_command","arguments":{"command":"..."}}</tool_call> '
    "or <answer>concise user-visible answer</answer>. Output no reasoning, role "
    "labels, Markdown fences, empty commands, or text outside the envelope. Tool "
    "environment contract: the selected workspace is `/workspace`, every command "
    "starts there, and only files inside it persist between calls. Each call is a "
    "new isolated process: `/tmp`, shell variables, and process state do not "
    "persist. There is no network or package installation. Python is available "
    "only as `python3`; `python`, Pytest, and dependency installation are "
    "unavailable. A Python test file supplied by the user is a standalone script "
    "and can be executed directly with `python3` and its filename. Do not create "
    "or modify artifacts the user did not request. Treat exact content as a byte "
    "contract. If a command fails, use its actual result to choose a different "
    "corrective action instead of repeating it. After verification succeeds, "
    "answer on the next turn. Function: run_command(command)."
)


@dataclass(frozen=True)
class ToolStep:
    command: str
    result: dict[str, Any]


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    family: str
    language: str
    task: str
    fixtures: dict[str, str]
    steps: tuple[ToolStep, ...]
    answer: str


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tool_result(*, stdout: str = "", stderr: str = "", exit_code: int = 0) -> dict[str, Any]:
    return {
        "exit_code": exit_code,
        "standalone_test_runner_rewritten": False,
        "status": "ok" if exit_code == 0 else "error",
        "stderr": stderr,
        "stdout": stdout,
        "truncated": False,
        "workspace_alias_rewritten": False,
    }


def tool_call(command: str, *, opening_supplied: bool) -> str:
    payload = json.dumps(
        {"name": "run_command", "arguments": {"command": command}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if opening_supplied:
        return payload + "</tool_call>"
    return f"<tool_call>{payload}</tool_call>"


def observation(result: dict[str, Any], completed: int, *, force_answer: bool = False) -> str:
    compact = json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if force_answer:
        instruction = (
            f"Tool step {completed}/{MAX_TOOL_STEPS} is complete. No tool budget remains. "
            "Return the best truthful final answer now and do not call another tool."
        )
        suffix = "Assistant: <answer>"
    else:
        instruction = (
            f"Tool step {completed}/{MAX_TOOL_STEPS} is complete. Continue the original task. "
            "Do not repeat an identical successful command. Call one tool only if a distinct "
            "required action or verification remains; otherwise return the final answer now."
        )
        suffix = "Assistant:"
    return (
        f"\n\nTool: <tool_result>{compact}</tool_result>\n\n"
        f"User: {instruction} Output exactly one protocol envelope and no reasoning.\n\n{suffix}"
    )


def initial_prompt(task: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\nSystem: The next User message supplies the current workspace "
        "task. Keep its complete execution trajectory in this task state."
        f"\n\nUser: Recent conversation:\n\nWorkspace task:\n{task}\n\nAssistant: <tool_call>"
    )


def records(trajectory: Trajectory, split: str, *, force_budget_answer: bool) -> list[dict[str, Any]]:
    prompt = initial_prompt(trajectory.task)
    rows: list[dict[str, Any]] = []
    for turn, step in enumerate(trajectory.steps, 1):
        response = tool_call(step.command, opening_supplied=turn == 1)
        rows.append(
            record(trajectory, split, turn, "initial_tool" if turn == 1 else "continue_tool", prompt, response)
        )
        prompt += response
        final_tool = turn == len(trajectory.steps)
        prompt += observation(
            step.result,
            turn,
            force_answer=force_budget_answer and final_tool,
        )
    if force_budget_answer:
        response = trajectory.answer + "</answer>"
        task = "budget_answer"
    else:
        response = f"<answer>{trajectory.answer}</answer>"
        task = "final_answer"
    rows.append(record(trajectory, split, len(trajectory.steps) + 1, task, prompt, response))
    return rows


def record(
    trajectory: Trajectory,
    split: str,
    turn: int,
    task: str,
    prompt: str,
    response: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"{trajectory.trajectory_id}::t{turn}",
        "trajectory_id": trajectory.trajectory_id,
        "split": split,
        "dataset": DATASET,
        "task": task,
        "family": trajectory.family,
        "language": trajectory.language,
        "turn": turn,
        "prompt": prompt,
        "response": response,
        "text": prompt + response,
    }


def _token(split: str, index: int, rng: random.Random) -> str:
    return f"{split[:2]}{index:05d}{rng.randrange(16**5):05x}"


def _inspect_rank(split: str, index: int, rng: random.Random, zh: bool) -> Trajectory:
    token = _token(split, index, rng)
    filename = f"metrics_{token}.txt"
    labels = [f"item_{token}_{suffix}" for suffix in ("a", "b", "c")]
    values = rng.sample(range(2, 90), 3)
    winner = max(range(3), key=values.__getitem__)
    content = "".join(f"{label}={value}\n" for label, value in zip(labels, values))
    task = (
        f"只检查 {filename}，回答数值最大的项目名称及其精确数值。不要修改或创建文件。"
        if zh
        else f"Inspect {filename} and report the item with the largest value and its exact value. Do not modify or create files."
    )
    answer = f"{labels[winner]}={values[winner]}"
    return Trajectory(
        f"{split}-{token}", "inspect_rank", "zh" if zh else "en", task, {filename: content},
        (ToolStep(f"cat {shlex.quote(filename)}", tool_result(stdout=content)),), answer,
    )


def _create_exact_json(split: str, index: int, rng: random.Random, zh: bool) -> Trajectory:
    token = _token(split, index, rng)
    filename = f"object_{token}.json"
    payload = {f"key_{token}": rng.randrange(10, 999), "enabled": bool(rng.randrange(2))}
    exact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    task = (
        f"创建 {filename}，内容必须精确为 {exact} 加一个换行；然后用 python3 验证JSON并打印 VALID_JSON。"
        if zh
        else f"Create {filename} with exactly {exact} followed by one newline; then validate it with python3 and print VALID_JSON."
    )
    write = f"printf '%s\\n' {shlex.quote(exact)} > {shlex.quote(filename)}"
    verify = (
        "python3 -c "
        + shlex.quote(f"import json; json.load(open('{filename}')); print('VALID_JSON')")
    )
    return Trajectory(
        f"{split}-{token}", "create_exact_json", "zh" if zh else "en", task, {},
        (ToolStep(write, tool_result()), ToolStep(verify, tool_result(stdout="VALID_JSON\n"))),
        f"{filename} created and validated.",
    )


def _sum_exact_file(split: str, index: int, rng: random.Random, zh: bool) -> Trajectory:
    token = _token(split, index, rng)
    source, target = f"values_{token}.txt", f"total_{token}.txt"
    values = [rng.randrange(1, 60) for _ in range(rng.randrange(3, 7))]
    total = sum(values)
    content = "".join(f"{value}\n" for value in values)
    exact = f"total={total}\n"
    task = (
        f"读取 {source}，把总和以精确内容 total=<数字> 写入 {target}，等号两边不能有空格，并打印该行。"
        if zh
        else f"Read {source}, write the sum to {target} as exact content total=<number> with no spaces around '=', and print that line."
    )
    calculate = (
        f"awk '{{s+=$1}} END{{printf \"total=%d\\n\",s}}' {shlex.quote(source)} "
        f"> {shlex.quote(target)} && cat {shlex.quote(target)}"
    )
    return Trajectory(
        f"{split}-{token}", "sum_exact_file", "zh" if zh else "en", task, {source: content},
        (ToolStep(f"cat {shlex.quote(source)}", tool_result(stdout=content)), ToolStep(calculate, tool_result(stdout=exact))),
        f"total={total}",
    )


def _update_two_files(split: str, index: int, rng: random.Random, zh: bool) -> Trajectory:
    token = _token(split, index, rng)
    first, second = f"alpha_{token}.ini", f"beta_{token}.ini"
    old, new = f"stage_{rng.randrange(10,99)}", f"stage_{rng.randrange(100,999)}"
    fixtures = {first: f"mode={old}\n", second: f"mode={old}\n"}
    task = (
        f"把 {first} 和 {second} 中的 mode={old} 都改为 mode={new}；在一次验证命令中打印两个文件的匹配行。"
        if zh
        else f"Change mode={old} to mode={new} in both {first} and {second}; verify both files in one command that prints their matching lines."
    )
    inspect = f"cat {shlex.quote(first)} {shlex.quote(second)}"
    edit = f"sed -i 's/mode={old}/mode={new}/g' {shlex.quote(first)} {shlex.quote(second)}"
    verify = f"grep -H '^mode={new}$' {shlex.quote(first)} {shlex.quote(second)}"
    verified = f"{first}:mode={new}\n{second}:mode={new}\n"
    return Trajectory(
        f"{split}-{token}", "update_two_files", "zh" if zh else "en", task, fixtures,
        (ToolStep(inspect, tool_result(stdout=f"mode={old}\nmode={old}\n")), ToolStep(edit, tool_result()), ToolStep(verify, tool_result(stdout=verified))),
        "Both requested files were updated and verified." if not zh else "两个指定文件都已更新并验证。",
    )


def _repair_delimiter(split: str, index: int, rng: random.Random, zh: bool) -> Trajectory:
    token = _token(split, index, rng)
    source, test = f"decode_{token}.py", f"check_decode_{token}.py"
    source_text = "def decode(line):\n    return line.strip().split(':', 1)\n"
    test_text = f"from decode_{token} import decode\nassert decode('left=right') == ['left', 'right']\nprint('CHECK_OK')\n"
    task = (
        f"先直接用 python3 运行 {test} 复现失败，再检查 {source} 与测试，修复实现以支持测试中的分隔符，最后再次直接运行测试直到输出 CHECK_OK。"
        if zh
        else f"First run {test} directly with python3 to reproduce the failure, inspect it with {source}, fix the implementation for the tested delimiter, then run the test directly again until it prints CHECK_OK."
    )
    failure = "Traceback (most recent call last):\nValueError: not enough values to unpack\n"
    inspect_out = f"--- {source} ---\n{source_text}--- {test} ---\n{test_text}"
    fixed = "def decode(line):\n    text = line.strip()\n    sep = '=' if '=' in text else ':'\n    return text.split(sep, 1)\n"
    write = f"cat > {shlex.quote(source)} <<'PY'\n{fixed}PY"
    return Trajectory(
        f"{split}-{token}", "repair_delimiter", "zh" if zh else "en", task, {source: source_text, test: test_text},
        (
            ToolStep(f"python3 {shlex.quote(test)}", tool_result(stderr=failure, exit_code=1)),
            ToolStep(f"printf '%s\\n' '--- {source} ---'; cat {source}; printf '%s\\n' '--- {test} ---'; cat {test}", tool_result(stdout=inspect_out)),
            ToolStep(write, tool_result()),
            ToolStep(f"python3 {shlex.quote(test)}", tool_result(stdout="CHECK_OK\n")),
        ),
        "The implementation was fixed and the standalone test prints CHECK_OK." if not zh else "实现已修复，独立测试输出 CHECK_OK。",
    )


def _repair_filter(split: str, index: int, rng: random.Random, zh: bool) -> Trajectory:
    token = _token(split, index, rng)
    source, test = f"select_{token}.py", f"check_select_{token}.py"
    threshold = rng.randrange(3, 20)
    source_text = "def select(values, limit):\n    return [value for value in values if value < limit]\n"
    test_text = (
        f"from select_{token} import select\n"
        f"assert select([{threshold-1}, {threshold}, {threshold+1}], {threshold}) == [{threshold}, {threshold+1}]\n"
        "print('FILTER_OK')\n"
    )
    task = (
        f"先用 python3 直接运行 {test}，失败后检查 {source} 和测试，只修复实现，再次直接运行直到输出 FILTER_OK。"
        if zh
        else f"Run {test} directly with python3 first. After it fails, inspect {source} and the test, fix only the implementation, and rerun directly until it prints FILTER_OK."
    )
    failure = "Traceback (most recent call last):\nAssertionError\n"
    fixed = "def select(values, limit):\n    return [value for value in values if value >= limit]\n"
    write = f"cat > {shlex.quote(source)} <<'PY'\n{fixed}PY"
    inspect_out = f"--- {source} ---\n{source_text}--- {test} ---\n{test_text}"
    return Trajectory(
        f"{split}-{token}", "repair_filter", "zh" if zh else "en", task, {source: source_text, test: test_text},
        (
            ToolStep(f"python3 {shlex.quote(test)}", tool_result(stderr=failure, exit_code=1)),
            ToolStep(f"printf '%s\\n' '--- {source} ---'; cat {source}; printf '%s\\n' '--- {test} ---'; cat {test}", tool_result(stdout=inspect_out)),
            ToolStep(write, tool_result()),
            ToolStep(f"python3 {shlex.quote(test)}", tool_result(stdout="FILTER_OK\n")),
        ),
        "The implementation was fixed and the standalone test prints FILTER_OK." if not zh else "实现已修复，独立测试输出 FILTER_OK。",
    )


def _recover_runtime(split: str, index: int, rng: random.Random, zh: bool) -> Trajectory:
    token = _token(split, index, rng)
    filename = f"payload_{token}.json"
    verifier = f"verify_payload_{token}.py"
    value = rng.randrange(1000, 9999)
    content = json.dumps({"value": value}, separators=(",", ":")) + "\n"
    verifier_content = (
        "from external_payload_parser import load_payload\n"
        f"print(load_payload('{filename}')['value'])\n"
    )
    task = (
        f"先直接用 python3 运行 {verifier}。如果它因缺少依赖失败，不要安装依赖或联网，改用现有标准库读取 {filename} 中的 value 并回答。"
        if zh
        else f"First run {verifier} directly with python3. If it fails because a dependency is missing, do not install anything or use the network; use the available standard library to read value from {filename} and answer."
    )
    return Trajectory(
        f"{split}-{token}", "recover_runtime", "zh" if zh else "en", task, {filename: content, verifier: verifier_content},
        (
            ToolStep(
                f"python3 {shlex.quote(verifier)}",
                tool_result(
                    stderr="ModuleNotFoundError: No module named 'external_payload_parser'\n",
                    exit_code=1,
                ),
            ),
            ToolStep(
                "python3 -c " + shlex.quote(f"import json; print(json.load(open('{filename}'))['value'])"),
                tool_result(stdout=f"{value}\n"),
            ),
        ),
        str(value),
    )


BUILDERS = {
    "inspect_rank": _inspect_rank,
    "create_exact_json": _create_exact_json,
    "sum_exact_file": _sum_exact_file,
    "update_two_files": _update_two_files,
    "repair_delimiter": _repair_delimiter,
    "repair_filter": _repair_filter,
    "recover_runtime": _recover_runtime,
}


def generate(split: str, count: int, seed: int) -> tuple[list[dict[str, Any]], list[Trajectory]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    trajectories: list[Trajectory] = []
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        trajectory = BUILDERS[family](split, index, rng, bool(index % 2))
        trajectories.append(trajectory)
        rows.extend(records(trajectory, split, force_budget_answer=index % 13 == 0))
    return rows, trajectories


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def validate(train: Sequence[dict[str, Any]], dev: Sequence[dict[str, Any]]) -> dict[str, Any]:
    train_ids = {str(row["trajectory_id"]) for row in train}
    dev_ids = {str(row["trajectory_id"]) for row in dev}
    if train_ids & dev_ids:
        raise ValueError("trajectory leakage between Train and Dev")
    train_pairs = {(str(row["prompt"]), str(row["response"])) for row in train}
    dev_pairs = {(str(row["prompt"]), str(row["response"])) for row in dev}
    if train_pairs & dev_pairs:
        raise ValueError("exact prompt/response leakage between Train and Dev")
    for split, rows in (("train", train), ("dev", dev)):
        if not rows:
            raise ValueError(f"empty {split} rows")
        for row in rows:
            response = str(row["response"])
            if row["task"] == "initial_tool":
                if not response.startswith("{") or not response.endswith("</tool_call>"):
                    raise ValueError(f"invalid initial response: {row['id']}")
            elif row["task"] in {"continue_tool"}:
                if not response.startswith("<tool_call>") or not response.endswith("</tool_call>"):
                    raise ValueError(f"invalid tool response: {row['id']}")
            elif row["task"] == "final_answer":
                if not response.startswith("<answer>") or not response.endswith("</answer>"):
                    raise ValueError(f"invalid answer response: {row['id']}")
            elif row["task"] == "budget_answer":
                if response.startswith("<answer>") or not response.endswith("</answer>"):
                    raise ValueError(f"invalid prefixed answer response: {row['id']}")
    return {
        "train_trajectory_ids": len(train_ids),
        "dev_trajectory_ids": len(dev_ids),
        "trajectory_overlap": 0,
        "exact_example_overlap": 0,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "trajectories": len({row["trajectory_id"] for row in rows}),
        "families": dict(sorted(Counter(row["family"] for row in rows).items())),
        "tasks": dict(sorted(Counter(row["task"] for row in rows).items())),
        "languages": dict(sorted(Counter(row["language"] for row in rows).items())),
        "max_prompt_chars": max(map(lambda row: len(str(row["prompt"])), rows)),
        "max_response_chars": max(map(lambda row: len(str(row["response"])), rows)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-trajectories", type=int, default=2100)
    parser.add_argument("--dev-trajectories", type=int, default=350)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args(argv)
    if args.train_trajectories < len(FAMILIES) or args.dev_trajectories < len(FAMILIES):
        parser.error("each split must cover every family")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    train, _ = generate("train", args.train_trajectories, args.seed)
    dev, _ = generate("dev", args.dev_trajectories, args.seed + 1)
    leakage = validate(train, dev)
    train_path, dev_path = output / "train.jsonl", output / "dev.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "seed": args.seed,
        "benchmark_inputs": [],
        "principle": "randomized procedural trajectories; frozen benchmark Gold is not read",
        "train": summarize(train) | {"path": str(train_path), "sha256": sha256(train_path)},
        "dev": summarize(dev) | {"path": str(dev_path), "sha256": sha256(dev_path)},
        "leakage_audit": leakage,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
