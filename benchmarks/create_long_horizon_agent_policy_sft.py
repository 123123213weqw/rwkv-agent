 #!/usr/bin/env python3
"""Generate executable, benchmark-independent long-horizon Agent Policy SFT.

The generator never reads public or frozen benchmark cases.  It creates
deterministic workspace trajectories whose tool observations are known from the
generated fixtures.  Every trajectory requires 12--28 ordered tool turns and
ends in an independent verification command followed by a bounded answer.
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
from typing import Any, Callable, Iterable, Sequence


SCHEMA_VERSION = "rwkv-agent-long-horizon-policy-sft.v1"
DATASET = "long_horizon_agent_policy"
MAX_TOOL_STEPS = 32
FAMILIES = (
    "sequential_manifest",
    "multi_bug_repair",
    "checkpointed_pipeline",
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
    "new isolated process. There is no network or package installation. Python is "
    "available only as `python3`. Preserve the ordered task plan across the entire "
    f"trajectory. You have at most {MAX_TOOL_STEPS} tool calls. Never skip a named "
    "inspection, mutation, or verification stage. If a command fails, use its real "
    "result and continue from the first incomplete stage instead of restarting or "
    "repeating the same successful command. Treat exact content as a byte contract. "
    "After the final independent verification succeeds, answer on the next turn. "
    "Function: run_command(command)."
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


def _tool_call(command: str, *, opening_supplied: bool) -> str:
    payload = json.dumps(
        {"name": "run_command", "arguments": {"command": command}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return payload + "</tool_call>" if opening_supplied else f"<tool_call>{payload}</tool_call>"


def _observation(result: dict[str, Any], completed: int, task: str) -> str:
    compact = json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    instruction = (
        f"Original task (authoritative): {task}\n"
        f"Progress ledger: tool step {completed}/{MAX_TOOL_STEPS} completed. Preserve every "
        "already completed ordered stage. Continue from the first incomplete named stage; "
        "do not repeat an identical successful command. Call exactly one tool if work or "
        "verification remains, otherwise return the final answer."
    )
    suffix = "Assistant: <answer>" if completed >= MAX_TOOL_STEPS else "Assistant:"
    return (
        f"\n\nTool: <tool_result>{compact}</tool_result>\n\n"
        f"User: {instruction} Output exactly one protocol envelope and no reasoning.\n\n{suffix}"
    )


def _initial_prompt(task: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\nSystem: The next User message supplies the current workspace "
        "task. Keep its complete execution trajectory in this task state."
        f"\n\nUser: Recent conversation:\n\nWorkspace task:\n{task}\n\nAssistant: <tool_call>"
    )


def _records(trajectory: Trajectory, split: str) -> list[dict[str, Any]]:
    prompt = _initial_prompt(trajectory.task)
    rows: list[dict[str, Any]] = []
    for turn, step in enumerate(trajectory.steps, 1):
        response = _tool_call(step.command, opening_supplied=turn == 1)
        rows.append(_record(trajectory, split, turn, "initial_tool" if turn == 1 else "continue_tool", prompt, response))
        prompt += response + _observation(step.result, turn, trajectory.task)
    rows.append(
        _record(
            trajectory,
            split,
            len(trajectory.steps) + 1,
            "final_answer",
            prompt,
            f"<answer>{trajectory.answer}</answer>",
        )
    )
    return rows


def _record(
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
        "horizon": len(trajectory.steps),
        "turn": turn,
        "prompt": prompt,
        "response": response,
        "text": prompt + response,
    }


def _token(split: str, index: int, rng: random.Random) -> str:
    return f"{split[:2]}{index:05d}{rng.randrange(16**6):06x}"


def _sequential_manifest(split: str, index: int, rng: random.Random, zh: bool) -> Trajectory:
    token = _token(split, index, rng)
    count = rng.randrange(4, 8)
    names = [f"part_{token}_{i}.txt" for i in range(1, count + 1)]
    outputs = [f"done_{token}_{i}.txt" for i in range(1, count + 1)]
    values = [rng.randrange(10, 999) for _ in names]
    manifest_name = f"manifest_{token}.txt"
    manifest = "".join(f"{name}->{output}\n" for name, output in zip(names, outputs))
    fixtures = {manifest_name: manifest}
    fixtures.update({name: f"value={value}\n" for name, value in zip(names, values)})
    task = (
        f"按顺序完成长任务：先单独读取 {manifest_name}；再按清单顺序用独立命令读取每个输入文件；"
        "然后按同一顺序用独立命令创建对应输出，精确内容为 processed=<原数字> 加换行；"
        "再逐个用独立命令读回输出；最后用一个 python3 命令验证全部映射并打印 ALL_PARTS_OK。"
        if zh
        else f"Complete this ordered long task: read {manifest_name} alone; then inspect every input in manifest order with a separate command; create each mapped output in the same order with exact content processed=<original number> plus a newline using separate commands; read every output back separately; finally validate every mapping with one python3 command that prints ALL_PARTS_OK."
    )
    steps: list[ToolStep] = [ToolStep(f"cat {shlex.quote(manifest_name)}", tool_result(stdout=manifest))]
    steps.extend(
        ToolStep(f"cat {shlex.quote(name)}", tool_result(stdout=f"value={value}\n"))
        for name, value in zip(names, values)
    )
    steps.extend(
        ToolStep(
            f"printf 'processed=%s\\n' {value} > {shlex.quote(output)}",
            tool_result(),
        )
        for output, value in zip(outputs, values)
    )
    steps.extend(
        ToolStep(f"cat {shlex.quote(output)}", tool_result(stdout=f"processed={value}\n"))
        for output, value in zip(outputs, values)
    )
    expected = {output: f"processed={value}\n" for output, value in zip(outputs, values)}
    script = f"expected={expected!r}; assert all(open(k).read()==v for k,v in expected.items()); print('ALL_PARTS_OK')"
    steps.append(ToolStep("python3 -c " + shlex.quote(script), tool_result(stdout="ALL_PARTS_OK\n")))
    return Trajectory(
        f"{split}-{token}",
        "sequential_manifest",
        "zh" if zh else "en",
        task,
        fixtures,
        tuple(steps),
        f"Processed and independently verified {count} manifest entries.",
    )


def _multi_bug_repair(split: str, index: int, rng: random.Random, zh: bool) -> Trajectory:
    token = _token(split, index, rng)
    count = rng.randrange(3, 6)
    fixtures: dict[str, str] = {}
    steps: list[ToolStep] = []
    tests: list[str] = []
    for number in range(1, count + 1):
        module = f"calc_{token}_{number}.py"
        test = f"test_calc_{token}_{number}.py"
        offset = rng.randrange(2, 20)
        fixtures[module] = "def adjust(value, offset):\n    return value - offset\n"
        fixtures[test] = (
            f"from calc_{token}_{number} import adjust\n"
            f"assert adjust(10, {offset}) == {10 + offset}\n"
            f"print('CASE_{number}_OK')\n"
        )
        tests.append(test)
        steps.extend(
            [
                ToolStep(
                    f"python3 {shlex.quote(test)}",
                    tool_result(stderr="Traceback (most recent call last):\nAssertionError\n", exit_code=1),
                ),
                ToolStep(
                    f"printf '%s\\n' '--- {module} ---'; cat {module}; printf '%s\\n' '--- {test} ---'; cat {test}",
                    tool_result(stdout=f"--- {module} ---\n{fixtures[module]}--- {test} ---\n{fixtures[test]}"),
                ),
                ToolStep(
                    f"printf '%s\\n' 'def adjust(value, offset):' '    return value + offset' > {shlex.quote(module)}",
                    tool_result(),
                ),
                ToolStep(f"python3 {shlex.quote(test)}", tool_result(stdout=f"CASE_{number}_OK\n")),
            ]
        )
    joined = " && ".join(f"python3 {shlex.quote(test)}" for test in tests)
    steps.append(ToolStep(joined, tool_result(stdout="".join(f"CASE_{i}_OK\n" for i in range(1, count + 1)))))
    task = (
        f"按编号顺序修复 {count} 个独立模块。对每个编号必须依次：先直接运行对应测试复现失败，"
        "再在一个命令中读取该源码和测试，只修改该源码，然后重新运行该测试直到通过；一个编号完成后再处理下一个。"
        "最后用一个命令重新运行全部测试。不要修改测试。"
        if zh
        else f"Repair {count} independent modules in numeric order. For each number: first run its test to reproduce the failure, inspect that source and test together, modify only that source, and rerun that test until it passes before moving to the next number. Finally rerun every test in one command. Never modify tests."
    )
    return Trajectory(
        f"{split}-{token}",
        "multi_bug_repair",
        "zh" if zh else "en",
        task,
        fixtures,
        tuple(steps),
        f"Repaired {count} modules and the complete ordered test suite passes.",
    )


def _checkpointed_pipeline(split: str, index: int, rng: random.Random, zh: bool) -> Trajectory:
    token = _token(split, index, rng)
    count = rng.randrange(6, 11)
    source = f"events_{token}.txt"
    checkpoint = f"checkpoint_{token}.txt"
    summary = f"summary_{token}.json"
    values = [rng.randrange(1, 50) for _ in range(count)]
    fixtures = {source: "".join(f"stage{i}={value}\n" for i, value in enumerate(values, 1))}
    task = (
        f"执行可恢复流水线：先读取 {source}；然后按 stage1 到 stage{count} 的顺序，每一步都用独立命令把"
        f"精确行 stageN=<值> 追加到 {checkpoint}，并在紧接着的独立命令中检查当前最后一行；"
        f"全部阶段完成后创建 {summary}，JSON 等价于 stages={count}, total={sum(values)}；最后用 python3"
        "同时验证 checkpoint 的完整顺序和 JSON，并打印 PIPELINE_OK。"
        if zh
        else f"Execute a resumable pipeline: inspect {source}; then process stage1 through stage{count} in order, appending the exact stageN=<value> line to {checkpoint} with one command per stage and checking the current last line with the immediately following separate command. After all stages, create {summary} as JSON equivalent to stages={count}, total={sum(values)}; finally validate both checkpoint order and JSON with python3 and print PIPELINE_OK."
    )
    steps: list[ToolStep] = [ToolStep(f"cat {shlex.quote(source)}", tool_result(stdout=fixtures[source]))]
    checkpoint_text = ""
    for stage, value in enumerate(values, 1):
        line = f"stage{stage}={value}\n"
        checkpoint_text += line
        steps.append(ToolStep(f"printf 'stage%s=%s\\n' {stage} {value} >> {shlex.quote(checkpoint)}", tool_result()))
        steps.append(ToolStep(f"tail -n 1 {shlex.quote(checkpoint)}", tool_result(stdout=line)))
    payload = {"stages": count, "total": sum(values)}
    exact_json = json.dumps(payload, separators=(",", ":")) + "\n"
    steps.append(ToolStep(f"printf '%s\\n' {shlex.quote(exact_json.rstrip())} > {shlex.quote(summary)}", tool_result()))
    script = (
        f"import json; assert open('{checkpoint}').read()=={checkpoint_text!r}; "
        f"assert json.load(open('{summary}'))=={payload!r}; print('PIPELINE_OK')"
    )
    steps.append(ToolStep("python3 -c " + shlex.quote(script), tool_result(stdout="PIPELINE_OK\n")))
    return Trajectory(
        f"{split}-{token}",
        "checkpointed_pipeline",
        "zh" if zh else "en",
        task,
        fixtures,
        tuple(steps),
        f"Completed and verified all {count} ordered pipeline stages.",
    )


BUILDERS: dict[str, Callable[[str, int, random.Random, bool], Trajectory]] = {
    "sequential_manifest": _sequential_manifest,
    "multi_bug_repair": _multi_bug_repair,
    "checkpointed_pipeline": _checkpointed_pipeline,
}


def generate(split: str, count: int, seed: int) -> tuple[list[dict[str, Any]], list[Trajectory]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    trajectories: list[Trajectory] = []
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        trajectory = BUILDERS[family](split, index, rng, bool(index % 2))
        if not 12 <= len(trajectory.steps) <= MAX_TOOL_STEPS:
            raise ValueError(f"trajectory horizon out of bounds: {trajectory.trajectory_id}")
        trajectories.append(trajectory)
        rows.extend(_records(trajectory, split))
    return rows, trajectories


def validate(train: Sequence[dict[str, Any]], dev: Sequence[dict[str, Any]]) -> dict[str, Any]:
    train_ids = {str(row["trajectory_id"]) for row in train}
    dev_ids = {str(row["trajectory_id"]) for row in dev}
    if train_ids & dev_ids:
        raise ValueError("trajectory leakage between Train and Dev")
    train_pairs = {(str(row["prompt"]), str(row["response"])) for row in train}
    dev_pairs = {(str(row["prompt"]), str(row["response"])) for row in dev}
    if train_pairs & dev_pairs:
        raise ValueError("exact example leakage between Train and Dev")
    for rows in (train, dev):
        if not rows:
            raise ValueError("empty split")
        for row in rows:
            response = str(row["response"])
            if row["task"] == "initial_tool":
                valid = response.startswith("{") and response.endswith("</tool_call>")
            elif row["task"] == "continue_tool":
                valid = response.startswith("<tool_call>") and response.endswith("</tool_call>")
            else:
                valid = response.startswith("<answer>") and response.endswith("</answer>")
            if not valid:
                raise ValueError(f"invalid protocol response: {row['id']}")
    return {
        "train_trajectory_ids": len(train_ids),
        "dev_trajectory_ids": len(dev_ids),
        "trajectory_overlap": 0,
        "exact_example_overlap": 0,
    }


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    horizons = {str(row["trajectory_id"]): int(row["horizon"]) for row in rows}
    return {
        "records": len(rows),
        "trajectories": len(horizons),
        "families": dict(sorted(Counter(row["family"] for row in rows).items())),
        "tasks": dict(sorted(Counter(row["task"] for row in rows).items())),
        "languages": dict(sorted(Counter(row["language"] for row in rows).items())),
        "horizon_min": min(horizons.values()),
        "horizon_max": max(horizons.values()),
        "horizon_mean": round(sum(horizons.values()) / len(horizons), 3),
        "max_prompt_chars": max(len(str(row["prompt"])) for row in rows),
        "max_response_chars": max(len(str(row["response"])) for row in rows),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-trajectories", type=int, default=900)
    parser.add_argument("--dev-trajectories", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args(argv)
    if args.train_trajectories < len(FAMILIES) or args.dev_trajectories < len(FAMILIES):
        parser.error("each split must cover every family")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    train, _ = generate("train", args.train_trajectories, args.seed)
    dev, _ = generate("dev", args.dev_trajectories, args.seed + 1)
    leakage = validate(train, dev)
    train_path = output / "train.jsonl"
    dev_path = output / "dev.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(dev_path, dev)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "seed": args.seed,
        "benchmark_inputs": [],
        "failure_trace_inputs": [],
        "principle": "executable randomized trajectories; no benchmark Gold or failed rollout is read",
        "train": _summary(train) | {"path": str(train_path), "sha256": sha256(train_path)},
        "dev": _summary(dev) | {"path": str(dev_path), "sha256": sha256(dev_path)},
        "leakage_audit": leakage,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
