from __future__ import annotations

import json
from pathlib import Path

from benchmarks.create_long_horizon_agent_policy_sft import (
    FAMILIES,
    MAX_TOOL_STEPS,
    generate,
    main,
    validate,
)


def test_long_horizon_generator_is_deterministic_and_bounded() -> None:
    first, trajectories = generate("train", len(FAMILIES) * 2, 41)
    second, _ = generate("train", len(FAMILIES) * 2, 41)

    assert first == second
    assert {row["family"] for row in first} == set(FAMILIES)
    assert all(12 <= len(value.steps) <= MAX_TOOL_STEPS for value in trajectories)
    assert all(row["horizon"] == len(next(value.steps for value in trajectories if value.trajectory_id == row["trajectory_id"])) for row in first)


def test_long_horizon_train_dev_are_disjoint_and_strict() -> None:
    train, _ = generate("train", len(FAMILIES), 51)
    dev, _ = generate("dev", len(FAMILIES), 52)

    audit = validate(train, dev)

    assert audit["trajectory_overlap"] == 0
    assert audit["exact_example_overlap"] == 0
    assert {row["task"] for row in train} == {"initial_tool", "continue_tool", "final_answer"}


def test_cli_writes_hashed_manifest_without_eval_or_trace_inputs(tmp_path: Path) -> None:
    output = tmp_path / "long-policy"

    assert main([
        "--output-dir",
        str(output),
        "--train-trajectories",
        str(len(FAMILIES)),
        "--dev-trajectories",
        str(len(FAMILIES)),
        "--seed",
        "61",
    ]) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["benchmark_inputs"] == []
    assert manifest["failure_trace_inputs"] == []
    assert manifest["train"]["horizon_min"] >= 12
    assert manifest["train"]["horizon_max"] <= MAX_TOOL_STEPS
    assert manifest["train"]["sha256"]
    assert manifest["dev"]["sha256"]
