from __future__ import annotations

import json
from pathlib import Path

from benchmarks.create_general_agent_policy_sft import (
    FAMILIES,
    generate,
    main,
    validate,
)


def test_generator_is_deterministic_and_covers_policy_families() -> None:
    first, _ = generate("train", len(FAMILIES) * 2, 17)
    second, _ = generate("train", len(FAMILIES) * 2, 17)

    assert first == second
    assert {row["family"] for row in first} == set(FAMILIES)
    assert {row["task"] for row in first} >= {
        "initial_tool",
        "continue_tool",
        "final_answer",
        "budget_answer",
    }


def test_train_dev_are_disjoint_and_protocol_is_strict() -> None:
    train, _ = generate("train", len(FAMILIES) * 2, 23)
    dev, _ = generate("dev", len(FAMILIES) * 2, 24)

    audit = validate(train, dev)

    assert audit["trajectory_overlap"] == 0
    assert audit["exact_example_overlap"] == 0
    assert all(row["dataset"] == "general_agent_policy" for row in train + dev)


def test_cli_writes_hashed_manifest_without_benchmark_inputs(tmp_path: Path) -> None:
    output = tmp_path / "policy"

    assert main([
        "--output-dir",
        str(output),
        "--train-trajectories",
        str(len(FAMILIES)),
        "--dev-trajectories",
        str(len(FAMILIES)),
        "--seed",
        "31",
    ]) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["benchmark_inputs"] == []
    assert manifest["leakage_audit"]["trajectory_overlap"] == 0
    assert manifest["train"]["sha256"]
    assert manifest["dev"]["sha256"]
    assert len((output / "train.jsonl").read_text().splitlines()) == manifest["train"]["records"]
