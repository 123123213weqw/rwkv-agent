from __future__ import annotations

import json
from pathlib import Path

from benchmarks.compose_agent_policy_curriculum import main
from benchmarks.create_general_agent_policy_sft import main as short_main
from benchmarks.create_long_horizon_agent_policy_sft import main as long_main


def test_curriculum_composes_only_generated_disjoint_splits(tmp_path: Path) -> None:
    short = tmp_path / "short"
    long = tmp_path / "long"
    output = tmp_path / "curriculum"
    assert short_main(["--output-dir", str(short), "--train-trajectories", "7", "--dev-trajectories", "7", "--seed", "71"]) == 0
    assert long_main(["--output-dir", str(long), "--train-trajectories", "3", "--dev-trajectories", "3", "--seed", "72"]) == 0

    assert main(["--short-dir", str(short), "--long-dir", str(long), "--output-dir", str(output)]) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["benchmark_inputs"] == []
    assert manifest["failure_trace_inputs"] == []
    assert manifest["leakage_audit"]["trajectory_overlap"] == 0
    assert set(manifest["train"]["datasets"]) == {"general_agent_policy", "long_horizon_agent_policy"}
    train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len({row["id"] for row in train}) == len(train)
