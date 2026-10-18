from pathlib import Path

from benchmarks.run_long_horizon_agent_bench import (
    load_cases,
    prepare,
    summarize,
    task_spec,
    verify,
)


DATASET = Path(__file__).parents[1] / "benchmarks/long_horizon_agent_v1.jsonl"


def test_dataset_is_multistage_and_bilingual() -> None:
    cases = load_cases(DATASET)
    assert len(cases) == 3
    assert {case["language"] for case in cases} == {"en", "zh"}
    assert all(len(case["stages"]) >= 3 for case in cases)
    assert all(case["expect"]["min_actions"] >= 7 for case in cases)


def test_task_spec_arms_only_add_explicit_stages_to_staged_arm() -> None:
    case = load_cases(DATASET)[0]
    flat = task_spec(case, "flat", "long-horizon/flat/case")
    staged = task_spec(case, "staged", "long-horizon/staged/case")
    assert flat["stages"] == []
    assert len(staged["stages"]) == 3
    assert flat["objective"] != staged["objective"]
    assert flat["working_directory"].startswith("long-horizon/flat/")
    assert staged["working_directory"].startswith("long-horizon/staged/")


def _successful_stage(stage_id: str, command: str, stdout: str = "") -> dict:
    return {
        "spec": {"id": stage_id},
        "status": "succeeded",
        "attempts": 1,
        "response": {
            "status": "ok",
            "answer": "done",
            "route": {"mode": "tool_loop", "strict": True},
            "trace": {
                "agent": {
                    "tool_steps": [
                        {
                            "name": "run_command",
                            "arguments": {"command": command},
                            "result": {
                                "status": "ok",
                                "exit_code": 0,
                                "stdout": stdout,
                            },
                        }
                    ],
                    "events": [{"type": "state_released", "success": True}],
                }
            },
        },
    }


def test_verify_requires_all_stage_releases_and_protected_inputs(tmp_path: Path) -> None:
    case = load_cases(DATASET)[0]
    root = tmp_path / "case"
    original = prepare(case, root)
    (root / "inventory.py").write_text(
        "def parse(line):\n"
        "    name, qty = line.strip().split('=', 1)\n"
        "    return name, int(qty)\n\n"
        "def total(lines):\n"
        "    return sum(parse(line)[1] for line in lines)\n"
    )
    (root / "REPORT.md").write_text("status=pass\ntotal=14\n")
    stages = [
        _successful_stage("one", "cat inventory.py"),
        _successful_stage("two", "python3 -m unittest tests.test_total -v"),
        _successful_stage("three", "python3 verify.py", "LONG_OK\n"),
    ]
    ledger = {"task": {"status": "succeeded", "stages": stages}}
    result = verify(case, root, original, 200, {"status": "ok"}, ledger)
    assert result["artifacts_ok"] is True
    assert result["protected_inputs_ok"] is True
    assert result["lifecycle_ok"] is True
    assert result["action_budget_ok"] is False

    stages[-1]["response"]["trace"]["agent"]["events"] = []
    result = verify(case, root, original, 200, {"status": "ok"}, ledger)
    assert result["lifecycle_ok"] is False


def test_summary_compares_flat_and_staged() -> None:
    base = {
        "case_id": "case",
        "elapsed_ms": 100.0,
        "passed": True,
        "artifacts_ok": True,
        "protocol_valid": True,
        "lifecycle_ok": True,
        "action_count": 9,
        "adjacent_repeats": 0,
    }
    summary = summarize([{**base, "arm": "flat"}, {**base, "arm": "staged"}])
    assert summary["flat"]["success_rate"] == 1.0
    assert summary["staged"]["success_rate"] == 1.0
    assert summary["comparison"]["protocol_no_regression"] is True
