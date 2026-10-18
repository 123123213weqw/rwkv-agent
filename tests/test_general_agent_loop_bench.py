from pathlib import Path

from benchmarks.run_general_agent_loop_bench import (
    _normalize_commands,
    _verify_case,
    load_cases,
    prepare_workspace,
    summarize,
)


DATASET = Path(__file__).parents[1] / "benchmarks/general_agent_loop_v1.jsonl"


def test_frozen_dataset_has_balanced_30_case_matrix() -> None:
    cases = load_cases(DATASET)
    assert len(cases) == 30
    assert len({case.id for case in cases}) == 30
    assert sum(case.language == "zh" for case in cases) == 15
    assert sum(case.language == "en" for case in cases) == 15
    assert {case.category for case in cases} == {
        "create",
        "transform",
        "analyze",
        "fix_test",
        "multi_file",
        "inspect",
    }
    assert all(
        sum(candidate.category == category for candidate in cases) == 5
        for category in {case.category for case in cases}
    )


def test_prepare_workspace_replaces_only_selected_run(tmp_path: Path) -> None:
    case = load_cases(DATASET)[2]
    root = prepare_workspace(case, workspace_base=tmp_path, run_dir="runs/c1/case")
    assert (root / "source.txt").read_text() == "delta=29\n"
    (root / "stale.txt").write_text("stale")
    root = prepare_workspace(case, workspace_base=tmp_path, run_dir="runs/c1/case")
    assert not (root / "stale.txt").exists()
    assert (root / "source.txt").read_text() == "delta=29\n"


def test_verify_case_requires_artifact_protocol_and_release(tmp_path: Path) -> None:
    case = load_cases(DATASET)[0]
    root = prepare_workspace(case, workspace_base=tmp_path, run_dir="runs/c1/case")
    (root / "output.txt").write_text("agent-ready\n")
    response = {
        "status": "ok",
        "answer": "完成",
        "route": {"mode": "tool_loop", "strict": True, "steps": 2},
        "trace": {
            "agent": {
                "tool_steps": [
                    {
                        "name": "run_command",
                        "arguments": {"command": "printf agent-ready > output.txt"},
                        "result": {"status": "ok", "exit_code": 0, "stdout": ""},
                    },
                    {
                        "name": "run_command",
                        "arguments": {"command": "cat output.txt"},
                        "result": {
                            "status": "ok",
                            "exit_code": 0,
                            "stdout": "agent-ready\n",
                        },
                    },
                ],
                "events": [{"type": "state_released", "success": True}],
            }
        },
    }
    verified = _verify_case(
        case,
        root=root,
        response=response,
        http_status=200,
        max_steps=8,
    )
    assert verified["passed"] is True
    response["answer"] = "<tool_result>leak</tool_result>"
    leaked = _verify_case(
        case,
        root=root,
        response=response,
        http_status=200,
        max_steps=8,
    )
    assert leaked["passed"] is False
    assert leaked["no_protocol_leak"] is False


def _row(case_id: str, profile: str, *, passed: bool = True) -> dict[str, object]:
    return {
        "case_id": case_id,
        "category": "fix_test" if case_id.startswith("fix") else "create",
        "language": "en",
        "profile": profile,
        "run_dir": f"runs/{profile}/{case_id}",
        "http_status": 200,
        "elapsed_ms": 1000.0,
        "passed": passed,
        "protocol_valid": True,
        "tool_only": True,
        "state_released": True,
        "no_protocol_leak": True,
        "artifacts_verified": passed,
        "minimum_actions_met": True,
        "step_budget_ok": True,
        "action_count": 1,
        "commands": [f"cat runs/{profile}/{case_id}/note.txt"],
        "answer": "done",
    }


def test_summary_reports_cross_profile_gate_and_normalizes_paths() -> None:
    rows = []
    for profile in ("c1", "c4"):
        rows.extend(_row(f"case-{index}", profile) for index in range(5))
        rows.extend(_row(f"fix-{index}", profile) for index in range(5))
    summary = summarize(rows, max_p95_ms=30000.0)
    assert summary["mvp_pass"] is True
    assert summary["candidate_pass"] is True
    assert summary["cross_profile"]["command_answer_exact"] == 10
    assert _normalize_commands(rows[0]) == ("cat {run_dir}/note.txt",)
