from pathlib import Path

import pytest

from benchmarks.run_command_state_loop_bench import (
    CASES,
    execute_command,
    parse_action,
    render_root_prompt,
    summarize,
    validate_command,
)


def test_parse_action_accepts_only_exact_tool_or_answer() -> None:
    tool = parse_action(
        '<tool_call>{"name":"run_command","arguments":{"command":"cat note.txt"}}</tool_call>'
    )
    assert tool == {"kind": "tool", "command": "cat note.txt", "error": ""}
    answer = parse_action("<answer>alpha is 7.</answer>")
    assert answer["kind"] == "answer"
    assert answer["answer"] == "alpha is 7."
    assert parse_action("reasoning\n<answer>7</answer>")["kind"] == "invalid"


@pytest.mark.parametrize(
    "command",
    [
        "cat ../secret",
        "curl https://example.com",
        "sudo cat note.txt",
        "cat /etc/passwd",
        "cd /testbed && ls",
        "python -c 'print(1)'\x00",
    ],
)
def test_validate_command_rejects_escape_and_external_actions(command: str) -> None:
    with pytest.raises(ValueError):
        validate_command(command)


def test_execute_command_is_bounded_to_fixture_directory(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("alpha=7\n")
    result = execute_command("cat note.txt", cwd=tmp_path)
    assert result["status"] == "ok"
    assert result["exit_code"] == 0
    assert result["stdout"] == "alpha=7\n"


def test_root_prompt_has_optional_complete_loop_example() -> None:
    case = CASES[0]
    plain = render_root_prompt(case, style="instruction_only")
    example = render_root_prompt(case, style="one_loop_example")
    assert "run_command(command)" in plain
    assert "demo.txt" not in plain
    assert "demo.txt" in example
    assert "<tool_result>" in example
    assert example.endswith(case.task)


def test_summary_reports_protocol_state_and_release() -> None:
    rows = []
    for repeat in range(2):
        rows.append(
            {
                "style": "one_loop_example",
                "case_id": "case",
                "repeat": repeat,
                "steps": [{"command": "cat note.txt"}],
                "answer": "7",
                "terminal": "answer",
                "passed": True,
                "protocol_valid": True,
                "verified": True,
                "minimum_actions_met": True,
                "state_constant": True,
                "released": True,
                "action_count": 1,
                "elapsed_ms": 10.0,
            }
        )
    summary = summarize(rows)["styles"]["one_loop_example"]
    assert summary["passed"] == 2
    assert summary["state_constant"] == 2
    assert summary["released"] == 2
    assert summary["repeat_command_answer_exact"] == 1
