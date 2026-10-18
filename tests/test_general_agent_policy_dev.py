import json
from pathlib import Path

import pytest

from benchmarks.run_general_agent_policy_dev import classify, load_resume_rows, load_rows, summarize


@pytest.mark.parametrize(
    ("task", "text", "valid", "action"),
    [
        ("initial_tool", '{"name":"run_command","arguments":{"command":"pwd"}}</tool_call>', True, "tool"),
        ("continue_tool", '<tool_call>{"name":"run_command","arguments":{"command":"pwd"}}</tool_call>', True, "tool"),
        ("final_answer", "<answer>done</answer>", True, "answer"),
        ("budget_answer", "done</answer>", True, "answer"),
        ("continue_tool", '<tool_call>{"name":"run_command","arguments":{"command":""}}</tool_call>', False, "invalid"),
        ("final_answer", "reasoning <answer>done</answer>", False, "invalid"),
    ],
)
def test_classify_prefix_contract(task: str, text: str, valid: bool, action: str) -> None:
    result = classify(task, text, "</s>", max_tokens=128, token_count=8)
    assert result["strict_envelope"] is valid
    assert result["actual_action"] == action
    assert result["stopped_before_limit"] is True


def test_load_rows_rejects_duplicate_ids(tmp_path: Path) -> None:
    row = {
        "id": "dev-1::t1",
        "trajectory_id": "dev-1",
        "family": "inspect_rank",
        "language": "en",
        "task": "initial_tool",
        "prompt": "prompt",
        "response": "response",
    }
    path = tmp_path / "dev.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_rows(path)


def test_summary_preserves_task_family_language_breakdowns() -> None:
    rows = [
        {
            "task": "initial_tool",
            "family": "inspect_rank",
            "language": "en",
            "strict_envelope": True,
            "expected_action": True,
            "nonempty": True,
            "no_reasoning_or_role_leak": True,
            "stopped_before_limit": True,
            "exact_response": False,
            "elapsed_ms": 10.0,
            "stop_reason": "</s>",
        },
        {
            "task": "final_answer",
            "family": "inspect_rank",
            "language": "zh",
            "strict_envelope": False,
            "expected_action": False,
            "nonempty": True,
            "no_reasoning_or_role_leak": True,
            "stopped_before_limit": False,
            "exact_response": False,
            "elapsed_ms": 30.0,
            "stop_reason": "max_tokens",
        },
    ]
    result = summarize(rows)
    assert result["strict_envelope_rate"] == 0.5
    assert result["p95_elapsed_ms"] == 30.0
    assert result["tasks"]["initial_tool"]["strict_envelope_rate"] == 1.0
    assert result["languages"]["zh"]["stopped_before_limit_rate"] == 0.0


def test_load_resume_rows_requires_contiguous_matching_prefix(tmp_path: Path) -> None:
    dataset = tmp_path / "dev.jsonl"
    rows = [
        {
            "id": f"dev-1::t{index}",
            "trajectory_id": "dev-1",
            "family": "inspect_rank",
            "language": "en",
            "task": "initial_tool",
            "prompt": f"prompt-{index}",
            "response": "response",
        }
        for index in (1, 2)
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "result.json"
    output.write_text(
        json.dumps(
            {
                "schema": "rwkv-agent-policy-dev-eval.v1",
                "dataset": {
                    "sha256": __import__("hashlib").sha256(dataset.read_bytes()).hexdigest(),
                    "rows": 2,
                },
                "endpoint": "http://127.0.0.1:8517",
                "max_tokens": 128,
                "elapsed_seconds": 1.5,
                "rows": [{"id": "dev-1::t1"}],
            }
        )
    )
    predictions, elapsed = load_resume_rows(
        output,
        dataset=dataset,
        rows=rows,
        endpoint="http://127.0.0.1:8517",
        max_tokens=128,
    )
    assert predictions == [{"id": "dev-1::t1"}]
    assert elapsed == 1.5

    value = json.loads(output.read_text())
    value["rows"] = [{"id": "dev-1::t2"}]
    output.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="contiguous"):
        load_resume_rows(
            output,
            dataset=dataset,
            rows=rows,
            endpoint="http://127.0.0.1:8517",
            max_tokens=128,
        )
