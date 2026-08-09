from benchmarks.run_native_tool_prior_probe import (
    ARMS,
    CASES,
    evaluate,
    parse_tool_call,
    render_prompt,
    semantic_valid,
    summarize,
)


def test_parse_tool_call_requires_exact_envelope() -> None:
    parsed = parse_tool_call(
        '<tool_call>{"name":"run_command","arguments":{"command":"pytest -q"}}</tool_call>'
    )
    assert parsed == {
        "strict": True,
        "tool": "run_command",
        "arguments": {"command": "pytest -q"},
        "error": "",
    }
    assert not parse_tool_call("prefix " + '<tool_call>{"name":"x","arguments":{}}</tool_call>')[
        "strict"
    ]


def test_native_trio_uses_task_specific_tools() -> None:
    arm = next(value for value in ARMS if value.name == "native_trio")
    cases = {value["kind"]: value for value in CASES}
    rows = {
        "read": '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>',
        "write": '<tool_call>{"name":"write_file","arguments":{"path":"note.txt","content":"hello"}}</tool_call>',
        "run": '<tool_call>{"name":"run_command","arguments":{"command":"pytest -q"}}</tool_call>',
    }
    chosen = {
        "read": next(value for value in CASES if value["id"] == "read_en_readme"),
        "write": next(value for value in CASES if value["id"] == "write_en_note"),
        "run": next(value for value in CASES if value["id"] == "run_en_pytest"),
    }
    for kind, raw in rows.items():
        parsed = parse_tool_call(raw)
        assert semantic_valid(arm, chosen[kind], parsed)
        assert evaluate(arm, chosen[kind], parsed)["passed"]
    assert cases


def test_generic_command_must_preserve_task_payload() -> None:
    arm = next(value for value in ARMS if value.name == "run_command_only")
    case = next(value for value in CASES if value["id"] == "write_en_note")
    good = parse_tool_call(
        '<tool_call>{"name":"run_command","arguments":{"command":"printf hello > note.txt"}}</tool_call>'
    )
    bad = parse_tool_call(
        '<tool_call>{"name":"run_command","arguments":{"command":"touch note.txt"}}</tool_call>'
    )
    assert evaluate(arm, case, good)["passed"]
    assert not evaluate(arm, case, bad)["semantic_valid"]


def test_workspace_requires_kind_specific_operation() -> None:
    arm = next(value for value in ARMS if value.name == "workspace_only")
    case = next(value for value in CASES if value["id"] == "read_en_readme")
    good = parse_tool_call(
        '<tool_call>{"name":"workspace","arguments":{"op":"read","input":"README.md"}}</tool_call>'
    )
    bad = parse_tool_call(
        '<tool_call>{"name":"workspace","arguments":{"op":"run","input":"cat README.md"}}</tool_call>'
    )
    assert evaluate(arm, case, good)["passed"]
    assert not evaluate(arm, case, bad)["semantic_valid"]


def test_prompt_has_no_demonstration() -> None:
    arm = next(value for value in ARMS if value.name == "native_trio")
    prompt = render_prompt(arm, CASES[0])
    assert "read_file(path)" in prompt
    assert "User: Read README.md." in prompt
    assert prompt.count("User:") == 1
    assert "<tool_call>{\"name\":..." in prompt


def test_summary_counts_pass_and_repeat_stability() -> None:
    rows = []
    for repeat in range(2):
        rows.append(
            {
                "arm": "run_command_only",
                "prefix_mode": "none",
                "case_id": "case",
                "raw": "same",
                "output_tokens": 2,
                "model_elapsed_ms": 10.0,
                "evaluation": {
                    "strict": True,
                    "tool_correct": True,
                    "schema_valid": True,
                    "semantic_valid": True,
                    "passed": True,
                },
            }
        )
    summary = summarize(rows)
    group = summary["groups"]["run_command_only/none"]
    assert group["passed"] == 2
    assert group["repeat_raw_exact_rate"] == 1.0
    assert summary["arms"]["run_command_only"]["pass_rate"] == 1.0
