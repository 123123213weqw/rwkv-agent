import json

from demos.generate_website_swarm import build_specs, render_html, validate_html
from demos.run_agent_website_swarm import (
    RUN_COMMAND_JSON_PREFIX,
    _trace_summary,
    build_identities,
    build_task_prompt,
    content_to_dsl,
    parse_run_command,
    state_root_prompt,
    validate_content,
)


def _content(brand: str) -> dict:
    return {
        "title": f"{brand} Night Atlas",
        "headline": "Map the city stars",
        "summary": "Meet curious neighbors for clear guided observations under the changing urban night sky.",
        "cta": "Join tonight",
        "features": ["Guided viewing", "Shared telescopes", "Local sky notes"],
    }


def test_gate4_identities_are_unique_and_use_the_controller_contract() -> None:
    identities = build_identities(build_specs(100, 7), "gate4-test")
    assert len({item.prompt_sha256 for item in identities}) == 100
    assert len({item.session_id for item in identities}) == 100
    assert len({item.workspace for item in identities}) == 100
    assert len({item.owner_id for item in identities}) == 100
    assert len({state_root_prompt(item) for item in identities}) == 100
    assert all(item.workspace.startswith("gate4/gate4-test/site-") for item in identities)
    assert all("one run_command containing both actions" in item.prompt for item in identities)
    assert all("exact line VALID" in item.prompt for item in identities)


def test_compact_rwkv_content_validates_and_renders_safe_html() -> None:
    specs = build_specs(2, 9)
    content, errors = validate_content(_content(specs[0].brand), specs[0], [item.brand for item in specs])
    assert errors == []
    dsl = content_to_dsl(specs[0], content)
    assert dsl["hero"]["summary"].startswith(specs[0].brand + " — ")
    html = render_html(specs[0], dsl)
    assert specs[0].brand in html
    assert content["headline"] in html
    assert validate_html(html) == []


def test_content_rejects_foreign_brand_and_protocol_leak() -> None:
    specs = build_specs(2, 11)
    value = _content(specs[1].brand)
    value["summary"] = "<think>private</think> words that must never reach a generated artifact page."
    content, errors = validate_content(value, specs[0], [item.brand for item in specs])
    assert content is None
    assert "title:missing_brand" in errors
    assert "title:foreign_brand" in errors
    assert "summary:protocol_or_markup" in errors


def test_trace_requires_separate_validator_and_complete_release() -> None:
    response = {
        "answer": "Completed after VALID",
        "trace": {
            "agent": {
                "model_turns": 2,
                "tool_steps": [
                    {"result": {"status": "ok", "stdout": ""}},
                    {"result": {"status": "ok", "stdout": "VALID\n"}},
                ],
                "events": [
                    {"type": "run_started", "owner_id": "owner-1"},
                    {"type": "state_opened", "state_id": "root"},
                    {"type": "state_opened", "state_id": "worker"},
                    {"type": "state_released", "state_id": "worker", "success": True},
                    {"type": "state_released", "state_id": "root", "success": True},
                ],
            }
        },
    }
    trace = _trace_summary(response)
    assert trace["runtime_validator_passed"]
    assert trace["all_states_released"]
    assert trace["tool_steps"] == 2
    assert json.dumps(trace)


def test_state_agent_tool_payload_is_strict() -> None:
    call, errors = parse_run_command(
        '{"name":"run_command","arguments":{"command":"printf ok"}}',
        "</tool_call>",
    )
    assert errors == []
    assert call["arguments"]["command"] == "printf ok"
    call, errors = parse_run_command(
        '{"name":"run_command","arguments":{"command":"printf ok"},"extra":1}',
        "max_tokens",
    )
    assert call is None
    assert "tool_stop:max_tokens" in errors
    assert "tool_json:envelope" in errors

    call, errors = parse_run_command(
        RUN_COMMAND_JSON_PREFIX + "printf ok\"}}",
        "</tool_call>",
    )
    assert errors == []
    assert call["arguments"]["command"] == "printf ok"


def test_repair_prompt_keeps_identity_without_prefilling_copy() -> None:
    spec = build_specs(1, 3)[0]
    prompt = build_task_prompt(spec, repair_errors=["summary:words"])
    assert spec.brand in prompt
    assert spec.site_id in prompt
    assert "summary:words" in prompt
    assert "headline\":\"" not in prompt


def test_repeated_model_feature_copy_is_structurally_disambiguated() -> None:
    spec = build_specs(1, 17)[0]
    value = _content(spec.brand)
    value["features"] = ["Shared studio", "Shared studio", "Shared studio"]
    content, errors = validate_content(value, spec, [spec.brand])
    assert errors == []
    dsl = content_to_dsl(spec, content)
    names = [row["title"] for row in dsl["features"]]
    assert names == ["Shared studio 1", "Shared studio 2", "Shared studio 3"]


def test_recovery_state_is_fresh_and_receives_only_bounded_failure_context() -> None:
    identity = build_identities(build_specs(1, 21), "gate4-test")[0]
    prompt = state_root_prompt(
        identity,
        recovery_attempt=2,
        recovery_errors=["tool_json:Expecting value@0", "content:missing"],
    )
    assert "fresh recovery State 2" in prompt
    assert "tool_json:Expecting value@0" in prompt
    assert "begin with a JSON object" in prompt
