import json

from demos.generate_website_swarm import (
    build_specs,
    extract_json,
    render_gallery,
    render_html,
    Generation,
    validate_dsl,
    validate_html,
    website_prompt,
)


def _dsl(brand: str) -> dict:
    return {
        "title": f"{brand} — Local Ideas",
        "tagline": "A concise original landing page.",
        "theme": {
            "background": "#101522",
            "surface": "#182035",
            "primary": "#73E2A7",
            "text": "#F4F7FF",
            "accent": "#F8C630",
        },
        "hero": {
            "eyebrow": "Built nearby",
            "headline": "Make small ideas visible",
            "summary": "A focused place for people to meet, build, and share useful work.",
            "cta": "Explore now",
        },
        "features": [
            {"title": "Meet", "description": "Find people who care about the same practical ideas."},
            {"title": "Build", "description": "Turn a small concept into a visible working project."},
            {"title": "Share", "description": "Invite the neighborhood to learn and contribute."},
        ],
        "stats": [
            {"value": "24", "label": "weekly sessions"},
            {"value": "680+", "label": "local members"},
            {"value": "12", "label": "open projects"},
        ],
        "footer": "Made locally with care.",
    }


def test_build_specs_is_deterministic_and_unique_for_100_workers() -> None:
    first = build_specs(100, 7)
    second = build_specs(100, 7)
    assert first == second
    assert len({item.site_id for item in first}) == 100
    assert len({item.brand for item in first}) == 100


def test_prompt_uses_opening_supplied_json_contract() -> None:
    prompt = website_prompt(build_specs(1, 3)[0])
    assert prompt.endswith("Assistant: {")
    assert "exactly these keys" in prompt


def test_extract_validate_render_pipeline_escapes_model_text() -> None:
    spec = build_specs(1, 9)[0]
    value = _dsl(spec.brand)
    parsed, errors = extract_json(json.dumps(value))
    assert errors == []
    normalized, errors = validate_dsl(parsed, spec)
    assert errors == []
    normalized["hero"]["summary"] = "safe & sound"
    html = render_html(spec, normalized)
    assert "safe &amp; sound" in html
    assert validate_html(html) == []


def test_validator_rejects_protocol_leak_and_wrong_shape() -> None:
    parsed, errors = extract_json('{"title":"x"}<think>hidden</think>')
    assert parsed is None
    assert errors == ["protocol_leak"]
    spec = build_specs(1, 4)[0]
    value = _dsl(spec.brand)
    value["features"] = value["features"][:2]
    normalized, errors = validate_dsl(value, spec)
    assert normalized is None
    assert "features:count" in errors


def test_gallery_links_only_valid_generations() -> None:
    spec = build_specs(1, 5)[0]
    dsl = _dsl(spec.brand)
    generation = Generation(spec, "prompt", json.dumps(dsl), dsl, [], 1, 20)
    gallery = render_gallery([generation])
    assert f'{spec.site_id}/index.html' in gallery
    assert "100 states. 100 websites." in gallery
