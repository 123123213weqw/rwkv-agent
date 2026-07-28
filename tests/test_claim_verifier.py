from __future__ import annotations

from rwkv_agent.claim_verifier import claim_units, verify_answer_claims


EVIDENCE = [
    {
        "id": "W1",
        "title": "Project report",
        "content": "The system launched on 15 September 2025 with model B.",
        "uri": "https://example.test/report",
    },
    {
        "id": "W2",
        "title": "项目报告",
        "content": "项目最终预算为80万元，正确率为92%。",
        "uri": "https://example.test/zh",
    },
]


def test_verifier_requires_a_real_citation() -> None:
    claims = verify_answer_claims("The system used model B.", EVIDENCE)
    assert claims[0]["supported"] is False
    assert claims[0]["support_reason"] == "missing_valid_citation"


def test_verifier_accepts_grounded_english_and_chinese_claims() -> None:
    claims = verify_answer_claims(
        "The system launched on 15 September 2025 with model B. [W1]\n"
        "项目最终预算为80万元。[W2]",
        EVIDENCE,
    )
    assert len(claims) == 2
    assert all(claim["supported"] for claim in claims)


def test_verifier_rejects_numeric_contradiction() -> None:
    claims = verify_answer_claims("The system launched in 2024. [W1]", EVIDENCE)
    assert claims[0]["supported"] is False
    assert claims[0]["support_reason"] == "number_mismatch"


def test_verifier_keeps_adjacent_citation_with_preceding_sentence() -> None:
    claims = verify_answer_claims(
        "The system used model B. [W1] The budget was 80万元. [W2]",
        EVIDENCE,
    )
    assert [claim["citations"] for claim in claims] == [["W1"], ["W2"]]


def test_verifier_does_not_split_initials_or_abbreviations() -> None:
    evidence = [
        {
            "id": "W1",
            "content": "David G. Hartwell edited Year's Best SF in the U.S.",
        }
    ]
    claims = verify_answer_claims(
        "David G. Hartwell edited Year's Best SF in the U.S. [W1]",
        evidence,
    )
    assert len(claims) == 1
    assert claims[0]["supported"] is True


def test_verifier_rejects_route_claim_cited_to_planning_page() -> None:
    evidence = [
        {
            "id": "W1",
            "title": "唐镇规划",
            "content": "唐镇位于浦东新区，规划面积32.32平方公里。",
        }
    ]
    claims = verify_answer_claims(
        "从唐镇站乘坐地铁2号线，全程约90分钟，票价10元。[W1]",
        evidence,
    )

    assert claims[0]["supported"] is False
    assert claims[0]["support_reason"] == "number_mismatch"


def test_claim_units_do_not_split_decimal_model_names() -> None:
    units = claim_units("版本包括RWKV-LM-1.5B和RWKV-LM-0.4B。[W1]")

    assert units == ["版本包括RWKV-LM-1.5B和RWKV-LM-0.4B。[W1]"]


def test_verifier_accepts_exact_cited_evidence_uri() -> None:
    evidence = [
        {
            "id": "W1",
            "title": "RWKV-LM",
            "content": "Official repository.",
            "uri": "https://github.com/BlinkDL/RWKV-LM",
        }
    ]

    claims = verify_answer_claims(
        "RWKV-LM：https://github.com/BlinkDL/RWKV-LM [W1]",
        evidence,
    )

    assert claims[0]["supported"] is True


def test_verifier_rejects_invented_same_domain_uri() -> None:
    evidence = [
        {
            "id": "W1",
            "title": "RWKV-LM",
            "content": "Official repository.",
            "uri": "https://github.com/BlinkDL/RWKV-LM",
        }
    ]

    claims = verify_answer_claims(
        "RWKV-v8：https://github.com/BlinkDL/RWKV-v8 [W1]",
        evidence,
    )

    assert claims[0]["supported"] is False
    assert claims[0]["support_reason"] == "url_mismatch"
