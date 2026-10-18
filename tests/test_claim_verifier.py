from __future__ import annotations

from rwkv_agent.claim_verifier import (
    claim_question_relevance,
    claim_units,
    verify_answer_claims,
)


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


def test_structured_exact_does_not_launder_an_invented_relation() -> None:
    evidence = [
        {
            "id": "W1",
            "content": (
                "RWKV is all you need. GitHub user: BlinkDL. "
                "Public repositories: 35."
            ),
        }
    ]

    grounded = verify_answer_claims("BlinkDL拥有35个公开仓库。[W1]", evidence)
    invented = verify_answer_claims(
        "BlinkDL拥有35个公开仓库，因此正式收购了RWKV。[W1]",
        evidence,
    )

    assert grounded[0]["supported"] is True
    assert grounded[0]["support_reason"] == "bounded_structured_exact"
    assert invented[0]["supported"] is False


def test_multiple_citations_cannot_combine_partial_sources_into_support() -> None:
    evidence = [
        {
            "id": "W1",
            "content": "Bo Peng leads and maintains RWKV. Stability AI supplied GPUs.",
        },
        {
            "id": "W2",
            "content": "Many Chinese contributors participated in RWKV development.",
        },
    ]
    claims = verify_answer_claims(
        "Bo Peng维护RWKV，Stability AI提供GPU，并有大量中国人参与开发。[W1][W2]",
        evidence,
    )

    assert claims[0]["supported"] is False
    assert claims[0]["support_evidence_id"] == ""


def test_question_relevance_ignores_answer_values_but_rejects_tangents() -> None:
    question = "核实 RWKV 的主要作者、维护组织和相关公司。"

    assert claim_question_relevance(question, "主要作者：彭博（Bo Peng）。") >= 0.25
    assert claim_question_relevance(
        question,
        "他拥有20年以上编程经验，最初因个人兴趣使用AI生成小说。",
    ) < 0.25
