from __future__ import annotations

from rwkv_agent.page_quality import classify_page_quality


def test_dictionary_is_rejected_for_factual_question_and_never_pivots() -> None:
    decision = classify_page_quality(
        "What is the average left-field distance in retractable-roof MLB parks?",
        {
            "title": "average definition and pronunciation",
            "content": "The meaning and pronunciation of average.",
            "uri": "https://dictionary.cambridge.org/dictionary/english/average",
        },
    )

    assert decision.page_type == "dictionary"
    assert not decision.evidence_allowed
    assert not decision.pivot_allowed


def test_dictionary_brand_compound_is_still_classified_by_page_shape() -> None:
    decision = classify_page_quality(
        "How many times did Argentina enter the Olympic Games?",
        {
            "title": "usdictionary.com",
            "content": "Many is a determiner used with countable nouns.",
            "uri": "https://usdictionary.com/definitions/many",
        },
    )

    assert decision.page_type == "dictionary"
    assert not decision.evidence_allowed
    assert not decision.pivot_allowed


def test_lexical_body_is_rejected_even_when_serp_title_is_only_a_hostname() -> None:
    decision = classify_page_quality(
        "Who is Stephen's brother?",
        {
            "title": "yingwenming.com",
            "content": "Stephen英文名的寓意、读音、来源和发音音标。",
            "uri": "https://yingwenming.com/meaning/Stephen",
        },
    )

    assert decision.page_type == "dictionary"
    assert not decision.evidence_allowed
    assert not decision.pivot_allowed


def test_dictionary_can_answer_lexical_request_but_cannot_seed_feedback() -> None:
    decision = classify_page_quality(
        "average是什么意思？",
        {
            "title": "average是什么意思",
            "content": "average：平均数；平均的。",
            "uri": "https://dict.example/average",
        },
    )

    assert decision.page_type == "dictionary"
    assert decision.evidence_allowed
    assert not decision.pivot_allowed


def test_short_generic_homepage_is_not_answer_evidence() -> None:
    decision = classify_page_quality(
        "Who won the award?",
        {
            "title": "Home",
            "content": "Home About Contact",
            "uri": "https://example.org/",
        },
    )

    assert decision.page_type == "navigation"
    assert not decision.evidence_allowed
    assert not decision.pivot_allowed


def test_normal_content_page_is_allowed_for_evidence_and_feedback() -> None:
    decision = classify_page_quality(
        "Who won the award?",
        {
            "title": "2024 Award Winners",
            "content": "Alice won the 2024 award after the final review.",
            "uri": "https://example.org/news/award-winners",
        },
    )

    assert decision.page_type == "content"
    assert decision.evidence_allowed
    assert decision.pivot_allowed
