from __future__ import annotations

from rwkv_agent.evidence_admission import EntityEvidenceAdmission


def test_entity_admission_rejects_relation_word_collision() -> None:
    evidence = [
        {
            "id": "W1",
            "title": "Leo501/awesome-CocosCreator",
            "content": "A collection of Cocos Creator game projects",
            "uri": "https://github.com/Leo501/awesome-CocosCreator",
        },
        {
            "id": "W2",
            "title": "BlinkDL/RWKV-LM",
            "content": "The official RWKV language model repository",
            "uri": "https://github.com/BlinkDL/RWKV-LM",
        },
        {
            "id": "W3",
            "title": "PENG Bo (@BlinkDL)",
            "content": "RWKV is all you need",
            "uri": "https://github.com/BlinkDL",
        },
    ]

    admitted, trace = EntityEvidenceAdmission().admit(
        "RWKV creator GitHub projects latest update",
        evidence,
    )

    assert [item["id"] for item in admitted] == ["W2", "W3"]
    assert trace.anchors == ("RWKV",)
    assert trace.rejection_counts == {"entity_mismatch": 1}


def test_entity_admission_blocks_origin_only_page_for_relation_question() -> None:
    evidence = [
        {
            "id": "W1",
            "title": "唐镇国土空间总体规划",
            "content": "唐镇位于上海市浦东新区。",
            "uri": "https://example.test/tangzhen",
        },
        {
            "id": "W2",
            "title": "RWKV company contact",
            "content": "RWKV contact and office information.",
            "uri": "https://example.test/rwkv-contact",
        },
    ]

    admitted, trace = EntityEvidenceAdmission().admit(
        "上海浦东新区唐镇站如何到RWKV公司？",
        evidence,
    )

    assert [item["id"] for item in admitted] == ["W2"]
    assert trace.rejected == 1


def test_entity_admission_preserves_recall_without_stable_anchor() -> None:
    evidence = [{"id": "W1", "title": "普通结果", "content": "一般内容"}]

    admitted, trace = EntityEvidenceAdmission().admit("怎么样？", evidence)

    assert admitted == evidence
    assert trace.anchors == ()
