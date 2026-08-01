from __future__ import annotations

import unittest

from rwkv_search.evidence import Evidence
from rwkv_search.rwkv_answerer import (
    HFLocalRWKVAnswerer,
    build_rwkv_chat_prompt,
    build_rwkv_grounded_prompt,
    clean_chat_output,
    extract_last_json,
    grounded_text_envelope,
    natural_answer_envelope,
    valid_answer_schema,
)
from rwkv_search.router import RouteDecision


class RWKVAnswerExtractionTests(unittest.TestCase):
    def test_chat_cleanup_does_not_rewrite_semantic_content(self) -> None:
        self.assertEqual(
            clean_chat_output("我是 OpenAI 的模型。", "你是谁？"),
            "我是 OpenAI 的模型。",
        )
        self.assertEqual(clean_chat_output("我是你爹", "我是你爹"), "我是你爹")

    def test_chat_prompt_is_compact_and_does_not_claim_search_mode(self) -> None:
        prompt = build_rwkv_chat_prompt("什么是奶龙", [])
        self.assertLess(len(prompt), 180)
        self.assertNotIn("search system", prompt.casefold())
        self.assertNotIn("搜索引擎", prompt)
        self.assertIn("User: 什么是奶龙", prompt)

    def test_grounded_prompt_mentions_evidence_without_search_identity(self) -> None:
        evidence = [
            Evidence(
                id="S1",
                title="奶龙官方介绍",
                url="https://example.com/nailong",
                source_type="official",
                published_at=None,
                fetched_at=1.0,
                authority=1.0,
                text="奶龙是一部动画作品。",
                score=1.0,
            )
        ]
        route = RouteDecision("static_knowledge", ["web_search"], "stable", "single", False, ["奶龙"], [], "test")
        prompt = build_rwkv_grounded_prompt(
            "什么是奶龙", route, evidence, as_of="now", timezone="Asia/Shanghai"
        )
        self.assertNotIn("最终回答模块", prompt)
        self.assertNotIn("搜索引擎", prompt)
        self.assertNotIn("金融", prompt)
        self.assertIn("[S1]", prompt)

    def test_grounded_prompt_labels_search_snippet_fallback(self) -> None:
        evidence = [
            Evidence(
                id="S1",
                title="Official release",
                url="https://example.com/release",
                source_type="official_docs",
                published_at=None,
                fetched_at=1.0,
                authority=0.82,
                text="Search result excerpt with limited release information.",
                score=0.5,
                metadata={"score_components": {"snippet_fallback": 1.0}},
            )
        ]
        route = RouteDecision(
            "latest_knowledge",
            ["web_search"],
            "latest",
            "single",
            False,
            ["release"],
            [],
            "test",
        )

        prompt = build_rwkv_grounded_prompt(
            "latest release", route, evidence, as_of="now", timezone="UTC"
        )

        self.assertIn("原网页抓取失败", prompt)
        self.assertIn("只能作为有限证据", prompt)

    def test_grounded_answer_gets_one_citation_repair_before_fallback(self) -> None:
        answerer = object.__new__(HFLocalRWKVAnswerer)
        answerer.session_cache_enabled = False
        answerer.session_cpu_offload = True
        answerer.repair_once = True
        answerer.max_new_tokens = 256
        responses = [
            {
                "raw": "奶龙是一部动画作品。",
                "latency_ms": 2.0,
                "new_tokens": 8,
            },
            {
                "raw": "奶龙是一部动画作品。[S1]",
                "latency_ms": 3.0,
                "new_tokens": 10,
            },
        ]

        def fake_incremental(*args, **kwargs):
            return responses.pop(0)

        answerer._generate_incremental_complete = fake_incremental
        evidence = [
            Evidence(
                id="S1",
                title="奶龙介绍",
                url="https://example.com/nailong",
                source_type="web",
                published_at=None,
                fetched_at=1.0,
                authority=1.0,
                text="奶龙是一部动画作品。",
                score=1.0,
            )
        ]
        route = RouteDecision("static_knowledge", ["web_search"], "stable", "single", False, ["奶龙"], [], "test")
        result = answerer._answer_locked(
            "什么是奶龙",
            route,
            evidence,
            as_of="now",
            timezone="Asia/Shanghai",
            history=[],
        )
        self.assertTrue(result.repaired)
        self.assertEqual(result.answer["citations"], ["S1"])
        self.assertIn("[S1]", result.answer["answer"])

    def test_extracts_last_complete_json_object_from_noisy_output(self) -> None:
        raw = (
            '<think>hidden</think>\n'
            '{"draft":true}\n'
            '```json\n'
            '{"answer":"最终回答","citations":["S1"],"data_time":"2026-07-16T00:00:00Z",'
            '"insufficient_evidence":false,"needs_clarification":false}\n```'
        )
        value = extract_last_json(raw)
        self.assertIsNotNone(value)
        self.assertEqual(value["answer"], "最终回答")
        self.assertTrue(valid_answer_schema(value))

    def test_rejects_partial_or_wrong_typed_answer_envelope(self) -> None:
        self.assertIsNone(extract_last_json('{"answer":'))
        self.assertFalse(
            valid_answer_schema(
                {
                    "answer": "x",
                    "citations": "S1",
                    "data_time": "now",
                    "insufficient_evidence": False,
                    "needs_clarification": False,
                }
            )
        )

    def test_wraps_clean_repair_text_in_valid_answer_json(self) -> None:
        evidence = [
            Evidence(
                id="S1",
                title="Example",
                url="https://example.com",
                source_type="web",
                published_at=None,
                fetched_at=1.0,
                authority=1.0,
                text="Example evidence",
                score=1.0,
            )
        ]
        value = natural_answer_envelope(
            "It is used for documentation examples.\nNow we need to output JSON.",
            evidence,
            as_of="2026-07-16T00:00:00Z",
        )
        self.assertTrue(valid_answer_schema(value))
        self.assertEqual(value["citations"], ["S1"])
        self.assertEqual(value["answer"], "It is used for documentation examples.")

    def test_server_wraps_grounded_natural_text_and_attaches_ledger_citation(self) -> None:
        evidence = [
            Evidence(
                id="S1",
                title="Official release",
                url="https://example.com/release",
                source_type="official_docs",
                published_at="2026-07-15",
                fetched_at=1.0,
                authority=1.0,
                text="Version 1 was released.",
                score=1.0,
            )
        ]
        value = grounded_text_envelope(
            "该版本已经发布。[S1]", evidence, as_of="2026-07-16T00:00:00Z"
        )
        self.assertTrue(valid_answer_schema(value))
        self.assertEqual(value["citations"], ["S1"])
        attached = grounded_text_envelope(
            "该版本已经发布。", evidence, as_of="2026-07-16T00:00:00Z"
        )
        self.assertIsNone(attached)
        partial = grounded_text_envelope(
            '{"answer":"该版本包含多个新特性。", "citations":',
            evidence,
            as_of="2026-07-16T00:00:00Z",
        )
        self.assertIsNone(partial)
        self.assertIsNone(
            grounded_text_envelope(
                "S1 搜索结果\nURL: https://example.com/release",
                evidence,
                as_of="2026-07-16T00:00:00Z",
            )
        )

    def test_soft_limit_continues_until_sentence_boundary(self) -> None:
        answerer = object.__new__(HFLocalRWKVAnswerer)
        responses = [
            {"raw": "回答还没有说", "_decoded": "回答还没有说", "new_tokens": 5, "latency_ms": 10.0},
            {"raw": "完。", "_decoded": "完。", "new_tokens": 2, "latency_ms": 3.0},
        ]
        calls = []

        def fake_generate(prompt, limit, stop_strings=(), cancel_event=None):
            calls.append((prompt, limit, stop_strings))
            return responses.pop(0)

        answerer._generate = fake_generate
        result = answerer._generate_complete(
            "prompt:", soft_limit=5, hard_limit=12, stop_strings=("\nUser:",)
        )
        self.assertEqual(result["raw"], "回答还没有说完。")
        self.assertEqual(result["new_tokens"], 7)
        self.assertEqual(len(calls), 2)
        self.assertTrue(HFLocalRWKVAnswerer._ends_at_boundary(result["raw"]))

    def test_early_eos_with_long_incomplete_prose_is_also_continued(self) -> None:
        answerer = object.__new__(HFLocalRWKVAnswerer)
        prefix = "这是一段仍未结束的长回答" * 12
        responses = [
            {"raw": prefix, "_decoded": prefix, "new_tokens": 40, "latency_ms": 5.0},
            {"raw": "，现在补充完整。", "_decoded": "，现在补充完整。", "new_tokens": 8, "latency_ms": 2.0},
        ]

        def fake_generate(prompt, limit, stop_strings=(), cancel_event=None):
            return responses.pop(0)

        answerer._generate = fake_generate
        result = answerer._generate_complete(
            "prompt:", soft_limit=64, hard_limit=256
        )
        self.assertEqual(result["new_tokens"], 48)
        self.assertTrue(result["raw"].endswith("补充完整。"))

    def test_continuation_overlap_is_not_repeated(self) -> None:
        self.assertEqual(
            HFLocalRWKVAnswerer._merge_continuation(
                "五、主要应", "主要应用领域包括 Web 开发。"
            ),
            "五、主要应用领域包括 Web 开发。",
        )

    def test_role_stop_is_removed_before_continuation(self) -> None:
        answerer = object.__new__(HFLocalRWKVAnswerer)
        prefix = "这是前面已经生成的长内容。" * 10 + "五、主要应"
        responses = [
            {
                "raw": prefix + "\nAssistant:",
                "_decoded": prefix + "\nAssistant:",
                "new_tokens": 80,
                "latency_ms": 5.0,
            },
            {
                "raw": "主要应用领域包括 Web 开发。",
                "_decoded": "主要应用领域包括 Web 开发。",
                "new_tokens": 12,
                "latency_ms": 2.0,
            },
        ]

        def fake_generate(prompt, limit, stop_strings=(), cancel_event=None):
            return responses.pop(0)

        answerer._generate = fake_generate
        result = answerer._generate_complete(
            "prompt:",
            soft_limit=80,
            hard_limit=256,
            stop_strings=("\nAssistant:",),
        )
        self.assertTrue(result["raw"].endswith("五、主要应用领域包括 Web 开发。"))
        self.assertNotIn("Assistant:", result["raw"])

    def test_failed_continuation_trims_only_dangling_tail(self) -> None:
        complete = "第一部分已经完整。" * 15
        text = complete + "\n三、尚未完"
        self.assertEqual(
            HFLocalRWKVAnswerer._trim_to_boundary(text),
            complete,
        )

    def test_stream_hides_reasoning_until_final_answer(self) -> None:
        partial = "<think>这里是尚未结束的推理"
        self.assertEqual(
            HFLocalRWKVAnswerer._hide_reasoning_for_stream(partial), ""
        )
        complete = "<think>这里是推理。</think>最终答案。[S1]"
        self.assertEqual(
            HFLocalRWKVAnswerer._hide_reasoning_for_stream(complete),
            "最终答案。[S1]",
        )

    def test_insufficient_grounded_answer_does_not_require_fake_citation(self) -> None:
        evidence = [
            Evidence(
                id="S1",
                title="Unrelated",
                url="https://example.com/unrelated",
                source_type="web",
                published_at=None,
                fetched_at=1.0,
                authority=0.5,
                text="Unrelated material",
                score=0.5,
            )
        ]
        value = grounded_text_envelope(
            "现有证据不足，无法回答该问题。",
            evidence,
            as_of="2026-07-17T00:00:00Z",
        )
        self.assertTrue(valid_answer_schema(value))
        self.assertEqual(value["citations"], [])
        self.assertTrue(value["insufficient_evidence"])

    def test_not_found_grounded_answer_is_treated_as_insufficient(self) -> None:
        evidence = [
            Evidence(
                id="S1",
                title="Unrelated",
                url="https://example.com/unrelated",
                source_type="web",
                published_at=None,
                fetched_at=1.0,
                authority=0.5,
                text="Unrelated material",
                score=0.5,
            )
        ]
        value = grounded_text_envelope(
            "根据提供资料，未找到关于“奶龙”的信息。",
            evidence,
            as_of="2026-07-17T00:00:00Z",
        )
        self.assertTrue(valid_answer_schema(value))
        self.assertEqual(value["citations"], [])
        self.assertTrue(value["insufficient_evidence"])


if __name__ == "__main__":
    unittest.main()
