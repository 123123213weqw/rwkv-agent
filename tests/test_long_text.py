from __future__ import annotations

import json
import threading
import time
import unittest

from rwkv_agent.tools.long_text import (
    LongTextQAAdapter,
    TextChunk,
    chunk_text,
    parse_chunk_candidate,
    rank_chunks,
    render_chunk_worker_prompt,
)


class FakeChunkModel:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.calls: list[str] = []

    def __call__(
        self,
        prompt: str,
        *,
        max_tokens: int = 96,
        stops: list[str] | None = None,
    ) -> dict:
        del max_tokens, stops
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(prompt)
        try:
            time.sleep(0.02)
            payload = json.loads(prompt.split("\n\nUser: ", 1)[1].split(
                '\n\nAssistant: {"answer":',
                1,
            )[0])
            chunk = payload["chunk"]
            if "147次" in chunk:
                raw = (
                    '"147","quote":"红岸工程第147次常规发射，'
                    '目标类别：甲三。"}'
                )
            else:
                raw = 'null,"quote":null}'
            return {
                "raw": raw,
                "output_tokens": 8,
                "model_elapsed_ms": 1.0,
            }
        finally:
            with self.lock:
                self.active -= 1


class LongTextTests(unittest.TestCase):
    def test_chunking_is_bounded_and_overlapped(self) -> None:
        text = "\n".join(f"第{index}行：" + "字" * 100 for index in range(20))
        chunks = chunk_text(text, max_chars=400, overlap_chars=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 400 for chunk in chunks))
        self.assertTrue(
            all(
                current.char_start < previous.char_end
                for previous, current in zip(chunks, chunks[1:])
            )
        )

    def test_rank_chunks_prefers_question_terms_without_gold_rules(self) -> None:
        chunks = [
            TextChunk(0, "无关的开篇内容。", 0, 8),
            TextChunk(1, "红岸工程第147次常规发射，目标类别：甲三。", 8, 34),
            TextChunk(2, "另一段无关内容。", 34, 43),
        ]
        ranked = rank_chunks("红岸工程是第几次发射？", chunks, top_k=2)
        self.assertEqual(ranked[0][1].chunk_id, 1)

    def test_json_prefix_parser_requires_grounded_quote(self) -> None:
        chunk = TextChunk(
            7,
            "红岸工程第147次常规发射，目标类别：甲三。",
            0,
            25,
        )
        candidate = parse_chunk_candidate(
            '"147","quote":"红岸工程第147次常规发射"}',
            chunk=chunk,
            retrieval_score=3.0,
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.answer, "147")
        unsupported = parse_chunk_candidate(
            '"999","quote":"不存在的原文"}',
            chunk=chunk,
            retrieval_score=3.0,
        )
        self.assertIsNone(unsupported)

    def test_parser_recovers_truncated_object_and_natural_answer(self) -> None:
        chunk = TextChunk(
            18,
            "中国人民解放军第二炮兵，红岸工程第147次常规发射，授权确认完毕。",
            0,
            35,
        )
        candidate = parse_chunk_candidate(
            (
                ' "这是红岸基地的第147次常规发射。",'
                ' "quote":"红岸工程第147次常规发射，授权确认完毕。",'
                ' "unfinished":"'
            ),
            chunk=chunk,
            retrieval_score=4.0,
            question="红岸工程这次常规发射是第几次？",
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.answer, "147次")

    def test_parser_grounds_answer_entity_when_model_omits_quote(self) -> None:
        chunk = TextChunk(
            76,
            "OZMA计划单通道接收，频率1420兆赫，搜索时间约200小时。",
            0,
            38,
        )
        candidate = parse_chunk_candidate(
            '"根据提供的文本，频率是1420兆赫。"}',
            chunk=chunk,
            retrieval_score=5.0,
            question="OZMA计划的单通道接收频率是多少兆赫？",
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.answer, "1420兆赫")
        self.assertIn("1420兆赫", candidate.quote)

    def test_worker_prompt_ends_with_answer_json_prefix(self) -> None:
        prompt = render_chunk_worker_prompt(
            "第几次？",
            TextChunk(0, "第147次。", 0, 7),
        )
        self.assertTrue(prompt.endswith('Assistant: {"answer":'))
        self.assertIn('"long_text_chunk_worker"', prompt)

    def test_adapter_fans_out_in_parallel_and_returns_evidence(self) -> None:
        document = "\n\n".join(
            [
                "无关内容。" + "甲" * 300,
                "红岸工程第147次常规发射，目标类别：甲三。" + "乙" * 300,
                "另一段内容。" + "丙" * 300,
                "结束部分。" + "丁" * 300,
            ]
        )
        model = FakeChunkModel()
        adapter = LongTextQAAdapter(
            model,
            top_k=4,
            concurrency=4,
            chunk_chars=320,
            overlap_chars=20,
        )
        result = adapter.execute(
            document,
            "红岸工程是第几次常规发射？",
        )
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(model.max_active, 2)
        self.assertEqual(result["workers"]["completed"], 4)
        self.assertEqual(result["evidence"][0]["answer_candidate"], "147")
        self.assertEqual(result["answer_hint"], "147")
        self.assertEqual(result["answer_hint_evidence_id"], "L1")
        self.assertEqual(
            result["document"]["source"],
            "session_pasted_text",
        )

    def test_adapter_extracts_grounded_structured_code_without_model(self) -> None:
        def unexpected_model_call(*_args, **_kwargs):
            raise AssertionError("structured code fast path called the model")

        document = "\n\n".join(
            [
                "背景说明：项目已经完成三轮普通审查。",
                "审批委员会最终确认：最终批准代号是 ORBIT-73；该决定由负责人签署。",
                "后续工作将依据审批结果安排。",
            ]
        )
        result = LongTextQAAdapter(
            unexpected_model_call,
            top_k=3,
        ).execute(document, "最终批准代号是什么？")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["answer_hint"], "ORBIT-73")
        self.assertEqual(result["workers"]["submitted"], 0)
        self.assertEqual(
            result["evidence"][0]["content"],
            "审批委员会最终确认：最终批准代号是 ORBIT-73；",
        )
        self.assertIn(
            result["evidence"][0]["content"],
            document,
        )

    def test_adapter_rejects_oversized_pasted_text(self) -> None:
        result = LongTextQAAdapter(
            FakeChunkModel(),
            max_document_chars=20,
        ).execute("x" * 21, "question")
        self.assertEqual(result["status"], "invalid")
        self.assertIn("character limit", result["message"])


if __name__ == "__main__":
    unittest.main()
