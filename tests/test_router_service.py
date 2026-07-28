from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from rwkv_search.db import SearchDatabase
from rwkv_search.router import RuleRouter
from rwkv_search.service import SearchService


class FailingAnswerer:
    def answer(self, query, route, evidence, *, as_of, timezone, history=None):
        raise RuntimeError("synthetic generation failure")


class GeneralAnswerer:
    def answer(self, query, route, evidence, *, as_of, timezone, history=None):
        self.evidence_count = len(evidence)
        return SimpleNamespace(
            answer={
                "answer": "你好，我是 RWKV Search。",
                "citations": [],
                "data_time": as_of,
                "insufficient_evidence": False,
                "needs_clarification": False,
            },
            latency_ms=5.0,
            new_tokens=12,
            repaired=False,
            error=None,
        )


class RouterServiceTests(unittest.TestCase):
    def test_rule_router_hard_guards(self) -> None:
        router = RuleRouter()
        time_route = router.route("今天星期几？", "Asia/Shanghai")
        self.assertEqual(time_route.tools, ["clock"])
        self.assertEqual(time_route.freshness, "realtime")
        for query in ("买什么股票好？", "搜索一下奶龙", "RWKV 是什么？", "你好"):
            decision = router.route(query)
            self.assertEqual(decision.intent, "chat", query)
            self.assertEqual(decision.tools, [], query)

    def test_generic_router_uses_execution_signals_not_domains(self) -> None:
        router = RuleRouter(
            lambda query: {
                "use_tool": query.endswith("SEARCH"),
                "query": query.removesuffix("SEARCH").strip(),
                "reason": "test semantic resolver",
            }
        )
        searched = router.route("任意未见主题 SEARCH")
        self.assertEqual(searched.intent, "search")
        self.assertEqual(searched.queries, ["任意未见主题"])
        self.assertEqual(router.route("任意未见主题").intent, "chat")

    def test_service_time_finance_and_grounded_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = SearchDatabase(Path(tmp) / "search.db")
            db.upsert_document(
                url="https://internal.example/sla",
                canonical_url="https://internal.example/sla",
                title="搜索服务 SLA",
                content="搜索服务月度可用性 SLA 为 99.9%，P95 响应时间目标小于 3 秒。",
                published_at=None,
                fetched_at=time.time(),
                etag=None,
                last_modified=None,
                content_type="text/html",
                language="zh-CN",
                source_type="local_document",
                authority=1.0,
            )
            router = RuleRouter(
                lambda query: {
                    "use_tool": True,
                    "query": query,
                    "reason": "test semantic resolver",
                }
            )
            service = SearchService(db, router=router)
            time_events = list(service.ask_events("今天星期几？"))
            self.assertIn("星期", next(e for e in time_events if e["type"] == "answer")["answer"]["answer"])
            finance_events = list(service.ask_events("买什么股票好？"))
            finance_answer = next(e for e in finance_events if e["type"] == "answer")["answer"]
            self.assertFalse(finance_answer["needs_clarification"])
            self.assertTrue(finance_answer["insufficient_evidence"])
            local_events = list(service.ask_events("我们搜索服务的 SLA 是多少？"))
            local_answer = next(e for e in local_events if e["type"] == "answer")["answer"]
            self.assertIn("99.9%", local_answer["answer"])
            self.assertIn("S1", local_answer["citations"])

            degraded_events = list(
                SearchService(db, answerer=FailingAnswerer(), router=router).ask_events(
                    "我们搜索服务的 SLA 是多少？"
                )
            )
            degraded = next(e for e in degraded_events if e["type"] == "answer")
            self.assertFalse(degraded["model"]["used"])
            self.assertIn("RuntimeError", degraded["model"]["error"])
            self.assertIn("99.9%", degraded["answer"]["answer"])

    def test_general_chat_calls_rwkv_without_search_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = SearchDatabase(Path(tmp) / "search.db")
            answerer = GeneralAnswerer()
            events = list(SearchService(db, answerer=answerer).ask_events("你好，你是谁？"))
            answer_event = next(event for event in events if event["type"] == "answer")
            self.assertEqual(answerer.evidence_count, 0)
            self.assertNotIn("sources", [event["type"] for event in events])
            self.assertNotIn("evidence", [event["type"] for event in events])
            self.assertTrue(answer_event["model"]["used"])
            self.assertIn("RWKV Search", answer_event["answer"]["answer"])

            tomorrow = next(
                event
                for event in SearchService(db).ask_events("明天星期几？")
                if event["type"] == "answer"
            )
            self.assertIn("明天是", tomorrow["answer"]["answer"])

            cleaned = SearchService._unstructured_model_answer(
                "<think>hidden reasoning</think>\n我是 OpenAI 的模型。",
                "now",
                "请介绍你自己",
            )
            self.assertNotIn("think", cleaned["answer"])
            self.assertIn("OpenAI", cleaned["answer"])

            echoed_abuse = SearchService._unstructured_model_answer(
                "我是你爹", "now", "我是你爹"
            )
            self.assertEqual(echoed_abuse["answer"], "我是你爹")

            complete = "这是一句完整内容。" * 15
            cleaned = SearchService._unstructured_model_answer(
                complete + "\n五、主要应\nAssistant: ignored。",
                "now",
                "写一篇文章",
            )
            self.assertEqual(cleaned["answer"], complete)

            follow_up_events = list(
                SearchService(db, answerer=answerer).ask_events(
                    "你是谁？",
                    history=[
                        {"role": "user", "content": "今天星期几？"},
                        {"role": "assistant", "content": "今天星期四。"},
                    ],
                )
            )
            follow_up_route = next(
                event["route"] for event in follow_up_events if event["type"] == "route"
            )
            self.assertEqual(follow_up_route["intent"], "chat")
            self.assertEqual(follow_up_route["tools"], [])

            forced_search_events = list(
                SearchService(db, answerer=answerer).ask_events(
                    "你好", search_mode="always", source_scope="local"
                )
            )
            forced_search_route = next(
                event["route"]
                for event in forced_search_events
                if event["type"] == "route"
            )
            self.assertEqual(forced_search_route["intent"], "search")
            self.assertEqual(forced_search_route["tools"], ["local_search"])
            self.assertIn("search switch", forced_search_route["reason"])

            fallback = SearchService._chat_fallback("你好", "now")
            self.assertIn("重试", fallback["answer"])
            self.assertFalse(fallback["insufficient_evidence"])
