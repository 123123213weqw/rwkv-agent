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
        recommendation = router.route("买什么股票好？")
        self.assertEqual(recommendation.intent, "search")
        self.assertFalse(recommendation.needs_clarification)
        self.assertEqual(recommendation.missing_context, [])
        self.assertEqual(recommendation.tools, ["local_search", "web_search"])
        self.assertEqual(recommendation.queries, ["买什么股票好"])
        self.assertEqual(router.route("今天有什么重要新闻？").intent, "search")
        self.assertEqual(router.route("今天买什么股票好？").intent, "search")
        self.assertEqual(router.route("北京今天会下雨吗？").intent, "search")
        self.assertEqual(router.route("RWKV 是什么？").intent, "chat")
        python_prefix = router.route("什么是python")
        self.assertEqual(python_prefix.intent, "chat")
        self.assertEqual(python_prefix.tools, [])
        spelled = router.route("什么是r w k v")
        self.assertEqual(spelled.intent, "chat")
        self.assertEqual(spelled.tools, [])
        searched = router.route("搜索一下 Python 3.14 是什么，请给出来源")
        self.assertEqual(searched.intent, "search")
        self.assertEqual(searched.queries, ["Python 3.14"])
        python_sources = router.route("什么是 Python？请给出来源")
        self.assertEqual(python_sources.queries, ["Python"])
        chinese_search = router.route("搜索一下奶龙是什么")
        self.assertEqual(chinese_search.intent, "search")
        self.assertEqual(chinese_search.queries, ["奶龙"])
        for colloquial_search in (
            "搜索下奶龙",
            "搜下奶龙",
            "查下奶龙",
            "查询下奶龙",
            "帮我搜索下奶龙",
        ):
            decision = router.route(colloquial_search)
            self.assertEqual(decision.intent, "search", colloquial_search)
            self.assertEqual(decision.queries, ["奶龙"], colloquial_search)
        self.assertEqual(router.route("奶龙是什么").tools, [])
        for query in (
            "你好",
            "你是谁？",
            "帮我写一个 Python 二分查找函数",
            "把这句话翻译成英文",
            "1+1等于几？",
            "给我讲个笑话",
        ):
            decision = router.route(query)
            self.assertEqual(decision.intent, "chat", query)
            self.assertEqual(decision.tools, [], query)

    def test_generic_router_uses_execution_signals_not_domains(self) -> None:
        router = RuleRouter()
        for query in (
            "RWKV 最近有什么进展",
            "最近有哪些台风",
            "Python 最新版本",
            "最近有什么新能源汽车政策",
            "奶龙最近为什么火",
            "网友如何评价这个产品",
            "给我找一下官方来源",
        ):
            decision = router.route(query)
            self.assertEqual(decision.intent, "search", query)
            self.assertIn("web_search", decision.tools, query)
            self.assertEqual(decision.queries, [query.rstrip("？?。.!！")], query)

        # Domain words alone do not create a vertical route.
        for query in ("股票是什么", "新闻是什么", "Python 是什么"):
            decision = router.route(query)
            self.assertEqual(decision.intent, "chat", query)
            self.assertEqual(decision.tools, [], query)

        for query in (
            "我今天心情不好",
            "把‘最新’翻译成英文",
            "写一个实时行情抓取脚本",
        ):
            self.assertEqual(router.route(query).intent, "chat", query)

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
            service = SearchService(db)
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
                SearchService(db, answerer=FailingAnswerer()).ask_events(
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
            self.assertNotIn("OpenAI", cleaned["answer"])
            self.assertIn("RWKV Search", cleaned["answer"])

            echoed_abuse = SearchService._unstructured_model_answer(
                "我是你爹", "now", "我是你爹"
            )
            self.assertNotEqual(echoed_abuse["answer"], "我是你爹")

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
            self.assertIn("你好", fallback["answer"])
            self.assertFalse(fallback["insufficient_evidence"])
