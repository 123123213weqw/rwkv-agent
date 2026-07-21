from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from rwkv_search.api import SearchHTTPServer
from rwkv_search.db import SearchDatabase
from rwkv_search.debug_trace import DebugTraceStore


class FakeRWKVAnswerer:
    def status(self):
        return {
            "enabled": True,
            "ready": True,
            "label": "RWKV Test",
            "model": "rwkv-test-hf",
            "device": "cpu",
            "dtype": "fp32",
            "native_model": False,
            "error": None,
        }

    def answer(self, query, route, evidence, *, as_of, timezone, history=None):
        self.last_history = history or []
        return SimpleNamespace(
            answer={
                "answer": "RWKV 已通过前端服务调用，并使用本地证据。[S1]",
                "citations": ["S1"],
                "data_time": as_of,
                "insufficient_evidence": False,
                "needs_clarification": False,
            },
            latency_ms=12.5,
            new_tokens=24,
            repaired=False,
            error=None,
        )


class FakeStreamingRWKVAnswerer(FakeRWKVAnswerer):
    supports_streaming = True
    supports_sessions = True
    supports_cancellation = True
    supports_debug = True

    def answer(
        self,
        query,
        route,
        evidence,
        *,
        as_of,
        timezone,
        history=None,
        cancel_event=None,
        conversation_id=None,
        on_delta=None,
        on_debug=None,
    ):
        self.conversation_id = conversation_id
        if on_debug:
            on_debug(
                {
                    "kind": "context_injection",
                    "phase": "chat_answer",
                    "text": "User: 你好\nAssistant:",
                    "input_token_ids": [1, 2],
                    "input_token_count": 2,
                    "temporary": True,
                }
            )
            on_debug(
                {
                    "kind": "token",
                    "phase": "chat_answer",
                    "index": 0,
                    "token_id": 3,
                    "token_text": "真",
                    "eos": False,
                    "hidden": False,
                }
            )
        for value in ("真正", "逐 token", " 流式输出"):
            if on_delta:
                on_delta(value)
        return SimpleNamespace(
            answer={
                "answer": "真正逐 token 流式输出",
                "citations": [],
                "data_time": as_of,
                "insufficient_evidence": False,
                "needs_clarification": False,
            },
            latency_ms=8.0,
            new_tokens=9,
            repaired=False,
            error=None,
        )


class APITests(unittest.TestCase):
    @staticmethod
    def _sse_events(body: str):
        events = []
        for block in body.split("\n\n"):
            data = "\n".join(
                line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")
            )
            if data:
                events.append(json.loads(data))
        return events

    def test_health_static_and_sse_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = SearchDatabase(Path(tmp) / "search.db")
            server = SearchHTTPServer(("127.0.0.1", 0), database)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(base + "/api/health", timeout=3) as response:
                    health = json.loads(response.read())
                self.assertTrue(health["ok"])
                self.assertEqual(health["protocol"]["version"], "1.0")
                self.assertFalse(health["shadow_search"]["enabled"])
                self.assertFalse(
                    health["shadow_search"]["visible_output_changed"]
                )
                with urllib.request.urlopen(base + "/", timeout=3) as response:
                    homepage = response.read()
                    self.assertIn(b"RWKV Search", homepage)
                    self.assertIn(b"FineWiki", homepage)
                    self.assertIn("仅下一条消息使用搜索".encode(), homepage)
                request = urllib.request.Request(
                    base + "/api/ask",
                    data=json.dumps({"query": "今天星期几？", "timezone": "Asia/Shanghai"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    body = response.read().decode("utf-8")
                self.assertIn("event: route", body)
                self.assertIn("event: answer", body)
                self.assertIn("星期", body)
            finally:
                server.shutdown()
                server.server_close()

    def test_v1_sse_contract_and_cancel_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = SearchDatabase(Path(tmp) / "search.db")
            server = SearchHTTPServer(("127.0.0.1", 0), database)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                request = urllib.request.Request(
                    base + "/api/v1/chat/stream",
                    data=json.dumps(
                        {
                            "schema_version": "1.0",
                            "request_id": "req-contract-test",
                            "conversation_id": "conv-contract-test",
                            "message_id": "msg-contract-test",
                            "query": "今天星期几？",
                            "history": [],
                            "search_mode": "auto",
                            "research_depth": "fast",
                            "source_scope": "auto",
                            "timezone": "Asia/Shanghai",
                            "locale": "zh-CN",
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    self.assertEqual(
                        response.headers["X-RWKV-Request-ID"], "req-contract-test"
                    )
                    events = self._sse_events(response.read().decode("utf-8"))
                kinds = [event["type"] for event in events]
                self.assertEqual(kinds[0], "request_started")
                self.assertIn("route", kinds)
                self.assertIn("generation_started", kinds)
                self.assertIn("answer_delta", kinds)
                self.assertIn("answer_final", kinds)
                self.assertEqual(kinds[-1], "done")
                self.assertEqual(
                    [event["sequence"] for event in events],
                    list(range(1, len(events) + 1)),
                )
                for event in events:
                    self.assertEqual(event["schema_version"], "1.0")
                    self.assertEqual(event["conversation_id"], "conv-contract-test")
                    self.assertEqual(event["message_id"], "msg-contract-test")
                final = next(event for event in events if event["type"] == "answer_final")
                self.assertIn("星期", final["answer"]["content"])
                self.assertIn("usage", final)

                cancel_event = server.request_registry.register("req-cancel-test")
                cancel_request = urllib.request.Request(
                    base + "/api/v1/requests/req-cancel-test/cancel",
                    data=b"",
                    method="POST",
                )
                with urllib.request.urlopen(cancel_request, timeout=3) as response:
                    cancelled = json.loads(response.read())
                self.assertTrue(cancelled["cancelled"])
                self.assertTrue(cancel_event.is_set())
                server.request_registry.finish("req-cancel-test")
            finally:
                server.shutdown()
                server.server_close()

    def test_health_and_sse_expose_rwkv_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = SearchDatabase(Path(tmp) / "search.db")
            database.upsert_document(
                url="https://docs.example/rwkv",
                canonical_url="https://docs.example/rwkv",
                title="RWKV 搜索",
                content="RWKV 搜索使用本地网页证据生成带引用的回答。",
                published_at=None,
                fetched_at=time.time(),
                etag=None,
                last_modified=None,
                content_type="text/html",
                language="zh-CN",
                source_type="official_docs",
                authority=1.0,
            )
            answerer = FakeRWKVAnswerer()
            server = SearchHTTPServer(
                ("127.0.0.1", 0), database, answerer=answerer
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(base + "/api/health", timeout=3) as response:
                    health = json.loads(response.read())
                self.assertTrue(health["model"]["ready"])
                self.assertEqual(health["model"]["model"], "rwkv-test-hf")

                request = urllib.request.Request(
                    base + "/api/ask",
                    data=json.dumps(
                        {
                            "query": "RWKV 搜索是什么？",
                            "timezone": "Asia/Shanghai",
                            "history": [
                                {"role": "user", "content": "我们在讨论本地搜索。"},
                                {"role": "assistant", "content": "好的。"},
                            ],
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    body = response.read().decode("utf-8")
                self.assertIn('"used":true', body)
                self.assertIn('"new_tokens":24', body)
                self.assertIn("已通过前端服务调用", body)
                self.assertEqual(answerer.last_history[0]["role"], "user")
            finally:
                server.shutdown()
                server.server_close()

    def test_v1_forwards_live_model_deltas_and_conversation_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = SearchDatabase(Path(tmp) / "search.db")
            answerer = FakeStreamingRWKVAnswerer()
            server = SearchHTTPServer(
                ("127.0.0.1", 0), database, answerer=answerer
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                request = urllib.request.Request(
                    base + "/api/v1/chat/stream",
                    data=json.dumps(
                        {
                            "request_id": "req-live-stream",
                            "conversation_id": "conv-live-stream",
                            "message_id": "msg-live-stream",
                            "query": "你好",
                            "history": [],
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    events = self._sse_events(response.read().decode("utf-8"))
                deltas = [
                    event["delta"]
                    for event in events
                    if event["type"] == "answer_delta"
                ]
                self.assertEqual(deltas, ["真正", "逐 token", " 流式输出"])
                self.assertEqual(answerer.conversation_id, "conv-live-stream")
                final = next(
                    event for event in events if event["type"] == "answer_final"
                )
                self.assertEqual(final["answer"]["content"], "真正逐 token 流式输出")
            finally:
                server.shutdown()
                server.server_close()

    def test_backend_trace_records_prompt_and_raw_token_without_sse_debug(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = SearchDatabase(Path(tmp) / "search.db")
            answerer = FakeStreamingRWKVAnswerer()
            server = SearchHTTPServer(
                ("127.0.0.1", 0), database, answerer=answerer
            )
            trace_directory = Path(tmp) / "debug-traces"
            server.debug_traces = DebugTraceStore(str(trace_directory))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                request = urllib.request.Request(
                    base + "/api/v1/chat/stream",
                    data=json.dumps(
                        {
                            "request_id": "req-debug-stream",
                            "conversation_id": "conv-debug-stream",
                            "message_id": "msg-debug-stream",
                            "query": "你好",
                            "history": [],
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    events = self._sse_events(response.read().decode("utf-8"))
                self.assertNotIn("debug", [event["type"] for event in events])
                trace_records = [
                    json.loads(line)
                    for line in (trace_directory / "latest.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                debug = [
                    item["debug"]
                    for item in trace_records
                    if item.get("category") == "model_debug"
                ]
                self.assertTrue(any(item.get("kind") == "request_context" for item in debug))
                prompt = next(item for item in debug if item.get("kind") == "context_injection")
                self.assertEqual(prompt["input_token_ids"], [1, 2])
                token = next(item for item in debug if item.get("kind") == "token")
                self.assertEqual(token["token_id"], 3)
                self.assertEqual(trace_records[-1]["category"], "trace_finished")
            finally:
                server.shutdown()
                server.server_close()
