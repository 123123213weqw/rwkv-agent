from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path

from rwkv_search.protocol import (
    SCHEMA_VERSION,
    ChatRequest,
    EventFactory,
    ProtocolError,
    RequestRegistry,
)


class ProtocolTests(unittest.TestCase):
    def test_contract_files_are_present_and_parseable(self) -> None:
        root = Path(__file__).resolve().parents[1] / "contracts"
        for name in (
            "chat_request.schema.json",
            "chat_event.schema.json",
            "source.schema.json",
            "evidence.schema.json",
            "error_codes.json",
        ):
            value = json.loads((root / name).read_text(encoding="utf-8"))
            self.assertIsInstance(value, dict)
        event_types = json.loads(
            (root / "chat_event.schema.json").read_text(encoding="utf-8")
        )["properties"]["type"]["enum"]
        self.assertIn("answer_final", event_types)
        self.assertIn("fetch_progress", event_types)

    def test_chat_request_defaults_normalizes_and_validates(self) -> None:
        request = ChatRequest.from_payload(
            {
                "schema_version": "1.0",
                "conversation_id": "conv-1",
                "message_id": "msg-1",
                "query": "  今天   星期几？ ",
                "history": [
                    {"role": "system", "content": "ignored"},
                    {"role": "user", "content": "  上一条   问题  "},
                ],
                "search_mode": "auto",
                "research_depth": "fast",
                "timezone": "Asia/Shanghai",
                "locale": "zh-CN",
            }
        )
        self.assertEqual(request.schema_version, SCHEMA_VERSION)
        self.assertEqual(request.query, "今天 星期几？")
        self.assertEqual(request.source_scope, "auto")
        self.assertFalse(request.use_finewiki)
        self.assertEqual(request.history, [{"role": "user", "content": "上一条 问题"}])
        self.assertTrue(request.request_id.startswith("req_"))

        with self.assertRaises(ProtocolError) as captured:
            ChatRequest.from_payload(
                {"query": "hello", "search_mode": "sometimes"}
            )
        self.assertEqual(captured.exception.code, "INVALID_REQUEST")
        self.assertEqual(captured.exception.field, "search_mode")

        enabled = ChatRequest.from_payload(
            {"query": "Python 是什么", "use_finewiki": True}
        )
        self.assertTrue(enabled.use_finewiki)
        with self.assertRaises(ProtocolError) as finewiki_error:
            ChatRequest.from_payload(
                {"query": "Python 是什么", "use_finewiki": "true"}
            )
        self.assertEqual(finewiki_error.exception.field, "use_finewiki")

    def test_event_sequence_and_request_registry(self) -> None:
        request = ChatRequest.from_payload({"query": "你好"})
        factory = EventFactory(request)
        first = factory.make("request_started")
        second = factory.make("route", route={})
        self.assertEqual([first["sequence"], second["sequence"]], [1, 2])
        self.assertEqual(second["request_id"], request.request_id)

        registry = RequestRegistry()
        cancel_event = registry.register("req-test")
        self.assertIsInstance(cancel_event, threading.Event)
        self.assertTrue(registry.cancel("req-test"))
        self.assertTrue(cancel_event.is_set())
        registry.finish("req-test")
        self.assertFalse(registry.cancel("req-test"))
