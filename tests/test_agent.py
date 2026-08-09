from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rwkv_agent.controller import (
    AgentController,
    parse_tool_call,
    policy_tool_gate,
    render_direct_answer_prompt,
    render_routing_context,
    render_session_context,
    strip_leading_think_blocks,
)
from rwkv_agent.memory import MemoryStore
from rwkv_agent.routing import (
    render_tool_gate_prompt,
    render_tool_gate_root,
    render_tool_gate_turn,
)
from rwkv_agent.tools.web import WebSearchAdapter


class MemoryStoreTests(unittest.TestCase):
    def test_memory_persists_and_is_session_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            first = MemoryStore(path)
            saved = first.save(
                "MVP retrieval budget is 3 rounds.",
                session_id="alpha",
            )
            second = MemoryStore(path)
            hits = second.search(
                "MVP retrieval budget",
                session_id="alpha",
            )
            self.assertEqual(hits[0].id, saved.id)
            self.assertEqual(
                second.search("MVP retrieval budget", session_id="beta"),
                [],
            )

    def test_empty_and_oversized_memory_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3")
            with self.assertRaises(ValueError):
                store.save("", session_id="alpha")
            with self.assertRaises(ValueError):
                store.save("x" * 4001, session_id="alpha")

    def test_conversation_history_is_persistent_ordered_and_session_scoped(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            first = MemoryStore(path)
            first.append_exchange(
                session_id="alpha",
                user="我是奶龙",
                assistant="好的。",
            )
            second = MemoryStore(path)
            history = second.history(session_id="alpha")
            self.assertEqual(
                [(item.role, item.content) for item in history],
                [("user", "我是奶龙"), ("assistant", "好的。")],
            )
            self.assertEqual(second.history(session_id="beta"), [])


class FakeWebEngine:
    def search_events(self, *args, **kwargs):
        yield {
            "type": "discovery_progress",
            "progress": {
                "candidates": [
                    {
                        "title": "Official release",
                        "snippet": "Version 1.2 is current.",
                        "url": "https://example.invalid/releases",
                        "rrf_score": 0.2,
                    }
                ]
            },
        }
        yield {"type": "realtime_result", "results": [], "stats": {}}

    def close(self) -> None:
        return


class WebAdapterTests(unittest.TestCase):
    def test_discovery_fallback_is_callable_evidence(self) -> None:
        adapter = WebSearchAdapter(engine=FakeWebEngine())
        result = adapter.execute("current release")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["retrieval"]["evidence_stage"], "discovery")
        self.assertEqual(result["evidence"][0]["id"], "W1")

    def test_web_trace_is_private_and_shadow_keeps_legacy_visible(self) -> None:
        class Shadow:
            def __init__(self):
                self.calls = []

            def submit(self, query, **kwargs):
                self.calls.append({"query": query, **kwargs})
                return {
                    "enabled": True,
                    "submitted": True,
                    "visible_strategy": "legacy",
                }

        shadow = Shadow()
        adapter = WebSearchAdapter(engine=FakeWebEngine(), shadow=shadow)
        result = adapter.execute("current release")
        self.assertEqual(result["evidence"][0]["uri"], "https://example.invalid/releases")
        self.assertNotIn("candidates", result)
        self.assertEqual(
            result["retrieval"]["shadow"]["visible_strategy"],
            "legacy",
        )
        self.assertEqual(
            shadow.calls[0]["legacy_trace"]["candidates"][0]["url"],
            "https://example.invalid/releases",
        )

    def test_original_question_repairs_model_invented_absolute_terms(self) -> None:
        class CaptureEngine(FakeWebEngine):
            def __init__(self) -> None:
                self.primary_query = ""
                self.execution_queries = []

            def search_events(self, query, queries, **kwargs):
                self.primary_query = query
                self.execution_queries = list(queries)
                yield from super().search_events(query, queries, **kwargs)

        engine = CaptureEngine()
        adapter = WebSearchAdapter(engine=engine, shadow=False)
        result = adapter.execute(
            "Microsoft FY2025 Q3 earnings release official investor relations",
            original_query="微软最近一个季度的业绩公告，请找官方投资者关系页面。",
        )
        self.assertEqual(
            engine.primary_query,
            "微软最近一个季度的业绩公告，请找官方投资者关系页面",
        )
        self.assertEqual(
            engine.execution_queries,
            ["Microsoft earnings release official investor relations"],
        )
        self.assertEqual(
            result["query_resolution"]["constraint_evaluation"][
                "removed_absolute_terms"
            ],
            ["FY2025", "Q3"],
        )
        self.assertTrue(
            result["query_resolution"]["constraint_evaluation"]["repair_applied"]
        )


class FakeModel:
    def __init__(
        self,
        *,
        use_tool: bool,
        routing_tool: str = "web_search",
    ):
        self.use_tool = use_tool
        self.routing_tool = routing_tool
        self.prompts: list[str] = []
        self.gate_messages: list[str] = []
        self.gate_requests: list[dict] = []

    def gate_tool(
        self,
        message: str,
        *,
        threshold: float = 0.7,
        context: str = "",
        has_pasted_text: bool = False,
    ) -> dict:
        self.gate_messages.append(message)
        self.gate_requests.append(
            {
                "message": message,
                "context": context,
                "has_pasted_text": has_pasted_text,
                "threshold": threshold,
            }
        )
        return {
            "use_tool": self.use_tool,
            "label": "tool" if self.use_tool else "chat",
            "margin": 3.0 if self.use_tool else -3.0,
            "threshold": threshold,
        }

    def health(self) -> dict:
        return {"status": "ready", "model": "fake"}

    def complete(self, prompt: str, *, max_tokens: int = 192) -> dict:
        self.prompts.append(prompt)
        if "Call exactly one function" in prompt:
            argument = (
                {
                    "question": "Who founded the base?",
                }
                if self.routing_tool == "long_text_qa"
                else {"query": "RWKV current release"}
            )
            raw = (
                "<tool_call>"
                f'{{"name":"{self.routing_tool}","arguments":'
                f"{__import__('json').dumps(argument)}}}"
                "</tool_call>"
            )
        else:
            raw = "你好！有什么可以帮助你的？"
        return {
            "raw": raw,
            "stop": "</s>",
            "output_tokens": 8,
            "model_elapsed_ms": 1.0,
            "request_elapsed_ms": 1.0,
            "model": "fake",
            "url": "fake://model",
        }


class FakeStateChatModel(FakeModel):
    def __init__(self, *, use_tool: bool = False) -> None:
        super().__init__(use_tool=use_tool)
        self.state_prefills: list[dict] = []
        self.state_continuations: list[dict] = []
        self.state_releases: list[str] = []
        self._state_counter = 0

    def state_prefill(self, *, owner_id: str, prompt: str) -> dict:
        self._state_counter += 1
        state_id = f"chat-state-{self._state_counter}"
        self.state_prefills.append(
            {
                "owner_id": owner_id,
                "prompt": prompt,
                "state_id": state_id,
            }
        )
        return {
            "state_id": state_id,
            "home_url": "fake://state-sidecar",
            "seen_tokens": len(prompt),
        }

    def state_chat_complete(self, **kwargs) -> dict:
        call = dict(kwargs)
        self.state_continuations.append(call)
        answer = f"state-answer-{len(self.state_continuations)}"
        return {
            "raw": answer,
            "stop": "</s>",
            "output_tokens": 4,
            "model_elapsed_ms": 1.0,
            "request_elapsed_ms": 1.0,
            "model": "fake-state",
            "url": kwargs["home_url"],
            "state_id": kwargs["state_id"],
            "seen_tokens": 100 + len(self.state_continuations),
        }

    def state_release(self, *, state_ids, **kwargs) -> dict:
        self.state_releases.extend(state_ids)
        return {"released": len(state_ids), "state_ids": list(state_ids)}


class FakeKnowledge:
    def execute(self, query: str) -> dict:
        return {
            "status": "ok",
            "evidence": [
                {
                    "id": "K1",
                    "title": "RWKV",
                    "content": query,
                    "uri": "knowledge://rwkv",
                }
            ],
        }


class FakeLongText:
    def execute(
        self,
        text: str,
        question: str,
        *,
        document_name: str = "pasted-text",
    ) -> dict:
        return {
            "status": "ok",
            "workers": {
                "submitted": 4,
                "completed": 4,
                "concurrency": 4,
                "candidates": 1,
                "errors": 0,
            },
            "evidence": [
                {
                    "id": "L1",
                    "title": f"{document_name} · chunk 3",
                    "content": f"{question}: answer",
                    "uri": "session-text://current#chunk=3",
                }
            ],
        }


class ControllerGateTests(unittest.TestCase):
    def controller(self, directory: str, model: FakeModel) -> AgentController:
        controller = AgentController(
            model_urls=["http://unused.invalid"],
            memory_path=str(Path(directory) / "memory.sqlite3"),
            web_adapter=WebSearchAdapter(engine=FakeWebEngine()),
            knowledge_adapter=FakeKnowledge(),
            long_text_adapter=FakeLongText(),
            long_text_capture_chars=256,
        )
        controller.model = model
        return controller

    def test_direct_gate_skips_all_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(use_tool=False)
            controller = self.controller(directory, model)
            result = controller.run("你好", session_id="gate-test")
            self.assertEqual(result["route"], {"mode": "direct", "tool": None})
            self.assertIsNone(result["tool_result"])
            self.assertEqual(result["answer"], "你好！有什么可以帮助你的？")
            self.assertEqual(result["trace"]["gate"]["label"], "chat")
            self.assertNotIn("<functions>", model.prompts[0])
            controller.close()

    def test_leading_think_is_hidden_without_relaxing_tool_protocol(self) -> None:
        block = (
            '<tool_call>{"name":"web_search","arguments":'
            '{"query":"RWKV creator GitHub"}}</tool_call>'
        )
        parsed = parse_tool_call(f"<think>Need current data.</think>\n{block}")
        self.assertTrue(parsed["strict"])
        self.assertTrue(parsed["reasoning_stripped"])
        self.assertEqual(parsed["arguments"]["query"], "RWKV creator GitHub")

        for invalid in (
            f"commentary\n{block}",
            f"<think>unfinished\n{block}",
            f"<think>done</think>\n{block}\ncommentary",
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(parse_tool_call(invalid)["strict"])

    def test_think_is_removed_from_visible_answers_history_and_router(self) -> None:
        class ThinkModel(FakeModel):
            def gate_tool(self, message: str, **kwargs) -> dict:
                self.use_tool = message != "你好"
                return super().gate_tool(message, **kwargs)

            def complete(self, prompt: str, *, max_tokens: int = 192) -> dict:
                result = super().complete(prompt, max_tokens=max_tokens)
                result["raw"] = "<think>private reasoning</think>\n" + result["raw"]
                return result

        with tempfile.TemporaryDirectory() as directory:
            model = ThinkModel(use_tool=False)
            controller = self.controller(directory, model)
            greeting = controller.run("你好", session_id="think-session")
            self.assertEqual(greeting["answer"], "你好！有什么可以帮助你的？")
            self.assertTrue(
                greeting["trace"]["answer_completion"]["reasoning_stripped"]
            )

            result = controller.run(
                "搜索RWKV创始人最热门的GitHub项目",
                session_id="think-session",
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["route"]["tool"], "web_search")
            self.assertTrue(
                result["trace"]["routing_completion"]["reasoning_stripped"]
            )
            self.assertNotIn("<think>", result["answer"])

            routing_prompt = next(
                prompt
                for prompt in reversed(model.prompts)
                if "Call exactly one function" in prompt
            )
            self.assertNotIn("private reasoning", routing_prompt)
            history = controller.memory.history(session_id="think-session")
            self.assertTrue(
                all("<think>" not in item.content for item in history)
            )
            controller.close()

    def test_controller_passes_original_question_to_web_query_guard(self) -> None:
        class InventedDateModel(FakeModel):
            def complete(self, prompt: str, *, max_tokens: int = 192) -> dict:
                if "Call exactly one function" in prompt:
                    return {
                        "raw": (
                            '<tool_call>{"name":"web_search","arguments":'
                            '{"query":"Microsoft FY2025 Q3 earnings release '
                            'official investor relations"}}</tool_call>'
                        ),
                        "stop": "</s>",
                        "output_tokens": 16,
                        "model_elapsed_ms": 1.0,
                        "request_elapsed_ms": 1.0,
                        "model": "fake",
                        "url": "fake://model",
                    }
                return super().complete(prompt, max_tokens=max_tokens)

        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(
                directory,
                InventedDateModel(use_tool=True),
            )
            message = "微软最近一个季度的业绩公告，请找官方投资者关系页面。"
            result = controller.run(message, session_id="query-guard")
            resolution = result["tool_result"]["query_resolution"]
            self.assertEqual(result["tool_result"]["original_query"], message)
            self.assertEqual(
                result["tool_result"]["effective_query"],
                "Microsoft earnings release official investor relations",
            )
            self.assertTrue(
                resolution["constraint_evaluation"]["repair_applied"]
            )
            controller.close()

    def test_web_answer_claim_gate_rejects_numeric_hallucination(self) -> None:
        class HallucinatingWebModel(FakeModel):
            def complete(self, prompt: str, *, max_tokens: int = 192) -> dict:
                result = super().complete(prompt, max_tokens=max_tokens)
                if "final evidence answer stage" in prompt:
                    result["raw"] = "当前版本是9.9。[W1]"
                return result

        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(
                directory,
                HallucinatingWebModel(use_tool=True),
            )
            result = controller.run(
                "搜索 current release",
                session_id="claim-gate",
            )

            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertEqual(
                result["answer"],
                "找到了一些相关信息，但现有证据不足以支持可靠结论。",
            )
            self.assertFalse(result["trace"]["answer_protocol"]["valid"])
            reasons = {
                claim["support_reason"]
                for claim in result["trace"]["answer_protocol"][
                    "claim_verification"
                ]
            }
            self.assertIn("number_mismatch", reasons)
            controller.close()

    def test_web_answer_claim_gate_keeps_grounded_answer(self) -> None:
        class GroundedWebModel(FakeModel):
            def complete(self, prompt: str, *, max_tokens: int = 192) -> dict:
                result = super().complete(prompt, max_tokens=max_tokens)
                if "final evidence answer stage" in prompt:
                    result["raw"] = "Version 1.2 is current. [W1]"
                return result

        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(
                directory,
                GroundedWebModel(use_tool=True),
            )
            result = controller.run(
                "搜索 current release",
                session_id="claim-gate-grounded",
            )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["answer"], "Version 1.2 is current. [W1]")
            self.assertTrue(result["trace"]["answer_protocol"]["valid"])
            controller.close()

    def test_direct_conversation_receives_prior_session_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(use_tool=False)
            controller = self.controller(directory, model)
            controller.run("我是奶龙", session_id="context-test")
            result = controller.run("我是谁？", session_id="context-test")
            self.assertEqual(result["trace"]["context"]["history_messages"], 2)
            self.assertIn("User: 我是奶龙", model.prompts[-1])
            self.assertIn("Assistant: 你好！有什么可以帮助你的？", model.prompts[-1])
            controller.close()

    def test_direct_chat_reuses_recurrent_state_without_history_prefill(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeStateChatModel()
            controller = self.controller(directory, model)

            first = controller.run("我是奶龙", session_id="state-chat")
            second = controller.run("我是谁？", session_id="state-chat")

            self.assertEqual(len(model.state_prefills), 1)
            self.assertEqual(len(model.state_continuations), 2)
            self.assertEqual(
                model.state_continuations[0]["state_id"],
                model.state_continuations[1]["state_id"],
            )
            self.assertEqual(
                model.state_continuations[0]["input_text"],
                "User: 我是奶龙\n\nAssistant:",
            )
            self.assertEqual(
                model.state_continuations[1]["input_text"],
                "\n\nUser: 我是谁？\n\nAssistant:",
            )
            self.assertEqual(model.prompts, [])
            self.assertEqual(
                first["trace"]["context"]["mode"],
                "recurrent_session_state",
            )
            self.assertFalse(
                first["trace"]["context"]["session_state"]["reused"]
            )
            self.assertTrue(
                second["trace"]["context"]["session_state"]["reused"]
            )
            self.assertTrue(
                second["trace"]["context"]["session_state"]["cached"]
            )
            self.assertEqual(
                second["trace"]["context"]["history_messages"],
                2,
            )
            controller.close()

    def test_tool_turn_releases_chat_state_and_next_chat_rebuilds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeStateChatModel()
            controller = self.controller(directory, model)
            controller.run("你好", session_id="mixed-session")
            first_state = model.state_prefills[0]["state_id"]

            model.use_tool = True
            model.routing_tool = "knowledge_search"
            tool_result = controller.run(
                "Search the local knowledge base for RWKV.",
                session_id="mixed-session",
            )
            self.assertEqual(tool_result["route"]["tool"], "knowledge_search")
            self.assertIn(first_state, model.state_releases)

            model.use_tool = False
            direct_result = controller.run(
                "继续刚才的话题",
                session_id="mixed-session",
            )
            self.assertEqual(len(model.state_prefills), 2)
            self.assertIn(
                "Search the local knowledge base for RWKV.",
                model.state_prefills[-1]["prompt"],
            )
            self.assertFalse(
                direct_result["trace"]["context"]["session_state"]["reused"]
            )
            controller.close()

    def test_expired_chat_state_rebuilds_once_from_transcript(self) -> None:
        class ExpiringStateModel(FakeStateChatModel):
            def state_chat_complete(self, **kwargs) -> dict:
                if (
                    kwargs["state_id"] == "chat-state-1"
                    and len(self.state_continuations) == 1
                ):
                    self.state_continuations.append(dict(kwargs))
                    raise KeyError("expired state")
                return super().state_chat_complete(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            model = ExpiringStateModel()
            controller = self.controller(directory, model)
            controller.run("第一轮", session_id="expiring-session")
            result = controller.run("第二轮", session_id="expiring-session")

            self.assertEqual(len(model.state_prefills), 2)
            self.assertIn("User: 第一轮", model.state_prefills[-1]["prompt"])
            self.assertIn("Assistant: state-answer-1", model.state_prefills[-1]["prompt"])
            self.assertTrue(
                result["trace"]["context"]["session_state"]["rebuilt"]
            )
            self.assertEqual(
                result["trace"]["context"]["session_state"]["fallback_reason"],
                "",
            )
            self.assertEqual(model.prompts, [])
            controller.close()

    def test_chat_state_continues_after_a_committed_user_stop(self) -> None:
        class UserStopModel(FakeStateChatModel):
            def state_chat_complete(self, **kwargs) -> dict:
                result = super().state_chat_complete(**kwargs)
                result["stop"] = "\n\nUser:"
                return result

        with tempfile.TemporaryDirectory() as directory:
            model = UserStopModel()
            controller = self.controller(directory, model)
            controller.run("第一轮", session_id="user-stop")
            controller.run("第二轮", session_id="user-stop")
            self.assertEqual(
                model.state_continuations[-1]["input_text"],
                " 第二轮\n\nAssistant:",
            )
            self.assertEqual(len(model.state_prefills), 1)
            controller.close()

    def test_hidden_reasoning_state_is_released_instead_of_reused(self) -> None:
        class HiddenReasoningModel(FakeStateChatModel):
            def state_chat_complete(self, **kwargs) -> dict:
                result = super().state_chat_complete(**kwargs)
                result["raw"] = "<think>private</think>\nvisible"
                return result

        with tempfile.TemporaryDirectory() as directory:
            model = HiddenReasoningModel()
            controller = self.controller(directory, model)
            first = controller.run("第一轮", session_id="reasoning-state")
            second = controller.run("第二轮", session_id="reasoning-state")
            self.assertEqual(first["answer"], "visible")
            self.assertEqual(second["answer"], "visible")
            self.assertEqual(len(model.state_prefills), 2)
            self.assertEqual(len(model.state_releases), 2)
            self.assertEqual(
                first["trace"]["context"]["session_state"][
                    "cache_reject_reason"
                ],
                "hidden_reasoning_was_generated",
            )
            controller.close()

    def test_recurrent_chat_states_are_session_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeStateChatModel()
            controller = self.controller(directory, model)
            controller.run("alpha-1", session_id="alpha")
            controller.run("beta-1", session_id="beta")
            controller.run("alpha-2", session_id="alpha")
            self.assertEqual(len(model.state_prefills), 2)
            alpha_state = model.state_prefills[0]["state_id"]
            beta_state = model.state_prefills[1]["state_id"]
            self.assertNotEqual(alpha_state, beta_state)
            self.assertEqual(
                model.state_continuations[0]["state_id"],
                alpha_state,
            )
            self.assertEqual(
                model.state_continuations[1]["state_id"],
                beta_state,
            )
            self.assertEqual(
                model.state_continuations[2]["state_id"],
                alpha_state,
            )
            controller.close()

    def test_tool_gate_continues_to_two_tool_router(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(use_tool=True, routing_tool="knowledge_search")
            controller = self.controller(directory, model)
            result = controller.run(
                "Search the local knowledge base for RWKV.",
                session_id="gate-test",
            )
            self.assertEqual(result["route"]["mode"], "tool")
            self.assertEqual(result["route"]["tool"], "knowledge_search")
            self.assertEqual(result["tool_result"]["status"], "ok")
            self.assertEqual(result["trace"]["gate"]["label"], "tool")
            controller.close()

    def test_pasted_text_routes_with_question_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(use_tool=True, routing_tool="long_text_qa")
            controller = self.controller(directory, model)
            captured = controller.run(
                "红岸工程第147次常规发射。" + "长文本" * 100,
                session_id="long-text-test",
            )
            self.assertEqual(
                captured["route"]["mode"],
                "document_capture",
            )
            self.assertFalse(captured["trace"]["model_called"])
            self.assertNotIn("红岸工程第147次", "\n".join(model.prompts))
            result = controller.run(
                "Who founded the base?",
                session_id="long-text-test",
            )
            self.assertEqual(result["route"]["tool"], "long_text_qa")
            self.assertEqual(
                result["route"]["arguments"],
                {
                    "question": "Who founded the base?",
                },
            )
            self.assertEqual(result["tool_result"]["evidence"][0]["id"], "L1")
            self.assertTrue(model.gate_requests[-1]["has_pasted_text"])
            routing_prompt = next(
                prompt
                for prompt in model.prompts
                if "Call exactly one function" in prompt
            )
            self.assertIn("Active pasted long text: yes.", routing_prompt)
            self.assertNotIn("红岸工程第147次", routing_prompt)
            controller.close()

    def test_active_pasted_text_does_not_force_an_unrelated_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(use_tool=False)
            controller = self.controller(directory, model)
            controller.run(
                "红岸工程第147次常规发射。" + "长文本" * 100,
                session_id="long-text-chat-test",
            )
            result = controller.run("你好", session_id="long-text-chat-test")
            self.assertEqual(result["route"], {"mode": "direct", "tool": None})
            self.assertIsNone(result["tool_result"])
            controller.close()

    def test_pasted_text_is_not_visible_in_another_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(
                directory,
                FakeModel(use_tool=False),
            )
            controller.run(
                "secret-source-" + "x" * 300,
                session_id="alpha",
            )
            result = controller.execute_tool(
                "long_text_qa",
                {"question": "What is the secret?"},
                session_id="beta",
            )
            self.assertEqual(result["status"], "empty")
            controller.close()

    def test_direct_long_text_tool_rejects_legacy_path_argument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(
                directory,
                FakeModel(use_tool=False),
            )
            controller.run(
                "source-" + "x" * 300,
                session_id="alpha",
            )
            result = controller.execute_tool(
                "long_text_qa",
                {
                    "path": "/tmp/source.txt",
                    "question": "What does it say?",
                },
                session_id="alpha",
            )
            self.assertEqual(result["status"], "invalid")
            controller.close()

    def test_direct_prompt_forbids_fake_search_and_citations(self) -> None:
        prompt = render_direct_answer_prompt("Hello")
        self.assertIn("Do not claim to have searched", prompt)
        self.assertIn("do not invent sources or citation IDs", prompt)

    def test_routing_context_is_recent_and_bounded(self) -> None:
        class Entry:
            def __init__(self, role: str, content: str) -> None:
                self.role = role
                self.content = content

        context = render_routing_context(
            [
                Entry("user", "old-" + "x" * 2500),
                Entry("assistant", "recent entity is RWKV"),
            ]
        )
        self.assertLessEqual(len(context), 2000)
        self.assertIn("recent entity is RWKV", context)

    def test_existing_think_history_is_hidden_from_both_contexts(self) -> None:
        class Entry:
            role = "assistant"
            content = "<think>private chain</think>\nVisible answer"

        visible, stripped = strip_leading_think_blocks(Entry.content)
        self.assertTrue(stripped)
        self.assertEqual(visible, "Visible answer")
        self.assertEqual(render_routing_context([Entry()]), "Assistant: Visible answer")
        self.assertEqual(render_session_context([Entry()]), "Assistant: Visible answer")

    def test_semantic_gate_prompt_uses_context_and_not_keyword_policy(self) -> None:
        prompt = render_tool_gate_prompt(
            "那它是谁创建的？",
            context="User: 我们在讨论RWKV。",
            has_pasted_text=True,
        )
        self.assertIn("Decide from meaning", prompt)
        self.assertIn("Mamba架构最初是谁提出的？\nAssistant: search", prompt)
        self.assertIn("把“Mamba架构是谁提出的”翻译成英文。\nAssistant: chat", prompt)
        self.assertIn("Recent conversation reference", prompt)
        self.assertIn("Active pasted long text: yes", prompt)
        self.assertIn("Current user request: 那它是谁创建的？", prompt)
        turn = render_tool_gate_turn(
            "那它是谁创建的？",
            context="User: 我们在讨论RWKV。",
            has_pasted_text=True,
        )
        self.assertNotIn("Mamba架构最初是谁提出的？", turn)
        self.assertEqual(prompt, render_tool_gate_root() + turn)

    def test_policy_gate_only_applies_explicit_ui_search_mode(self) -> None:
        for message in (
            "你好",
            "把“今天天气很好”翻译成英文",
            "Implement binary search in Python.",
            "Search the local knowledge base for RWKV.",
        ):
            self.assertIsNone(policy_tool_gate(message), message)
        forced = policy_tool_gate("你好", search_mode="always")
        self.assertEqual(forced["label"], "tool")
        self.assertTrue(forced["forced"])
        disabled = policy_tool_gate("查最新新闻", search_mode="never")
        self.assertEqual(disabled["label"], "chat")
        self.assertTrue(disabled["forced"])

    def test_tool_parser_exposes_only_bounded_tools(self) -> None:
        valid = parse_tool_call(
            '<tool_call>{"name":"knowledge_search","arguments":'
            '{"query":"RWKV Agent"}}</tool_call>'
        )
        self.assertTrue(valid["strict"])
        self.assertEqual(valid["arguments"]["query"], "RWKV Agent")
        long_text = parse_tool_call(
            '<tool_call>{"name":"long_text_qa","arguments":'
            '{"question":"Who founded it?"}}'
            "</tool_call>"
        )
        self.assertTrue(long_text["strict"])
        self.assertEqual(
            long_text["arguments"]["question"],
            "Who founded it?",
        )
        legacy_path = parse_tool_call(
            '<tool_call>{"name":"long_text_qa","arguments":'
            '{"path":"/tmp/source.txt","question":"Who founded it?"}}'
            "</tool_call>"
        )
        self.assertFalse(legacy_path["strict"])
        invalid = parse_tool_call(
            '<tool_call>{"name":"memory","arguments":'
            '{"action":"read","text":"answer preference"}}</tool_call>'
        )
        self.assertFalse(invalid["strict"])

    def test_gate_and_router_receive_bounded_context_for_followups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(use_tool=True, routing_tool="web_search")
            controller = self.controller(directory, model)
            controller.run("我是奶龙", session_id="router-isolation")
            model.gate_messages.clear()
            result = controller.run(
                "它怎么样？",
                session_id="router-isolation",
            )
            self.assertEqual(result["route"]["tool"], "web_search")
            self.assertEqual(result["trace"]["gate"]["source"], "g1i")
            self.assertEqual(model.gate_messages[-1], "它怎么样？")
            self.assertIn("我是奶龙", model.gate_requests[-1]["context"])
            routing_prompt = next(
                prompt
                for prompt in reversed(model.prompts)
                if "Call exactly one function" in prompt
            )
            self.assertIn("Use this recent conversation", routing_prompt)
            self.assertIn("我是奶龙", routing_prompt)
            self.assertIn("User: 它怎么样？", routing_prompt)
            controller.close()

    def test_model_specific_gate_threshold_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(use_tool=True)
            controller = self.controller(directory, model)
            controller.tool_gate_threshold = -3.2
            controller.decide_tool("它怎么样？")
            self.assertEqual(model.gate_requests[-1]["threshold"], -3.2)
            self.assertEqual(controller.health()["tool_gate"]["threshold"], -3.2)
            controller.close()

    def test_long_term_memory_is_dormant_in_context_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(use_tool=False)
            controller = self.controller(directory, model)
            controller.memory.save(
                "回答尽量简短",
                session_id="profile-test",
            )
            result = controller.run("你好", session_id="profile-test")
            self.assertEqual(
                result["trace"]["context"]["mode"],
                "session_transcript",
            )
            self.assertNotIn("<core_memory>", model.prompts[-1])
            self.assertNotIn("回答尽量简短", model.prompts[-1])
            controller.close()

    def test_direct_answer_does_not_extract_long_term_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = FakeModel(use_tool=False)
            controller = self.controller(directory, model)
            result = controller.run(
                "以后回答尽量短一点。",
                session_id="implicit-memory",
            )
            self.assertEqual(
                result["trace"]["memory_consolidation"],
                {
                    "enabled": False,
                    "reason": "context_only_mode",
                },
            )
            self.assertEqual(
                controller.memory.recent(
                    session_id="implicit-memory",
                    limit=5,
                ),
                [],
            )
            controller.close()

    def test_memory_function_is_disabled_but_transcript_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(
                directory,
                FakeModel(use_tool=False),
            )
            disabled = controller.execute_tool(
                "memory",
                {"action": "write", "text": "回答尽量简短"},
                session_id="alpha",
            )
            self.assertEqual(disabled["status"], "disabled")
            controller.run("我是奶龙", session_id="alpha")
            self.assertEqual(
                len(controller.memory.history(session_id="alpha")),
                2,
            )
            controller.close()


if __name__ == "__main__":
    unittest.main()
