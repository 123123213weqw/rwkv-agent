from __future__ import annotations

import json
import unittest

from rwkv_agent.controller import parse_tool_call
from rwkv_agent.state_agent import (
    ANSWER_MAX_TOKENS,
    ANSWER_STOPS,
    StateNativeSearchAgent,
    _merge_evidence,
    attach_evidence_citations,
    compact_answer_evidence,
    coordinate_search_query,
    coordinate_answer_output,
    reconstruct_tool_call,
    render_answer_fallback_prompt,
    render_compact_answer_prompt,
    render_branch_step,
    render_root_final_input,
    render_root_prompt,
    validate_answer_output,
)


class FakeStateModel:
    def __init__(self) -> None:
        self.calls = []
        self.rounds: dict[str, int] = {}
        self.released = []
        self.fallback_calls = []

    def state_prefill(self, *, owner_id: str, prompt: str):
        self.calls.append(("prefill", owner_id, prompt))
        return {"state_id": "root", "home_url": "fake://sidecar"}

    def state_fork(
        self,
        *,
        home_url: str,
        owner_id: str,
        parent_state_id: str,
        branches: list[str],
    ):
        self.calls.append(("fork", home_url, parent_state_id, list(branches)))
        return [
            {"state_id": f"child-{index}", "branch": branch}
            for index, branch in enumerate(branches, 1)
        ]

    def state_batch_continue(
        self,
        *,
        home_url: str,
        owner_id: str,
        items: list[dict[str, str]],
        stops: list[str],
        max_tokens: int,
    ):
        self.calls.append(("continue", home_url, list(items), list(stops)))
        output = []
        for item in items:
            state_id = item["state_id"]
            if state_id == "root":
                output.append(
                    {
                        "state_id": "root",
                        "branch": "root",
                        "text": "The verified answer is supported [W1].",
                        "stop_reason": "</answer>",
                        "seen_tokens": 50,
                    }
                )
                continue
            round_index = self.rounds.get(state_id, 0) + 1
            self.rounds[state_id] = round_index
            query = f"project {state_id} round {round_index}"
            raw = (
                '<tool_call>{"name":"web_search","arguments":'
                + json.dumps({"query": query}, separators=(",", ":"))
                + "}"
            )
            output.append(
                {
                    "state_id": state_id,
                    "branch": state_id,
                    "text": raw,
                    "stop_reason": "</tool_call>",
                    "seen_tokens": 20 + round_index,
                }
            )
        return output

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stops: list[str] | None = None,
    ):
        self.fallback_calls.append((prompt, max_tokens, list(stops or [])))
        return {
            "raw": "The fallback answer is supported [W1].",
            "stop": "</answer>",
            "output_tokens": 8,
        }

    def state_release(
        self,
        *,
        home_url: str,
        owner_id: str,
        state_ids: list[str],
    ):
        self.released.append((home_url, owner_id, list(state_ids)))
        return {"status": "ok", "released": len(state_ids)}


class StateNativeSearchAgentTests(unittest.TestCase):
    def test_compact_answer_evidence_preserves_sources_and_ranks_spans(self) -> None:
        evidence = [
            {
                "id": "W1",
                "title": "Morning project",
                "content": (
                    "Unrelated background. " * 30
                    + "The Morning project launched on September 15. "
                    + "More unrelated background. " * 30
                ),
                "uri": "https://example.invalid/morning",
            }
        ]
        compact = compact_answer_evidence(
            "When did the Morning project launch?",
            evidence,
            max_chars_per_source=360,
        )
        self.assertEqual(compact[0]["id"], "W1")
        self.assertEqual(compact[0]["uri"], evidence[0]["uri"])
        self.assertLessEqual(len(compact[0]["content"]), 360)
        self.assertIn("September 15", compact[0]["content"])
        prompt = render_compact_answer_prompt("When?", compact)
        self.assertTrue(prompt.endswith("Assistant: <answer>"))
        self.assertNotIn("ExampleDB", prompt)

    def test_evidence_merge_ranks_relevance_and_caps_one_domain(self) -> None:
        evidence = _merge_evidence(
            [
                {
                    "evidence": [
                        {
                            "title": f"Generic discussion {index}",
                            "content": "unrelated animation forum",
                            "uri": f"https://www.zhihu.com/question/{index}",
                        }
                        for index in range(5)
                    ]
                },
                {
                    "evidence": [
                        {
                            "title": "Wickham Skinner Awards | POMS",
                            "content": "Wickham Skinner Award winners and years.",
                            "uri": "https://www.poms.org/awards/skinner",
                        }
                    ]
                },
            ],
            question="Wickham Skinner Award winners in 2016 and 2022?",
            limit=4,
        )

        self.assertEqual(evidence[0]["uri"], "https://www.poms.org/awards/skinner")
        self.assertLessEqual(
            sum("zhihu.com" in item["uri"] for item in evidence),
            3,
        )

    def test_evidence_merge_uses_generated_query_views_without_source_shapes(self) -> None:
        generic = [
            {
                "title": f"RWKV discussion {index}",
                "content": "RWKV creator projects recent news",
                "uri": f"https://news.example/{index}",
                "score": 1.0,
            }
            for index in range(12)
        ]
        github = [
            {
                "title": "PENG Bo (@BlinkDL)",
                "content": "GitHub user BlinkDL",
                "uri": "https://github.com/BlinkDL",
                "source": "github",
                "discovery_stage": "github_owner_profile",
            },
            {
                "title": "BlinkDL GitHub repositories",
                "content": "Public repositories: RWKV-LM, ChatRWKV, RWKV-CUDA",
                "uri": "https://github.com/BlinkDL?tab=repositories",
                "source": "github",
                "discovery_stage": "github_owner_repository_index",
            },
            {
                "title": "Commit 1234: Update README",
                "content": "Update README",
                "uri": "https://github.com/BlinkDL/RWKV-LM/commit/1234",
                "source": "github",
                "published_at": "2026-07-27T01:00:00Z",
                "discovery_stage": "github_latest_commit",
            },
            {
                "title": "Commit 1220: Older update",
                "content": "Older update",
                "uri": "https://github.com/BlinkDL/RWKV-LM/commit/1220",
                "source": "github",
                "published_at": "2026-07-26T01:00:00Z",
                "discovery_stage": "github_latest_commit",
            },
            {
                "title": "Commit 1210: Even older update",
                "content": "Even older update",
                "uri": "https://github.com/BlinkDL/RWKV-LM/commit/1210",
                "source": "github",
                "published_at": "2026-07-25T01:00:00Z",
                "discovery_stage": "github_latest_commit",
            },
        ]

        evidence = _merge_evidence(
            [
                {
                    "query": "RWKV founder BlinkDL identity",
                    "evidence": generic[:4] + [github[0]],
                },
                {
                    "query": "BlinkDL GitHub repositories complete list",
                    "evidence": generic[4:8] + [github[1]],
                },
                {
                    "query": "BlinkDL newest commit update timestamp",
                    "evidence": generic[8:] + github[2:],
                },
            ],
            question="RWKV创始人是谁，他的GitHub项目和最新更新是什么？",
            limit=8,
        )
        uris = {item["uri"] for item in evidence}

        self.assertIn("https://github.com/BlinkDL", uris)
        self.assertIn("https://github.com/BlinkDL?tab=repositories", uris)
        self.assertIn("https://github.com/BlinkDL/RWKV-LM/commit/1234", uris)
        commit = next(
            item
            for item in evidence
            if item["uri"].endswith("/commit/1234")
        )
        self.assertEqual(commit["published_at"], "2026-07-27T01:00:00Z")

    def test_compact_answer_evidence_uses_one_generic_source_budget(self) -> None:
        records = ", ".join(f"record-{index}" for index in range(35))
        evidence = compact_answer_evidence(
            "What records are in the catalog?",
            [
                {
                    "id": "W1",
                    "title": "Complete catalog",
                    "content": f"Public records (35): {records}. FINAL_RECORD",
                    "uri": "https://example.invalid/catalog",
                }
            ],
            max_chars_per_source=900,
        )

        self.assertIn("FINAL_RECORD", evidence[0]["content"])
        self.assertGreater(len(evidence[0]["content"]), 360)

    def test_query_coordinator_builds_complementary_first_round_views(self) -> None:
        used: set[str] = set()
        values = [
            coordinate_search_query(
                "ACL 2025 Industry Track submission deadline",
                "ACL 2025 Industry Track deadline",
                branch_index=index,
                round_index=1,
                observation=None,
                used_queries=used,
            )
            for index in range(4)
        ]
        self.assertEqual(
            [value[1] for value in values],
            ["model", "exact_anchors", "primary_source", "raw_question"],
        )
        self.assertEqual(len({value[0].casefold() for value in values}), 4)
        self.assertTrue(all("ACL" in value[0] for value in values))

    def test_answer_coordinator_normalizes_groups_and_drops_unknown_ids(self) -> None:
        coordinated = coordinate_answer_output(
            "Supported claim [W1, W9]. Another claim [W2; W2].",
            [
                {"id": "W1", "title": "one", "content": "Supported claim."},
                {"id": "W2", "title": "two", "content": "Another claim."},
            ],
        )
        self.assertTrue(coordinated["valid"])
        self.assertTrue(coordinated["citation_sanitized"])
        self.assertEqual(coordinated["citations"], ["W1", "W2"])
        self.assertIn("[W1]", coordinated["answer"])
        self.assertIn("[W2]", coordinated["answer"])
        self.assertNotIn("W9", coordinated["answer"])

    def test_answer_coordinator_cites_each_uncited_sentence(self) -> None:
        coordinated = coordinate_answer_output(
            "The launch was in 2025. The model was B [W2].",
            [
                {"id": "W1", "title": "Launch", "content": "The launch was in 2025."},
                {"id": "W2", "title": "Model", "content": "The model was B."},
            ],
        )
        self.assertTrue(coordinated["valid"])
        self.assertTrue(coordinated["citation_repaired"])
        self.assertIn("2025. [W1]", coordinated["answer"])
        self.assertIn("B [W2]", coordinated["answer"])

    def test_answer_coordinator_splits_chinese_sentences_for_claim_citations(self) -> None:
        coordinated = coordinate_answer_output(
            "甲公司由林岚创立。主要产品是星图。最新公告发布于2026年7月。",
            [
                {"id": "W1", "title": "创始人", "content": "甲公司由林岚创立。"},
                {"id": "W2", "title": "产品", "content": "主要产品是星图。"},
                {"id": "W3", "title": "公告", "content": "最新公告发布于2026年7月。"},
            ],
        )
        self.assertTrue(coordinated["valid"])
        answer = coordinated["answer"]
        self.assertIn("创立。 [W1]", answer)
        self.assertIn("星图。 [W2]", answer)
        self.assertIn("7月。 [W3]", answer)

    def test_answer_coordinator_rejects_cited_but_unsupported_claim(self) -> None:
        coordinated = coordinate_answer_output(
            "从唐镇站乘坐地铁2号线，全程约90分钟，票价10元。[W1]",
            [
                {
                    "id": "W1",
                    "title": "唐镇规划",
                    "content": "唐镇位于浦东新区，规划面积32.32平方公里。",
                }
            ],
        )

        self.assertFalse(coordinated["valid"])
        self.assertIn("unsupported_claim", coordinated["errors"])
        self.assertEqual(coordinated["unsupported_claim_count"], 1)

    def test_answer_coordinator_reattributes_a_supported_claim(self) -> None:
        coordinated = coordinate_answer_output(
            "最新更新是2026-07-23，提交了README.md更新。[W1]",
            [
                {
                    "id": "W1",
                    "title": "项目主页",
                    "content": "项目主页与仓库列表。",
                },
                {
                    "id": "W2",
                    "title": "Commit: Update README.md",
                    "content": "Update README.md",
                    "published_at": "2026-07-23T08:56:31Z",
                },
            ],
        )

        self.assertTrue(coordinated["valid"])
        self.assertEqual(coordinated["citation_reattributed_count"], 1)
        self.assertIn("[W2]", coordinated["answer"])
        self.assertNotIn("[W1]", coordinated["answer"])

    def test_answer_coordinator_keeps_supported_subset(self) -> None:
        coordinated = coordinate_answer_output(
            "RWKV的创始人是彭博。[W1] 从唐镇乘地铁2号线需90分钟。[W2]",
            [
                {
                    "id": "W1",
                    "title": "RWKV",
                    "content": "RWKV的创始人是彭博（Bo Peng）。",
                },
                {
                    "id": "W2",
                    "title": "唐镇规划",
                    "content": "唐镇位于上海浦东新区。",
                },
            ],
        )

        self.assertTrue(coordinated["valid"])
        self.assertTrue(coordinated["partial_answer"])
        self.assertEqual(coordinated["dropped_claim_count"], 1)
        self.assertIn("彭博", coordinated["answer"])
        self.assertNotIn("地铁", coordinated["answer"])

    def test_partial_answer_drops_orphaned_list_fragments(self) -> None:
        coordinated = coordinate_answer_output(
            "创始人是彭博。[W1] 他的项目包括：[W2]\nRWKV-LM [W2]",
            [
                {"id": "W1", "content": "创始人是彭博。"},
                {"id": "W2", "content": "RWKV-LM 是相关项目。"},
            ],
        )

        self.assertTrue(coordinated["partial_answer"])
        self.assertIn("彭博", coordinated["answer"])
        self.assertNotIn("RWKV-LM", coordinated["answer"])

    def test_branch_step_forces_strict_tool_call_prefix(self) -> None:
        step = render_branch_step(
            question="谁创建了RWKV？",
            mission="Find the primary source.",
            round_index=1,
            observation=None,
        )
        self.assertTrue(step.endswith("Assistant: <tool_call>"))
        self.assertIn("arguments object must contain only query", step)
        self.assertEqual(
            reconstruct_tool_call(
                {
                    "text": '{"name":"web_search","arguments":'
                    '{"query":"RWKV creator"}}',
                    "stop_reason": "</tool_call>",
                }
            ),
            '<tool_call>{"name":"web_search","arguments":'
            '{"query":"RWKV creator"}}</tool_call>',
        )
        root = render_root_prompt("谁创建了RWKV？")
        self.assertIn("Never call a function from the retained root", root)
        self.assertNotIn('"name":"web_search"', root)

        final_input = render_root_final_input(
            "谁创建了RWKV？",
            [
                {
                    "id": "W1",
                    "title": "RWKV",
                    "content": "RWKV evidence",
                    "uri": "https://example.invalid/rwkv",
                }
            ],
        )
        self.assertTrue(final_input.endswith("Assistant: <answer>"))
        self.assertIn("never reproduce the Tool Result", final_input)

    def test_answer_validator_blocks_protocol_json_and_invalid_ids(self) -> None:
        evidence = [{"id": "W1"}, {"id": "W2"}]
        valid = validate_answer_output(
            "<answer>RWKV is supported [W1].</answer>",
            evidence,
        )
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["answer"], "RWKV is supported [W1].")
        self.assertEqual(valid["citations"], ["W1"])

        grouped = validate_answer_output(
            "RWKV is supported [W1, W2].",
            evidence,
        )
        self.assertTrue(grouped["valid"])
        self.assertEqual(grouped["citations"], ["W1", "W2"])

        missing = validate_answer_output(
            "RWKV is supported but the citation was omitted.",
            evidence,
        )
        self.assertFalse(missing["valid"])
        self.assertIn("missing_citation", missing["errors"])

        repaired = coordinate_answer_output(
            "RWKV is maintained by Example Foundation.",
            [
                {
                    "id": "W1",
                    "title": "RWKV",
                    "content": "Example Foundation maintains RWKV.",
                },
                {"id": "W2", "title": "Unrelated", "content": "weather"},
            ],
        )
        self.assertTrue(repaired["valid"])
        self.assertTrue(repaired["citation_repaired"])
        self.assertEqual(repaired["citations"], ["W1"])
        self.assertEqual(
            attach_evidence_citations("Fact one.\nFact two.", evidence).count("[W"),
            2,
        )

        invalid_grouped = validate_answer_output(
            "RWKV is supported [W1, W9].",
            evidence,
        )
        self.assertFalse(invalid_grouped["valid"])
        self.assertEqual(invalid_grouped["invalid_citations"], ["W9"])

        cases = (
            '<tool_result>{"status":"ok","evidence":[]}</tool_result>',
            '{"status":"ok","evidence":[]}',
            "Tool: copied observation",
            "Unsupported citation [W9].",
            "<tool_call>{}</tool_call>",
        )
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertFalse(validate_answer_output(raw, evidence)["valid"])

    def test_fallback_prompt_precommits_answer_envelope(self) -> None:
        prompt = render_answer_fallback_prompt(
            "谁维护RWKV？",
            [
                {
                    "id": "W1",
                    "title": "RWKV",
                    "content": "RWKV is maintained by its organization.",
                    "uri": "https://example.invalid/rwkv",
                }
            ],
        )
        self.assertTrue(prompt.endswith("Assistant: <answer>"))
        self.assertIn("Never output Tool Result", prompt)

    def test_root_fork_multi_round_search_and_root_resume(self) -> None:
        model = FakeStateModel()
        tool_queries = []

        def execute_tool(name: str, arguments: dict, *, session_id: str):
            self.assertEqual(name, "web_search")
            tool_queries.append(arguments["query"])
            index = len(tool_queries)
            return {
                "status": "ok",
                "evidence": [
                    {
                        "id": "W1",
                        "title": f"Source {index}",
                        "content": (
                            f"{arguments['query']}. "
                            "The verified answer is supported. "
                            "The fallback answer is supported."
                        ),
                        "uri": f"https://example.invalid/{index}",
                    }
                ],
            }

        agent = StateNativeSearchAgent(
            state_model=model,
            parse_tool_call=parse_tool_call,
            execute_tool=execute_tool,
        )
        result = agent.run(
            "Who created the project?",
            session_id="session-a",
            branch_width=4,
            max_rounds=2,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["route"]["mode"], "state_parallel_search")
        self.assertEqual(result["route"]["branch_width"], 4)
        self.assertGreaterEqual(len(tool_queries), 7)
        self.assertEqual(
            len(result["tool_result"]["evidence"]),
            len(tool_queries),
        )
        self.assertTrue(
            all(
                len(item["content"]) <= 360
                for item in result["tool_result"]["evidence"]
            )
        )
        self.assertEqual(
            result["trace"]["answer_evidence_profile"]["name"],
            "compact-question-span-v1",
        )
        self.assertEqual(result["answer"], "The verified answer is supported [W1].")
        self.assertEqual(len(result["trace"]["rounds"]), 2)
        query_views = [
            branch["route"].get("query_view")
            for round_trace in result["trace"]["rounds"]
            for branch in round_trace["branches"]
        ]
        self.assertTrue(all(view for view in query_views))
        self.assertEqual(
            len({query.casefold() for query in tool_queries}),
            len(tool_queries),
        )
        self.assertEqual(
            result["trace"]["state_runtime"]["release"]["released"],
            5,
        )
        self.assertEqual(model.released[0][0], "fake://sidecar")
        self.assertEqual(
            model.released[0][2],
            [
                "root",
                "child-1",
                "child-2",
                "child-3",
                "child-4",
            ],
        )
        root_continue = [
            call
            for call in model.calls
            if call[0] == "continue" and call[2][0]["state_id"] == "root"
        ][0]
        self.assertEqual(root_continue[3], list(ANSWER_STOPS))
        self.assertFalse(result["trace"]["answer_protocol"]["fallback_used"])
        self.assertNotIn("text", result["trace"]["answer_completion"])

    def test_leaked_primary_uses_answer_only_fallback_without_new_search(self) -> None:
        class LeakingPrimaryModel(FakeStateModel):
            def state_batch_continue(self, **kwargs):
                items = kwargs["items"]
                if items[0]["state_id"] == "root":
                    self.calls.append(
                        (
                            "continue",
                            kwargs["home_url"],
                            list(items),
                            list(kwargs["stops"]),
                        )
                    )
                    return [
                        {
                            "state_id": "root",
                            "branch": "root",
                            "text": (
                                '<tool_result>{"status":"ok",'
                                '"evidence":[]}</tool_result>'
                            ),
                            "stop_reason": "max_tokens",
                            "seen_tokens": 50,
                        }
                    ]
                return super().state_batch_continue(**kwargs)

        model = LeakingPrimaryModel()
        tool_queries = []

        def execute_tool(name: str, arguments: dict, *, session_id: str):
            tool_queries.append(arguments["query"])
            return {
                "status": "ok",
                "evidence": [
                    {
                        "id": "W1",
                        "title": "Source",
                        "content": "The fallback answer is supported.",
                        "uri": "https://example.invalid/source",
                    }
                ],
            }

        result = StateNativeSearchAgent(
            state_model=model,
            parse_tool_call=parse_tool_call,
            execute_tool=execute_tool,
        ).run(
            "Who maintains the project?",
            session_id="session-leak",
            branch_width=1,
            max_rounds=1,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["answer"], "The fallback answer is supported [W1].")
        self.assertEqual(len(tool_queries), 1)
        self.assertEqual(len(model.fallback_calls), 1)
        self.assertEqual(model.fallback_calls[0][1], ANSWER_MAX_TOKENS)
        self.assertEqual(model.fallback_calls[0][2], list(ANSWER_STOPS))
        protocol = result["trace"]["answer_protocol"]
        self.assertTrue(protocol["fallback_used"])
        self.assertIn("protocol_tag", protocol["primary"]["errors"])
        self.assertNotIn("text", result["trace"]["answer_completion"])
        self.assertNotIn("raw", protocol["fallback_completion"])
        self.assertNotIn("<tool_result>", json.dumps(result).lower())
        self.assertEqual(model.released[0][2], ["root", "child-1"])

    def test_double_answer_failure_returns_safe_error_and_releases(self) -> None:
        class BrokenAnswerModel(FakeStateModel):
            def state_batch_continue(self, **kwargs):
                items = kwargs["items"]
                if items[0]["state_id"] == "root":
                    return [
                        {
                            "state_id": "root",
                            "branch": "root",
                            "text": '{"status":"ok","evidence":[]}',
                            "stop_reason": "max_tokens",
                            "seen_tokens": 50,
                        }
                    ]
                return super().state_batch_continue(**kwargs)

            def complete(self, *args, **kwargs):
                self.fallback_calls.append((args, kwargs))
                return {
                    "raw": '<tool_call>{"name":"web_search"}</tool_call>',
                    "stop": "max_tokens",
                    "output_tokens": 6,
                }

        model = BrokenAnswerModel()

        def execute_tool(name: str, arguments: dict, *, session_id: str):
            return {
                "status": "ok",
                "evidence": [
                    {
                        "id": "W1",
                        "title": "Source",
                        "content": "这个项目由团队维护。 Supported fact.",
                        "uri": "https://example.invalid/source",
                    }
                ],
            }

        result = StateNativeSearchAgent(
            state_model=model,
            parse_tool_call=parse_tool_call,
            execute_tool=execute_tool,
        ).run(
            "谁维护这个项目？",
            session_id="session-double-failure",
            branch_width=1,
            max_rounds=1,
        )
        self.assertEqual(result["status"], "answer_error")
        self.assertEqual(
            result["answer"],
            "检索已完成，但回答生成未通过输出协议校验。请重试。",
        )
        self.assertNotIn("<tool_result>", result["answer"].lower())
        self.assertNotIn("<tool_call>", result["answer"].lower())
        self.assertNotIn(
            "raw",
            result["trace"]["answer_protocol"]["fallback_completion"],
        )
        self.assertEqual(model.released[0][2], ["root", "child-1"])

    def test_tool_failure_still_releases_root_and_branches(self) -> None:
        model = FakeStateModel()

        def fail_tool(name: str, arguments: dict, *, session_id: str):
            raise RuntimeError("upstream failed")

        agent = StateNativeSearchAgent(
            state_model=model,
            parse_tool_call=parse_tool_call,
            execute_tool=fail_tool,
        )
        with self.assertRaisesRegex(RuntimeError, "upstream failed"):
            agent.run(
                "Who created the project?",
                session_id="session-a",
                branch_width=2,
                max_rounds=1,
            )
        self.assertEqual(
            model.released[0][2],
            ["root", "child-1", "child-2"],
        )


if __name__ == "__main__":
    unittest.main()
