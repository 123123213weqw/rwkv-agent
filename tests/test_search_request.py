import unittest
import re

from rwkv_search.g1i_types import G1ICompletion
from rwkv_search.p4_search import (
    P4SearchPlanner,
    P4_SYSTEM_PROMPT,
    evaluate_query_constraints,
    sanitize_introduced_absolute_terms,
)
from rwkv_search.search_request import SearchRequestBuilder
from rwkv_search.pipeline.query_compiler import QueryHints


class SearchRequestBuilderTest(unittest.TestCase):
    def setUp(self):
        self.builder = SearchRequestBuilder()

    def test_explicit_hints_are_metadata_not_keyword_injection(self):
        request = self.builder.build(
            "Python 当前最新稳定版本是什么？请以官网为准。",
            "Python latest stable version official",
            hints=QueryHints(freshness="realtime", source_preference="official"),
        )
        self.assertEqual(request.freshness, "realtime")
        self.assertEqual(request.source_policy, "official_preferred")
        self.assertEqual(request.source_preference, "official")
        self.assertIn("official", request.execution_queries[0])
        self.assertEqual(len(request.execution_queries), 1)
        self.assertNotIn("请以官网为准", request.execution_queries[0])
        self.assertEqual(request.depth, "single")

    def test_raw_semantics_are_not_reinserted_after_model_compilation(self):
        request = self.builder.build(
            "RWKV 最近一个月有什么新进展？",
            "RWKV progress last month",
            hints=QueryHints(time_terms=("最近一个月",)),
        )
        self.assertIn("RWKV", request.execution_queries[0])
        self.assertNotIn("最近一个月", request.execution_queries[0])
        self.assertEqual(request.time_terms, ("最近一个月",))

    def test_source_policy_comes_from_explicit_hint(self):
        request = self.builder.build(
            "查找这条新闻的首发来源",
            "news headline original source",
            hints=QueryHints(source_preference="original"),
        )
        self.assertEqual(request.source_policy, "original_source")
        self.assertEqual(request.source_preference, "original")
        self.assertIn("original source", request.execution_queries[0])

    def test_explicit_site_is_preserved_without_raw_query_replay(self):
        request = self.builder.build(
            "请在 site:python.org 找最新稳定版", "Python latest stable release"
        )
        self.assertEqual(
            request.execution_queries,
            ("Python latest stable release site:python.org",),
        )

    def test_long_lookup_stays_single_pass(self):
        request = self.builder.build(
            "Find the latest significant earthquake information from the USGS earthquake site.",
            "USGS earthquake latest significant earthquake",
        )
        self.assertEqual(request.depth, "single")
        self.assertEqual(len(request.execution_queries), 1)
        self.assertFalse(any(item["raw_query_executed"] for item in request.trace if item["stage"] == "query_selection"))

    def test_frozen_p4_queries_have_no_legacy_pollution(self):
        forbidden = re.compile(
            r"\b(site\s+site|announcement\s+announcement|official\s+official)\b",
            re.I,
        )
        rows = [
            {
                "id": "python",
                "user_query": "Python当前最新稳定版本是什么？请以官网为准。",
                "model_query": "Python latest stable version",
            },
            {
                "id": "rwkv",
                "user_query": "RWKV最近有什么官方项目进展？",
                "model_query": "RWKV latest project progress",
            },
            {
                "id": "site",
                "user_query": "请在site:python.org找最新稳定版",
                "model_query": "Python latest stable release",
            },
        ]
        for row in rows:
            request = self.builder.build(row["user_query"], row["model_query"])
            self.assertEqual(len(request.execution_queries), 1, row["id"])
            self.assertEqual(request.depth, "single", row["id"])
            self.assertFalse(forbidden.search(request.execution_queries[0]), row["id"])
            self.assertNotEqual(request.execution_queries[0], request.raw_query, row["id"])

    def test_p4_planner_builds_request_from_strict_call(self):
        completion = G1ICompletion(
            '<tool_call>\n{"name":"web_search","arguments":{"query":"Python latest version"}}',
            "</tool_call>",
            (1, 2, 3),
            12.0,
        )
        planner = P4SearchPlanner(lambda prompt, stops, max_tokens: completion)
        plan = planner.plan("Python 当前最新版本是什么？请以官网为准。")
        self.assertIsNotNone(plan.search_request)
        self.assertEqual(
            plan.search_request.execution_queries,
            ("Python latest version", "Python 当前最新版本是什么？请以官网为准"),
        )
        self.assertIn("continuing the prefilled JSON", P4_SYSTEM_PROMPT)
        self.assertFalse(plan.fallback_to_raw)

    def test_p4_planner_accepts_prefilled_json_suffix(self):
        planner = P4SearchPlanner(
            lambda prompt, stops, max_tokens: G1ICompletion(
                'Python latest stable release"}}',
                "</tool_call>",
                (1, 2, 3),
            )
        )
        plan = planner.plan("Python最新稳定版本")
        self.assertTrue(plan.format_evaluation["strict_success"])
        self.assertFalse(plan.fallback_to_raw)
        self.assertEqual(
            plan.search_request.execution_queries,
            ("Python latest stable release", "Python最新稳定版本"),
        )

    def test_raw_recall_lane_is_opt_in_and_preserves_site_constraints(self):
        request = self.builder.build(
            "请在 site:python.org 找最新稳定版",
            "Python latest stable release",
            preserve_raw_query=True,
        )
        self.assertEqual(
            request.execution_queries,
            (
                "Python latest stable release site:python.org",
                "请在 site:python.org 找最新稳定版",
            ),
        )
        self.assertEqual(request.trace[-1]["stage"], "raw_query_recall_lane")
        self.assertEqual(request.trace[-1]["status"], "added")

    def test_raw_recall_lane_skips_unbounded_raw_query(self):
        request = self.builder.build(
            "很长的问题" * 100,
            "short search query",
            preserve_raw_query=True,
            raw_query_max_characters=64,
        )
        self.assertEqual(request.execution_queries, ("short search query",))
        self.assertEqual(request.trace[-1]["status"], "skipped_too_long")

    def test_p4_planner_deletes_invented_year_without_losing_rewrite(self):
        planner = P4SearchPlanner(
            lambda prompt, stops, max_tokens: G1ICompletion(
                '国家卫健委 疫情概况 2025"}}',
                "</tool_call>",
            )
        )
        plan = planner.plan("国家卫健委最新疫情概况")
        self.assertTrue(plan.format_evaluation["strict_success"])
        self.assertTrue(plan.constraint_evaluation["valid"])
        self.assertTrue(plan.constraint_evaluation["repair_applied"])
        self.assertEqual(
            plan.constraint_evaluation["removed_absolute_terms"], ["2025"]
        )
        self.assertEqual(
            plan.constraint_evaluation["original_evaluation"]["introduced_years"],
            ["2025"],
        )
        self.assertFalse(plan.fallback_to_raw)
        self.assertEqual(
            plan.search_request.execution_queries,
            ("国家卫健委 疫情概况", "国家卫健委最新疫情概况"),
        )

    def test_absolute_term_sanitizer_preserves_translation_and_user_versions(self):
        repaired, removed = sanitize_introduced_absolute_terms(
            "微软最近一个季度的业绩公告",
            "Microsoft FY2025 Q3 earnings release 1.24.0",
        )
        self.assertEqual(repaired, "Microsoft earnings release")
        self.assertEqual(removed, ("FY2025", "1.24.0", "Q3"))

        preserved, removed = sanitize_introduced_absolute_terms(
            "Python3.14 release notes",
            "Python 3.14 release notes official",
        )
        self.assertEqual(preserved, "Python 3.14 release notes official")
        self.assertEqual(removed, ())

    def test_p4_planner_still_falls_back_for_non_temporal_violation(self):
        planner = P4SearchPlanner(
            lambda prompt, stops, max_tokens: G1ICompletion(
                'Python release site:python.org site:example.com"}}',
                "</tool_call>",
            )
        )
        plan = planner.plan("Python最新版本")
        self.assertTrue(plan.fallback_to_raw)
        self.assertEqual(plan.fallback_reason, "query_constraint_violation")
        self.assertIn("excessive_site_operators", plan.constraint_evaluation["reasons"])

    def test_explicit_user_year_is_not_rejected(self):
        result = evaluate_query_constraints(
            "2024年中国CPI是多少",
            "中国 CPI 2024 国家统计局",
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["introduced_years"], [])

    def test_p4_planner_does_not_repair_non_tool_answer(self):
        planner = P4SearchPlanner(
            lambda prompt, stops, max_tokens: G1ICompletion("Please provide the headline.", "</s>")
        )
        plan = planner.plan("核实这条新闻")
        self.assertFalse(plan.format_evaluation["strict_success"])
        self.assertTrue(plan.fallback_to_raw)
        self.assertEqual(plan.fallback_reason, "format_invalid")
        self.assertEqual(plan.search_request.execution_queries, ("核实这条新闻",))


if __name__ == "__main__":
    unittest.main()
