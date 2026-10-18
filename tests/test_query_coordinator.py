from __future__ import annotations

import unittest

from rwkv_agent.query_coordinator import QueryCoordinator


class QueryCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinator = QueryCoordinator()

    def test_first_round_assigns_four_distinct_query_views(self) -> None:
        used: set[str] = set()
        views = [
            self.coordinator.coordinate(
                (
                    "rwkv的创始人是谁，他的github项目都是那些，"
                    "最新的更新是什么。"
                ),
                "RWKV 创始人 GitHub 项目 最新更新",
                branch_index=index,
                round_index=1,
                observation=None,
                used_queries=used,
            )
            for index in range(4)
        ]

        self.assertTrue(all(view.accepted for view in views))
        self.assertEqual(
            [view.strategy for view in views],
            ["model", "exact_anchors", "primary_source", "raw_question"],
        )
        self.assertEqual(len({view.query.casefold() for view in views}), 4)
        self.assertTrue(all(view.retained_anchors for view in views))

    def test_unrelated_github_title_cannot_become_evidence_pivot(self) -> None:
        used = {
            "rwkv 创始人 github 项目 最新更新",
            '"rwkv" "创始人" "github" 项目',
            "rwkv 创始人 github 项目 最新更新 官网 官方",
        }
        view = self.coordinator.coordinate(
            (
                "rwkv的创始人是谁，他的github项目都是那些，"
                "最新的更新是什么。"
            ),
            "RWKV GitHub latest update",
            branch_index=3,
            round_index=2,
            observation={
                "evidence": [
                    {
                        "title": "Leo501/awesome-CocosCreator",
                        "content": "A collection of Cocos Creator projects",
                        "source": "github",
                    }
                ]
            },
            used_queries=used,
        )

        self.assertTrue(view.accepted)
        self.assertNotEqual(view.strategy, "evidence_pivot")
        self.assertNotIn("CocosCreator", view.query)

    def test_second_round_uses_model_gap_without_copying_page_title(self) -> None:
        view = self.coordinator.coordinate(
            "RWKV最近有什么官方项目进展？",
            "RWKV newest release",
            branch_index=3,
            round_index=2,
            observation={
                "evidence": [
                    {
                        "title": "BlinkDL/RWKV-LM releases",
                        "content": "RWKV-LM official repository releases",
                        "source": "github",
                    }
                ]
            },
            used_queries=set(),
        )

        self.assertTrue(view.accepted)
        self.assertEqual(view.strategy, "gap_anchor_merge")
        self.assertTrue(view.evidence_based)
        self.assertTrue(view.gap_validated)
        self.assertEqual(view.safe_observation_count, 1)
        self.assertNotIn("RWKV-LM releases", view.query)

    def test_drifted_model_query_falls_back_to_original_anchors(self) -> None:
        view = self.coordinator.coordinate(
            "RWKV 的创始人和 GitHub 项目是什么？",
            "Cocos Creator game resources",
            branch_index=0,
            round_index=1,
            observation=None,
            used_queries=set(),
        )

        self.assertTrue(view.accepted)
        self.assertEqual(view.strategy, "exact_anchors")
        self.assertIn("RWKV", view.query)
        self.assertNotIn("Cocos", view.query)

    def test_location_question_does_not_pivot_to_unrelated_tourism_page(self) -> None:
        view = self.coordinator.coordinate(
            "上海浦东新区唐镇站如何到 RWKV 公司？",
            "唐镇站 RWKV 公司 地址 路线",
            branch_index=3,
            round_index=2,
            observation={
                "evidence": [
                    {
                        "title": "上海旅游景点大全",
                        "content": "浦东新区旅游和酒店推荐",
                        "source": "web",
                    }
                ]
            },
            used_queries=set(),
        )

        self.assertTrue(view.accepted)
        self.assertNotEqual(view.strategy, "evidence_pivot")
        self.assertNotIn("旅游", view.query)

    def test_duplicate_views_are_skipped_without_synthetic_suffix(self) -> None:
        question = "RWKV creator"
        used = {
            "rwkv creator",
            '"rwkv" "creator"',
            "rwkv creator official primary source",
        }
        view = self.coordinator.coordinate(
            question,
            question,
            branch_index=3,
            round_index=1,
            observation=None,
            used_queries=used,
        )

        self.assertFalse(view.accepted)
        self.assertEqual(view.strategy, "skipped_duplicate")
        self.assertEqual(view.query, "")
        self.assertIn("duplicate_query", view.rejection_reasons)

    def test_trace_exposes_validation_without_hidden_reasoning(self) -> None:
        view = self.coordinator.coordinate(
            "Find the latest stable Python release",
            "Python latest stable release",
            branch_index=0,
            round_index=1,
            observation=None,
            used_queries=set(),
        )
        trace = view.to_trace()

        self.assertTrue(trace["accepted"])
        self.assertEqual(trace["strategy"], "model")
        self.assertIn("Python", trace["anchors"])
        self.assertNotIn("reasoning", trace)

    def test_dictionary_observation_cannot_seed_second_round_query(self) -> None:
        view = self.coordinator.coordinate(
            "What is the average left-field distance in retractable-roof MLB parks?",
            "MLB retractable roof left field park dimensions",
            branch_index=0,
            round_index=2,
            observation={
                "evidence": [
                    {
                        "title": "average definition and pronunciation",
                        "content": "The meaning of average.",
                        "uri": "https://dictionary.cambridge.org/dictionary/english/average",
                    }
                ]
            },
            used_queries={"MLB retractable roof stadium average distance"},
        )

        self.assertTrue(view.accepted)
        self.assertEqual(view.safe_observation_count, 0)
        self.assertNotIn("definition", view.query)
        self.assertNotIn("Cambridge", view.query)


if __name__ == "__main__":
    unittest.main()
