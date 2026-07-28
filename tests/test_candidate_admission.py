from __future__ import annotations

import unittest

from rwkv_search.realtime.candidate_ranker import (
    admit_candidates,
    candidate_rejection_reasons,
)
from rwkv_search.realtime.types import DiscoveredURL


def candidate(url: str, title: str, rank: int) -> DiscoveredURL:
    return DiscoveredURL(
        url=url,
        title=title,
        engine="fixture",
        rank=rank,
        rrf_score=1.0 / (60 + rank),
    )


class CandidateAdmissionTests(unittest.TestCase):
    def test_target_page_outranks_official_homepage_and_tutorial(self) -> None:
        values = [
            candidate("https://python.org/", "Welcome to Python", 1),
            candidate("https://example.net/python-tutorial", "Python tutorial", 2),
            candidate(
                "https://python.org/downloads/release/python-314/",
                "Python 3.14 official release",
                3,
            ),
        ]
        result = admit_candidates(
            "Python latest official release",
            ["Python latest official release"],
            values,
            max_candidates=10,
        )
        self.assertIn("/downloads/release/", result.admitted[0].url)
        self.assertGreater(result.admitted[0].candidate_score, 0)
        self.assertIn("url_coverage", result.admitted[0].score_components)

    def test_explicit_source_preference_reaches_generic_reranker(self) -> None:
        values = [
            candidate(
                "https://industry.example/quasarkit-release",
                "QuasarKit stable release analysis",
                1,
            ),
            candidate(
                "https://quasarkit.example/releases/v3",
                "QuasarKit v3",
                3,
            ),
        ]
        result = admit_candidates(
            "QuasarKit stable release",
            ["QuasarKit stable release"],
            values,
            max_candidates=10,
            source_preference="official_required",
        )
        self.assertEqual(result.admitted[0].url, values[1].url)
        self.assertEqual(
            result.admitted[0].score_components["source_preference_alignment"],
            1.0,
        )

    def test_dictionary_is_a_structural_rejection_not_an_intent_branch(self) -> None:
        value = candidate(
            "https://dictionary.example/latest",
            "latest dictionary definition",
            1,
        )
        self.assertIn(
            "dictionary",
            candidate_rejection_reasons("latest Python release", value),
        )
        self.assertIn(
            "dictionary",
            candidate_rejection_reasons("latest 的词典释义", value),
        )
        iciba = candidate(
            "https://www.iciba.com/word?w=latest",
            "latest是什么意思_latest的翻译_音标_读音_用法_例句",
            1,
        )
        self.assertIn("dictionary", candidate_rejection_reasons("Linux latest release", iciba))

    def test_article_discussing_a_login_problem_is_not_rejected_by_snippet(self) -> None:
        value = candidate("https://example.com/help", "Account troubleshooting", 1)
        value.snippet = "How to solve a captcha error during registration"
        self.assertNotIn(
            "login_or_captcha",
            candidate_rejection_reasons("account troubleshooting", value),
        )

    def test_explicit_site_is_a_hard_constraint(self) -> None:
        values = [
            candidate("https://example.com/release", "Project release", 1),
            candidate("https://project.org/release", "Project release", 2),
        ]
        result = admit_candidates(
            "site:project.org Project release",
            ["site:project.org Project release"],
            values,
            max_candidates=10,
        )
        self.assertEqual([item.url for item in result.admitted], [values[1].url])
        self.assertEqual(result.rejection_counts, {"outside_explicit_site": 1})

    def test_domain_diversity_only_moves_overflow_to_the_tail(self) -> None:
        values = [
            candidate(f"https://a.example/release-{index}", f"Project release {index}", index)
            for index in range(1, 5)
        ] + [candidate("https://b.example/release", "Project release", 5)]
        result = admit_candidates(
            "Project release",
            ["Project release"],
            values,
            max_candidates=5,
            per_domain_limit=2,
        )
        hosts = [item.url.split("/")[2] for item in result.admitted]
        self.assertIn("b.example", hosts[:3])
        self.assertEqual(len(result.admitted), 5)

    def test_rerank_preserves_admitted_search_top_ten_set(self) -> None:
        values = [
            candidate(
                f"https://site-{index}.example/item",
                "unrelated" if index == 10 else f"Project release {index}",
                index,
            )
            for index in range(1, 13)
        ]
        values[10].title = "Project official release exact target"
        result = admit_candidates(
            "Project official release exact target",
            ["Project official release exact target"],
            values,
            max_candidates=12,
            per_domain_limit=2,
        )
        expected = {item.url for item in values[:10]}
        actual = {item.url for item in result.admitted[:10]}
        self.assertEqual(actual, expected)
        self.assertNotIn(values[10].url, actual)

    def test_login_error_and_empty_metadata_are_rejected(self) -> None:
        values = [
            candidate("https://example.com/login", "Sign in to continue", 1),
            candidate("https://example.com/404", "Page not found", 2),
            candidate("https://example.com/empty", "", 3),
        ]
        result = admit_candidates("project release", ["project release"], values, max_candidates=5)
        self.assertEqual(result.admitted, [])
        self.assertEqual(result.rejection_counts["login_or_captcha"], 1)
        self.assertEqual(result.rejection_counts["error_page"], 1)
        self.assertEqual(result.rejection_counts["empty_metadata"], 1)


if __name__ == "__main__":
    unittest.main()
