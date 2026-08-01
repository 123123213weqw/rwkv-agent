from __future__ import annotations

import unittest

from benchmarks.replay_evidence_merge_ab import evaluate_replay, replay_selectors


class EvidenceMergeReplayTests(unittest.TestCase):
    def test_candidate_is_non_destructive_bounded_append(self) -> None:
        snapshot = {
            "calls": [
                {
                    "query": f"facet {index}",
                    "evidence": [
                        {
                            "title": f"Facet {index} source",
                            "uri": f"https://facet-{index}.example/source",
                            "score": 1.0,
                        },
                        *[
                            {
                                "title": f"Overview {index}-{child}",
                                "uri": (
                                    "https://overview.example/"
                                    f"{index}-{child}"
                                ),
                                "score": 2.0,
                            }
                            for child in range(3)
                        ],
                    ],
                }
                for index in range(6)
            ]
        }

        replay = replay_selectors("compare six facets", snapshot, limit=8)
        control = set(replay["control_uris"])
        candidate = set(replay["candidate_uris"])

        self.assertTrue(control.issubset(candidate))
        self.assertLessEqual(len(candidate), 12)

    def test_gold_is_used_only_by_the_evaluator(self) -> None:
        replay = {
            "call_count": 2,
            "nonempty_query_views": 2,
            "call_evidence_uris": [
                "https://example.com/control",
                "https://example.com/gold",
            ],
            "control_uris": ["https://example.com/control"],
            "candidate_uris": [
                "https://example.com/control",
                "https://example.com/gold",
            ],
        }
        case = {
            "id": "case-1",
            "language": "en",
            "gold": {"source_uris": ["https://example.com/gold"]},
        }

        row = evaluate_replay(case, replay)

        self.assertEqual(row["comparison"], "win")
        self.assertEqual(row["control_exact_page_recall"], 0.0)
        self.assertEqual(row["candidate_exact_page_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
