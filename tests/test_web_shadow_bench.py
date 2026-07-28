from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from benchmarks.run_web_shadow_bench import (
    classify_failure_stage,
    load_cases,
    run_case,
    summarize,
)


class WebShadowBenchTests(unittest.TestCase):
    def test_frozen_dataset_is_hash_locked(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "data"
            / "realtime_web_retrieval_v1.jsonl"
        )
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "6900404d43deac290b599f10ee3b1f6e2fb8d8db06f821b346809049ab2e57dc",
        )
        rows = load_cases(path)
        self.assertEqual(len(rows), 50)
        self.assertEqual({row["language"] for row in rows}, {"zh", "en"})

    def test_summary_preserves_paired_wins(self) -> None:
        def arm(*, domain: bool, result: bool, profile: str):
            return {
                "id": profile,
                "language": "en",
                "category": "test",
                "source_policy": "official_required",
                "profile": profile,
                "status": "ok",
                "evidence_nonempty": result,
                "failure_stage": "ok" if result else "final_ranking_empty",
                "discovery_engines": ["bing", "duckduckgo"],
                "metrics": {
                    "candidate_domain_hit_at_5": domain,
                    "candidate_domain_hit_at_10": domain,
                    "candidate_domain_hit_at_20": domain,
                    "candidate_organization_domain_hit_at_10": domain,
                    "candidate_organization_domain_hit_at_20": domain,
                    "result_domain_hit_at_5": result,
                    "result_domain_hit_at_10": result,
                    "result_domain_hit_at_20": result,
                    "result_organization_domain_hit_at_10": result,
                    "result_organization_domain_hit_at_20": result,
                    "candidate_target_page_hit_at_10": False,
                    "candidate_target_page_hit_at_20": False,
                    "result_target_page_hit_at_10": False,
                    "result_target_page_hit_at_20": False,
                    "candidate_count": 1,
                    "result_count": int(result),
                    "nonempty_result": result,
                    "garbage_result_count": 0,
                    "fetch_attempted": 1,
                    "fetch_succeeded": int(result),
                    "fetch_failed": int(not result),
                    "fetch_cancelled": 0,
                },
                "candidate_stage_metrics": {},
                "total_elapsed_ms": 10.0,
            }

        value = summarize(
            [
                {
                    "execution_order": ["legacy", "enhanced"],
                    "legacy": arm(
                        domain=False,
                        result=False,
                        profile="legacy",
                    ),
                    "enhanced": arm(
                        domain=True,
                        result=True,
                        profile="enhanced",
                    ),
                }
            ]
        )
        self.assertEqual(
            value["paired"]["candidate_domain_hit_at_10"]["enhanced_wins"],
            1,
        )
        self.assertEqual(
            value["paired"]["nonempty_result"]["enhanced_wins"],
            1,
        )
        self.assertEqual(
            value["by_execution_order"]["legacy_first"]["enhanced_nonempty"],
            1,
        )
        self.assertEqual(
            value["failure_stages"]["legacy"]["final_ranking_empty"],
            1,
        )
        self.assertEqual(value["evidence_nonempty_rate"]["enhanced"], 1.0)
        self.assertEqual(
            value["discovery_engines"]["enhanced"],
            ["bing", "duckduckgo"],
        )

    def test_run_case_records_effective_legacy_evidence_fallback(self) -> None:
        class Adapter:
            def __init__(self, profile: str, evidence: list[dict]) -> None:
                self.profile = profile
                self.evidence = evidence

            def execute_with_trace(self, query: str):
                del query
                return (
                    {"status": "ok", "evidence": self.evidence},
                    {
                        "status": "empty",
                        "candidates": [],
                        "initial_candidates": [],
                        "post_pivot_candidates": [],
                        "results": [],
                        "fetches": [],
                        "stats": {},
                        "latency_ms": 1.0,
                    },
                )

        path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "data"
            / "realtime_web_retrieval_v1.jsonl"
        )
        case = load_cases(path)[0]
        row = run_case(
            case,
            legacy=Adapter(
                "legacy",
                [{"uri": "https://legacy.invalid"}],
            ),
            enhanced=Adapter("enhanced", []),
            enhanced_first=False,
        )
        self.assertTrue(row["enhanced"]["fallback_used"])
        self.assertEqual(
            row["enhanced"]["effective_evidence"][0]["uri"],
            "https://legacy.invalid",
        )

    def test_failure_stage_separates_discovery_fetch_and_ranking(self) -> None:
        def trace(**stats):
            return {"status": "empty", "stats": stats}

        self.assertEqual(
            classify_failure_stage(trace(raw_candidates=0)),
            "discovery_empty",
        )
        self.assertEqual(
            classify_failure_stage(
                trace(raw_candidates=2, candidates=2, attempted=2, fetched=0)
            ),
            "fetch_failed",
        )
        self.assertEqual(
            classify_failure_stage(
                trace(
                    raw_candidates=2,
                    candidates=2,
                    attempted=2,
                    fetched=2,
                    usable=2,
                    selected=0,
                )
            ),
            "final_ranking_empty",
        )


if __name__ == "__main__":
    unittest.main()
