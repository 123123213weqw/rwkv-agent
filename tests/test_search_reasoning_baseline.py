from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = (
    ROOT / "bench/baselines/realtime_retrieval/search-reasoning-abcd-v1"
)
C_INTEGRATION = (
    ROOT / "bench/baselines/realtime_retrieval/c-feedback-integration-smoke-v1"
)
DEV_V2_C_LIVE = (
    ROOT / "bench/baselines/realtime_retrieval/retrieval-dev-v2-c-live-v1"
)
AC_DEV_V2 = (
    ROOT / "bench/baselines/realtime_retrieval/search-reasoning-ac-dev-v2-v1"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SearchReasoningBaselineTest(unittest.TestCase):
    def test_manifest_and_public_artifacts_are_frozen(self) -> None:
        manifest = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["name"], "search-reasoning-abcd-v1")
        self.assertEqual(manifest["status"], "public_summary")
        self.assertEqual(manifest["benchmark"]["case_count"], 50)
        self.assertEqual(
            manifest["benchmark"]["sha256"],
            sha256(ROOT / manifest["benchmark"]["path"]),
        )
        for value in manifest["files"]:
            path = BASELINE / value["path"]
            self.assertEqual(value["bytes"], path.stat().st_size, value["path"])
            self.assertEqual(value["sha256"], sha256(path), value["path"])
        self.assertFalse(any(BASELINE.glob("*.jsonl")))
        visible = "\n".join(
            path.read_text(encoding="utf-8")
            for path in BASELINE.iterdir()
            if path.is_file()
        ).casefold()
        self.assertNotIn("http://", visible)
        self.assertNotIn("https://", visible)
        self.assertNotIn("/home/", visible)
        self.assertNotIn("/users/wangyue/", visible)

    def test_feedback_wins_and_react_is_not_approved(self) -> None:
        summary = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))
        comparison = json.loads(
            (BASELINE / "comparison.json").read_text(encoding="utf-8")
        )
        budget = json.loads(
            (BASELINE / "budget_sensitivity.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["overall"]["direct"]["domain_hit_at_10_rate"], 0.56)
        self.assertEqual(summary["overall"]["feedback"]["domain_hit_at_10_rate"], 0.62)
        self.assertEqual(
            summary["overall"]["feedback"]["target_page_hit_at_20_rate"], 0.14
        )
        self.assertEqual(summary["overall"]["react"]["domain_hit_at_10_rate"], 0.48)
        self.assertEqual(
            summary["overall"]["react"]["stop_reason_counts"].get("model_final", 0),
            0,
        )
        self.assertEqual(comparison["decision"]["best_quality_cost_tradeoff"], "feedback")
        self.assertTrue(comparison["decision"]["react_is_rejected_for_fast_path"])
        self.assertEqual(budget["aggregate"]["384_case_success_count"], 3)
        self.assertFalse(summary["production_changed"])

    def test_c_feedback_integration_smoke_is_frozen_and_public_safe(self) -> None:
        manifest = json.loads(
            (C_INTEGRATION / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["name"], "c-feedback-integration-smoke-v1")
        self.assertFalse(manifest["raw_traces_included"])
        self.assertFalse(manifest["production_changed"])
        for value in manifest["files"]:
            path = C_INTEGRATION / value["path"]
            self.assertEqual(value["sha256"], sha256(path), value["path"])
            self.assertEqual(value["bytes"], path.stat().st_size, value["path"])
        self.assertFalse(any(C_INTEGRATION.glob("*.jsonl")))
        visible = "\n".join(
            path.read_text(encoding="utf-8")
            for path in C_INTEGRATION.iterdir()
            if path.is_file()
        ).casefold()
        self.assertNotIn("http://", visible)
        self.assertNotIn("https://", visible)
        self.assertNotIn("/home/", visible)
        self.assertNotIn("/users/wangyue/", visible)

        summary = json.loads(
            (C_INTEGRATION / "summary.json").read_text(encoding="utf-8")
        )
        comparison = json.loads(
            (C_INTEGRATION / "comparison.json").read_text(encoding="utf-8")
        )
        checks = summary["integration_checks"]
        self.assertTrue(checks["passed"])
        self.assertEqual(checks["feedback_query_executed_cases"], 4)
        self.assertEqual(checks["feedback_candidate_provenance_cases"], 4)
        self.assertEqual(checks["max_discovery_requests"], 2)
        self.assertTrue(
            comparison["protocol_alignment"]["candidate_merge_before_fetch"]
        )

    def test_dev_v2_c_live_baseline_is_frozen_and_public_safe(self) -> None:
        manifest = json.loads(
            (DEV_V2_C_LIVE / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["name"], "retrieval-dev-v2-c-live-v1")
        self.assertEqual(manifest["benchmark"]["case_count"], 100)
        self.assertEqual(
            manifest["benchmark"]["sha256"],
            sha256(ROOT / manifest["benchmark"]["path"]),
        )
        self.assertFalse(manifest["raw_traces_included"])
        self.assertFalse(manifest["production_changed"])
        for value in manifest["files"]:
            path = DEV_V2_C_LIVE / value["path"]
            self.assertEqual(value["sha256"], sha256(path), value["path"])
            self.assertEqual(value["bytes"], path.stat().st_size, value["path"])
        self.assertFalse(any(DEV_V2_C_LIVE.glob("*.jsonl")))
        visible = "\n".join(
            path.read_text(encoding="utf-8")
            for path in DEV_V2_C_LIVE.iterdir()
            if path.is_file()
        ).casefold()
        self.assertNotIn("http://", visible)
        self.assertNotIn("https://", visible)
        self.assertNotIn("/home/", visible)
        self.assertNotIn("/users/wangyue/", visible)

        summary = json.loads(
            (DEV_V2_C_LIVE / "summary.json").read_text(encoding="utf-8")
        )
        comparison = json.loads(
            (DEV_V2_C_LIVE / "comparison.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["overall"]["candidate_domain_hit_at_10_rate"], 0.42)
        self.assertEqual(
            summary["overall"]["candidate_target_page_hit_at_20_rate"], 0.04
        )
        self.assertEqual(summary["planner"]["initial"]["strict_action_rate"], 1.0)
        self.assertEqual(summary["planner"]["feedback"]["strict_action_rate"], 0.9342)
        self.assertEqual(comparison["candidate_domain_hit_at_10"]["gained_cases"], 5)
        self.assertEqual(comparison["candidate_domain_hit_at_10"]["regressed_cases"], 0)
        self.assertEqual(comparison["max_discovery_requests"], 2)
        self.assertFalse(summary["benchmark"]["is_user_log"])
        self.assertFalse(summary["benchmark"]["is_blind_test"])

    def test_dev_v2_ac_paired_baseline_is_frozen_and_public_safe(self) -> None:
        manifest = json.loads(
            (AC_DEV_V2 / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["name"], "search-reasoning-ac-dev-v2-v1")
        self.assertEqual(manifest["benchmark"]["case_count"], 100)
        self.assertEqual(
            manifest["benchmark"]["sha256"],
            sha256(ROOT / manifest["benchmark"]["path"]),
        )
        self.assertFalse(manifest["raw_traces_included"])
        self.assertFalse(manifest["production_changed"])
        for value in manifest["files"]:
            path = AC_DEV_V2 / value["path"]
            self.assertEqual(value["sha256"], sha256(path), value["path"])
            self.assertEqual(value["bytes"], path.stat().st_size, value["path"])
        self.assertFalse(any(AC_DEV_V2.glob("*.jsonl")))
        visible = "\n".join(
            path.read_text(encoding="utf-8")
            for path in AC_DEV_V2.iterdir()
            if path.is_file()
        ).casefold()
        self.assertNotIn("http://", visible)
        self.assertNotIn("https://", visible)
        self.assertNotIn("/home/", visible)
        self.assertNotIn("/users/wangyue/", visible)

        summary = json.loads(
            (AC_DEV_V2 / "summary.json").read_text(encoding="utf-8")
        )
        comparison = json.loads(
            (AC_DEV_V2 / "comparison.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(summary["overall"]), {"direct", "feedback"})
        self.assertTrue(summary["protocol"]["shared_q1_candidates"])
        self.assertEqual(
            summary["overall"]["direct"]["domain_hit_at_10_rate"], 0.36
        )
        self.assertEqual(
            summary["overall"]["feedback"]["domain_hit_at_10_rate"], 0.39
        )
        self.assertEqual(
            summary["overall"]["feedback"]["target_page_hit_at_20_rate"],
            0.04,
        )
        self.assertEqual(comparison["shared_q1_candidate_cases"], 100)
        self.assertEqual(
            comparison["metrics"]["domain_hit_at_10"]["gained_cases"], 3
        )
        self.assertEqual(
            comparison["metrics"]["domain_hit_at_10"]["regressed_cases"], 0
        )


if __name__ == "__main__":
    unittest.main()
