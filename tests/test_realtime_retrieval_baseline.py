from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "bench/baselines/realtime_retrieval/precision-discovery-v1"
HYBRID_AUDIT = (
    ROOT / "bench/baselines/realtime_retrieval/hybrid-live-audit-v1"
)
ENGINE_STABILITY = (
    ROOT / "bench/baselines/realtime_retrieval/searxng-engine-stability-v1"
)
CANDIDATE_ENGINES = (
    ROOT / "bench/baselines/realtime_retrieval/searxng-candidate-engines-v1"
)
V100_DIRECT = (
    ROOT / "bench/baselines/realtime_retrieval/searxng-v100-direct-v1"
)
RERANK_BASELINE = (
    ROOT / "bench/baselines/realtime_retrieval/candidate-rerank-bge-m3-v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RealtimeRetrievalBaselineTest(unittest.TestCase):
    def test_public_manifest_and_small_artifacts_are_frozen(self) -> None:
        manifest = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "public_summary")
        self.assertEqual(manifest["benchmark"]["case_count"], 50)
        self.assertEqual(
            manifest["benchmark"]["sha256"],
            sha256(ROOT / manifest["benchmark"]["path"]),
        )
        for name, expected in manifest["frozen_files"].items():
            self.assertEqual(sha256(BASELINE / name), expected, name)
        self.assertFalse(any(BASELINE.glob("*.jsonl")))

    def test_precision_discovery_metrics_are_frozen(self) -> None:
        manifest = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
        comparison = json.loads((BASELINE / "comparison.json").read_text(encoding="utf-8"))
        baseline = json.loads((BASELINE / "baseline_summary.json").read_text(encoding="utf-8"))
        precision = json.loads((BASELINE / "precision_summary.json").read_text(encoding="utf-8"))
        metrics = manifest["headline_metrics"]
        self.assertEqual(metrics["candidate_domain_recall_at_10"], 0.48)
        self.assertEqual(metrics["candidate_target_page_recall_at_20"], 0.12)
        self.assertEqual(metrics["result_domain_recall_at_10"], 0.32)
        self.assertEqual(metrics["nonempty_result_rate"], 0.68)
        self.assertEqual(metrics["garbage_result_rate"], 0.0097)
        self.assertEqual(metrics["fetch_success_rate"], 0.5395)
        self.assertEqual(comparison["resource_usage"]["maximum_pivot_rounds"], 1)
        self.assertEqual(comparison["resource_usage"]["maximum_pivot_domains"], 2)
        self.assertEqual(comparison["resource_usage"]["maximum_one_hop_links"], 8)
        self.assertEqual(comparison["resource_usage"]["average_discovery_requests"], 1.64)
        self.assertEqual(baseline["case_ids"], precision["case_ids"])
        self.assertEqual(baseline["bench_sha256"], precision["bench_sha256"])

    def test_hybrid_live_audit_artifacts_are_frozen_and_public_safe(self) -> None:
        manifest = json.loads(
            (HYBRID_AUDIT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "public_summary")
        self.assertEqual(manifest["run_count"], 2)
        self.assertEqual(
            manifest["benchmark"]["sha256"],
            sha256(ROOT / manifest["benchmark"]["path"]),
        )
        for name, expected in manifest["frozen_files"].items():
            self.assertEqual(sha256(HYBRID_AUDIT / name), expected, name)
        self.assertFalse(any(HYBRID_AUDIT.glob("*.jsonl")))
        visible = "\n".join(
            path.read_text(encoding="utf-8")
            for path in HYBRID_AUDIT.iterdir()
            if path.is_file()
        ).casefold()
        self.assertNotIn("http://", visible)
        self.assertNotIn("https://", visible)
        self.assertNotIn("/home/", visible)
        self.assertNotIn("/users/wangyue/", visible)

    def test_hybrid_live_audit_freezes_failure_priority(self) -> None:
        comparison = json.loads(
            (HYBRID_AUDIT / "comparison.json").read_text(encoding="utf-8")
        )
        runs = comparison["hybrid_live_runs"]
        self.assertEqual(
            runs["range"]["candidate_domain_recall_at_10"],
            {"min": 0.44, "max": 0.48},
        )
        self.assertEqual(
            runs["range"]["candidate_target_page_recall_at_20"],
            {"min": 0.12, "max": 0.2},
        )
        self.assertEqual(
            runs["range"]["result_target_page_recall_at_20"],
            {"min": 0.02, "max": 0.04},
        )
        buckets = comparison["failure_attribution"]["common_largest_buckets"]
        self.assertEqual(buckets[0]["bucket"], "initial_domain_miss")
        self.assertEqual(buckets[0]["run1"], 28)
        self.assertEqual(buckets[0]["run2"], 26)
        self.assertTrue(comparison["decision"]["hybrid_is_not_primary_live_failure"])

    def test_engine_stability_baseline_is_frozen_and_public_safe(self) -> None:
        manifest = json.loads(
            (ENGINE_STABILITY / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["name"], "searxng-engine-stability-v1")
        for value in manifest["files"]:
            path = ENGINE_STABILITY / value["path"]
            self.assertEqual(value["sha256"], sha256(path), value["path"])
            self.assertEqual(value["bytes"], path.stat().st_size, value["path"])
        self.assertFalse(any(ENGINE_STABILITY.glob("*.jsonl")))
        visible = "\n".join(
            path.read_text(encoding="utf-8")
            for path in ENGINE_STABILITY.iterdir()
            if path.is_file()
        ).casefold()
        self.assertNotIn("http://", visible)
        self.assertNotIn("https://", visible)
        self.assertNotIn("/home/", visible)
        self.assertNotIn("/users/wangyue/", visible)

    def test_engine_stability_selection_is_not_overstated(self) -> None:
        comparison = json.loads(
            (ENGINE_STABILITY / "comparison.json").read_text(encoding="utf-8")
        )
        mwmbl = comparison["general"]["mwmbl"]
        search360 = comparison["general"]["360search"]
        self.assertEqual(
            mwmbl["languages"]["en"]["stable_nonempty_rate"], 0.92
        )
        self.assertEqual(
            mwmbl["languages"]["zh"]["stable_domain_recall_at_10"], 0.16
        )
        self.assertEqual(search360["overall"]["stable_nonempty_rate"], 0.0)
        self.assertEqual(comparison["selection"]["general_zh"]["selected"], [])
        self.assertTrue(comparison["selection"]["general_en"]["provisional"])
        self.assertFalse(comparison["production_changed"])

    def test_candidate_engine_trial_is_frozen_and_public_safe(self) -> None:
        manifest = json.loads(
            (CANDIDATE_ENGINES / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["name"], "searxng-candidate-engines-v1")
        for value in manifest["files"]:
            path = CANDIDATE_ENGINES / value["path"]
            self.assertEqual(value["sha256"], sha256(path), value["path"])
            self.assertEqual(value["bytes"], path.stat().st_size, value["path"])
        self.assertFalse(any(CANDIDATE_ENGINES.glob("*.jsonl")))
        visible = "\n".join(
            path.read_text(encoding="utf-8")
            for path in CANDIDATE_ENGINES.iterdir()
            if path.is_file()
        ).casefold()
        self.assertNotIn("http://", visible)
        self.assertNotIn("https://", visible)
        self.assertNotIn("/home/", visible)
        self.assertNotIn("/users/wangyue/", visible)

    def test_bing_trial_wins_without_approving_production(self) -> None:
        comparison = json.loads(
            (CANDIDATE_ENGINES / "comparison.json").read_text(encoding="utf-8")
        )
        metrics = comparison["qualified_full_benchmark"]["metrics"]
        self.assertEqual(metrics["overall"]["stable_domain_recall_at_10"], 0.56)
        self.assertEqual(metrics["languages"]["zh"]["stable_domain_recall_at_10"], 0.68)
        self.assertEqual(metrics["languages"]["en"]["stable_domain_recall_at_10"], 0.44)
        self.assertEqual(metrics["overall"]["garbage_result_rate"], 0.1622)
        self.assertEqual(comparison["decision"]["winner"], "bing")
        self.assertIsNone(comparison["decision"]["proxy_trial_winner"])
        self.assertEqual(
            comparison["v100_proxy_trial"]["duckduckgo_full_stability"][
                "overall"
            ]["request_success_rate"],
            0.0,
        )
        self.assertFalse(comparison["decision"]["ready_for_production"])
        self.assertFalse(comparison["existing_8888_instance_changed"])
        self.assertFalse(comparison["v100_proxy_changed"])
        self.assertTrue(comparison["temporary_tunnel_closed"])
        self.assertFalse(comparison["production_changed"])

    def test_v100_direct_trial_is_frozen_and_public_safe(self) -> None:
        manifest = json.loads(
            (V100_DIRECT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["name"], "searxng-v100-direct-v1")
        for value in manifest["files"]:
            path = V100_DIRECT / value["path"]
            self.assertEqual(value["sha256"], sha256(path), value["path"])
            self.assertEqual(value["bytes"], path.stat().st_size, value["path"])
        self.assertFalse(any(V100_DIRECT.glob("*.jsonl")))
        visible = "\n".join(
            path.read_text(encoding="utf-8")
            for path in V100_DIRECT.iterdir()
            if path.is_file()
        ).casefold()
        self.assertNotIn("http://", visible)
        self.assertNotIn("https://", visible)
        self.assertNotIn("/home/", visible)
        self.assertNotIn("/users/wangyue/", visible)

    def test_v100_direct_trial_rejects_all_smoke_candidates(self) -> None:
        comparison = json.loads(
            (V100_DIRECT / "comparison.json").read_text(encoding="utf-8")
        )
        self.assertEqual(comparison["decision"]["qualified_for_full_benchmark"], [])
        self.assertTrue(comparison["decision"]["full_benchmark_skipped"])
        self.assertTrue(comparison["decision"]["tunnel_was_not_the_root_cause"])
        self.assertEqual(comparison["engines"]["duckduckgo"]["nonempty_rate"], 0.1667)
        self.assertEqual(comparison["engines"]["google"]["nonempty_rate"], 0.0)
        self.assertEqual(comparison["engines"]["brave"]["request_success_rate"], 0.0)
        self.assertFalse(comparison["v100_proxy_changed"])
        self.assertFalse(comparison["production_changed"])

    def test_candidate_rerank_baseline_is_frozen_and_public_safe(self) -> None:
        manifest = json.loads(
            (RERANK_BASELINE / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["name"], "candidate-rerank-bge-m3-v1")
        for value in manifest["files"]:
            path = RERANK_BASELINE / value["path"]
            self.assertEqual(value["sha256"], sha256(path), value["path"])
            self.assertEqual(value["bytes"], path.stat().st_size, value["path"])
        self.assertFalse(any(RERANK_BASELINE.glob("*.jsonl")))
        visible = "\n".join(
            path.read_text(encoding="utf-8")
            for path in RERANK_BASELINE.iterdir()
            if path.is_file()
        ).casefold()
        self.assertNotIn("http://", visible)
        self.assertNotIn("https://", visible)
        self.assertNotIn("/home/", visible)
        self.assertNotIn("/users/wangyue/", visible)

    def test_hybrid_rerank_passes_offline_quality_gates(self) -> None:
        comparison = json.loads(
            (RERANK_BASELINE / "comparison.json").read_text(encoding="utf-8")
        )
        raw = comparison["strategies"]["raw"]
        hybrid = comparison["strategies"]["hybrid"]
        self.assertEqual(raw["top8_garbage_rate"], 0.1738)
        self.assertEqual(hybrid["top8_garbage_rate"], 0.0)
        self.assertEqual(hybrid["domain_recall_at_5"], 0.56)
        self.assertEqual(hybrid["domain_recall_at_10"], 0.56)
        self.assertEqual(hybrid["target_page_recall_at_20"], 0.08)
        self.assertGreater(hybrid["domain_mrr"], raw["domain_mrr"])
        self.assertGreater(hybrid["target_mrr"], raw["target_mrr"])
        self.assertEqual(hybrid["rejected_useful_expected_domain_count"], 0)
        self.assertEqual(hybrid["rejected_target_page_count"], 0)
        self.assertTrue(comparison["decision"]["quality_gates_passed"])
        self.assertFalse(comparison["decision"]["semantic_only_selected"])
        self.assertFalse(comparison["decision"]["production_changed"])


if __name__ == "__main__":
    unittest.main()
