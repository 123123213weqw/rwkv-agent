from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "bench/baselines/realtime_retrieval/precision-discovery-v1"


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


if __name__ == "__main__":
    unittest.main()
