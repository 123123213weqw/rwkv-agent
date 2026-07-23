from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "bench/baselines/web_extraction/static-extractors-v1"
HYBRID_BASELINE = ROOT / "bench/baselines/web_extraction/hybrid-fast-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WebExtractionBaselineTest(unittest.TestCase):
    def test_manifest_hashes_and_public_artifact_boundary(self) -> None:
        manifest = json.loads((BASELINE / "manifest.json").read_text())
        self.assertFalse(manifest["raw_web_bodies_included"])
        for item in manifest["files"]:
            path = ROOT / item["path"]
            self.assertTrue(path.is_file(), path)
            self.assertGreater(item["bytes"], 0)
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            # Baseline-local artifacts must remain byte-for-byte frozen. Source
            # files are provenance records and are expected to evolve in later
            # candidates; comparing them with the old hash would prevent any
            # legitimate post-baseline implementation work.
            if path.is_relative_to(BASELINE):
                self.assertEqual(path.stat().st_size, item["bytes"], path)
                self.assertEqual(sha256(path), item["sha256"], path)
        dataset = next(
            item
            for item in manifest["files"]
            if item["path"] == "bench/web_extraction_cases.jsonl"
        )
        self.assertEqual(
            sha256(ROOT / dataset["path"]), dataset["sha256"]
        )
        self.assertFalse(any(BASELINE.glob("*.html")))
        self.assertFalse(any(BASELINE.glob("*.pdf")))
        self.assertFalse(any(BASELINE.glob("extract-*.md")))

    def test_headline_metrics_and_case_matrix_are_frozen(self) -> None:
        summary = json.loads((BASELINE / "summary.json").read_text())
        comparison = json.loads((BASELINE / "comparison.json").read_text())
        matrix = json.loads((BASELINE / "case_matrix.json").read_text())
        self.assertEqual(summary["case_count"], 30)
        self.assertEqual(summary["fetch_success_rate"], 0.9333)
        self.assertEqual(summary["repetitions"], 3)
        self.assertTrue(summary["all_outputs_deterministic"])
        self.assertEqual(summary["extractors"]["current"]["pass_rate"], 0.7857)
        self.assertEqual(summary["extractors"]["resiliparse"]["pass_rate"], 0.75)
        self.assertEqual(
            comparison["finding"]["accuracy_baseline"], "current"
        )
        self.assertEqual(comparison["finding"]["fast_candidate"], "resiliparse")
        self.assertEqual(len(matrix["cases"]), 30)
        self.assertTrue(
            all(len(case["extractors"]) == 5 for case in matrix["cases"])
        )

    def test_hybrid_baseline_is_frozen_and_excludes_web_bodies(self) -> None:
        manifest = json.loads((HYBRID_BASELINE / "manifest.json").read_text())
        self.assertFalse(manifest["raw_web_bodies_included"])
        for item in manifest["files"]:
            path = ROOT / item["path"]
            self.assertTrue(path.is_file(), path)
            self.assertGreater(item["bytes"], 0)
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            if path.is_relative_to(HYBRID_BASELINE):
                self.assertEqual(path.stat().st_size, item["bytes"], path)
                self.assertEqual(sha256(path), item["sha256"], path)
        self.assertFalse(any(HYBRID_BASELINE.glob("*.html")))
        self.assertFalse(any(HYBRID_BASELINE.glob("*.pdf")))

    def test_hybrid_headline_metrics_and_fallback_behavior_are_frozen(self) -> None:
        summary = json.loads((HYBRID_BASELINE / "summary.json").read_text())
        comparison = json.loads(
            (HYBRID_BASELINE / "comparison.json").read_text()
        )
        matrix = json.loads((HYBRID_BASELINE / "case_matrix.json").read_text())
        hybrid = summary["extractors"]["hybrid_fast"]
        self.assertEqual(summary["case_count"], 30)
        self.assertEqual(summary["repetitions"], 3)
        self.assertTrue(summary["all_outputs_deterministic"])
        self.assertEqual(hybrid["pass_rate"], 0.8571)
        self.assertEqual(hybrid["fallback_trigger_rate"], 0.1429)
        self.assertEqual(hybrid["fallback_use_rate"], 0.0357)
        self.assertEqual(hybrid["author_hit_rate"], 1.0)
        self.assertEqual(hybrid["published_at_hit_rate"], 1.0)
        self.assertEqual(
            comparison["candidate_minus_current"]["pass_rate"], 0.0714
        )
        self.assertFalse(comparison["finding"]["production_switched"])
        self.assertEqual(len(matrix["cases"]), 30)
        self.assertTrue(
            all(len(case["extractors"]) == 4 for case in matrix["cases"])
        )


if __name__ == "__main__":
    unittest.main()
