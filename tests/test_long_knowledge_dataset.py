from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

from bench.long_knowledge_schema import load_cases


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "bench" / "external" / "miracl-v1"
COMPAT_ROOT = ROOT / "bench" / "long_knowledge_compat_v1"
BASELINE_ROOT = ROOT / "bench" / "baselines" / "long_knowledge" / "finewiki-zh-miracl-dev-v1"
COMPAT_BASELINE_ROOT = (
    ROOT / "bench" / "baselines" / "long_knowledge" / "finewiki-zh-compat-v1"
)
EN_BASELINE_ROOT = (
    ROOT / "bench" / "baselines" / "long_knowledge" / "finewiki-en-miracl-dev-v1"
)
EN_COMPAT_BASELINE_ROOT = (
    ROOT / "bench" / "baselines" / "long_knowledge" / "finewiki-en-compat-v1"
)
BILINGUAL_BASELINE_ROOT = (
    ROOT / "bench" / "baselines" / "long_knowledge" / "finewiki-bilingual-v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class LongKnowledgeDatasetTests(unittest.TestCase):
    def test_bilingual_baseline_is_hash_locked_and_public_safe(self) -> None:
        manifest = json.loads(
            (BILINGUAL_BASELINE_ROOT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["name"], "finewiki-bilingual-v1")
        input_paths = {
            "zh_miracl_summary_sha256": BASELINE_ROOT / "summary.json",
            "zh_compat_summary_sha256": COMPAT_BASELINE_ROOT / "summary.json",
            "en_miracl_summary_sha256": EN_BASELINE_ROOT / "summary.json",
            "en_compat_summary_sha256": EN_COMPAT_BASELINE_ROOT / "summary.json",
        }
        for key, path in input_paths.items():
            self.assertEqual(manifest["inputs"][key], sha256(path))
        for name, expected in manifest["files"].items():
            self.assertEqual(expected, sha256(BILINGUAL_BASELINE_ROOT / name))
        self.assertFalse(manifest["raw_traces_included"])
        self.assertFalse(manifest["production_changed"])

        comparison_text = (BILINGUAL_BASELINE_ROOT / "comparison.json").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("/home/", comparison_text)
        self.assertNotIn("127.0.0.1", comparison_text)
        comparison = json.loads(comparison_text)
        self.assertEqual(
            comparison["benchmarks"]["miracl_dev"]["en"]["cases_total"],
            799,
        )
        self.assertEqual(
            comparison["benchmarks"]["miracl_dev"]["zh"]["cases_total"],
            393,
        )
        self.assertIn(
            "hybrid lexical+dense retrieval for descriptive queries",
            comparison["next_milestone_candidates_not_started"],
        )

    def test_frozen_english_baselines_are_hash_locked_and_public_safe(self) -> None:
        cases_path = DATASET_ROOT / "miracl_long_knowledge_dev_v1.jsonl"
        compat_cases_path = COMPAT_ROOT / "cases.jsonl"
        expected = (
            (EN_BASELINE_ROOT, cases_path, 799, 0),
            (EN_COMPAT_BASELINE_ROOT, compat_cases_path, 24, 4),
        )
        for baseline, test_set_path, cases_total, expected_missing in expected:
            manifest = json.loads(
                (baseline / "manifest.json").read_text(encoding="utf-8")
            )
            summary_path = baseline / "summary.json"
            summary_text = summary_path.read_text(encoding="utf-8")
            self.assertEqual(manifest["summary_sha256"], sha256(summary_path))
            self.assertEqual(manifest["test_set"]["sha256"], sha256(test_set_path))
            self.assertFalse(manifest["raw_traces_included"])
            self.assertFalse(manifest["production_changed"])
            self.assertNotIn("/home/", summary_text)
            self.assertNotIn("127.0.0.1", summary_text)

            payload = json.loads(summary_text)
            self.assertEqual(payload["cases_total"], cases_total)
            self.assertEqual(payload["expected_missing_cases"], expected_missing)
            self.assertEqual(payload["language_filter"], "en")

    def test_frozen_chinese_compatibility_baseline_is_hash_locked(self) -> None:
        manifest = json.loads(
            (COMPAT_BASELINE_ROOT / "manifest.json").read_text(encoding="utf-8")
        )
        summary = COMPAT_BASELINE_ROOT / "summary.json"
        self.assertEqual(manifest["summary_sha256"], sha256(summary))
        self.assertEqual(
            manifest["test_set"]["sha256"],
            sha256(COMPAT_ROOT / "cases.jsonl"),
        )
        payload = json.loads(summary.read_text(encoding="utf-8"))
        self.assertEqual(payload["cases_total"], 24)
        self.assertEqual(payload["expected_missing_cases"], 3)

    def test_compatibility_set_is_balanced_and_hash_locked(self) -> None:
        manifest = json.loads((COMPAT_ROOT / "manifest.json").read_text(encoding="utf-8"))
        cases_path = COMPAT_ROOT / manifest["cases"]["path"]
        cases = load_cases(cases_path)
        self.assertEqual(manifest["license"], "CC0-1.0")
        self.assertEqual(manifest["cases"]["sha256"], sha256(cases_path))
        self.assertEqual(len(cases), 48)
        self.assertEqual(Counter(case.language for case in cases), {"zh": 24, "en": 24})
        self.assertEqual(sum(case.expectation == "missing" for case in cases), 7)
        self.assertEqual(len({case.id for case in cases}), len(cases))
        self.assertTrue(all(case.query_type != "unspecified" for case in cases))

    def test_frozen_chinese_baseline_is_public_and_hash_locked(self) -> None:
        manifest = json.loads((BASELINE_ROOT / "manifest.json").read_text(encoding="utf-8"))
        summary_path = BASELINE_ROOT / "summary.json"
        summary_text = summary_path.read_text(encoding="utf-8")
        self.assertEqual(manifest["summary_sha256"], sha256(summary_path))
        self.assertEqual(
            manifest["test_set"]["sha256"],
            sha256(DATASET_ROOT / "miracl_long_knowledge_dev_v1.jsonl"),
        )
        self.assertNotIn("/home/", summary_text)
        self.assertNotIn("127.0.0.1", summary_text)
        self.assertEqual(json.loads(summary_text)["overall"]["cases"], 393)

    def test_manifest_and_derived_dataset_are_frozen(self) -> None:
        manifest = json.loads((DATASET_ROOT / "manifest.json").read_text(encoding="utf-8"))
        dataset = DATASET_ROOT / manifest["output"]["path"]
        self.assertEqual(manifest["source_repository"], "miracl/miracl")
        self.assertEqual(manifest["source_license"], "Apache-2.0")
        self.assertEqual(manifest["split"], "dev")
        self.assertEqual(manifest["output"]["sha256"], sha256(dataset))
        for item in manifest["upstream_files"]:
            path = DATASET_ROOT / item["path"]
            self.assertEqual(item["bytes"], path.stat().st_size)
            self.assertEqual(item["sha256"], sha256(path))

    def test_expected_language_counts_and_unique_ids(self) -> None:
        cases = load_cases(DATASET_ROOT / "miracl_long_knowledge_dev_v1.jsonl")
        self.assertEqual(len(cases), 1192)
        self.assertEqual(Counter(case.language for case in cases), {"zh": 393, "en": 799})
        self.assertEqual(len({case.id for case in cases}), len(cases))
        self.assertTrue(all(case.source_dataset == "MIRACL" for case in cases))
        self.assertTrue(all(case.source_split == "dev" for case in cases))
        self.assertTrue(all(case.relevant_pages for case in cases))


if __name__ == "__main__":
    unittest.main()
