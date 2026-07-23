import unittest
from collections import Counter
import hashlib
import json
from pathlib import Path

from bench.build_retrieval_dev_v2 import build_rows
from bench.retrieval_schema import RetrievalCaseError, load_cases, validate_cases


BENCH = Path(__file__).parents[1] / "bench/realtime_web_retrieval.jsonl"
DEV_V2 = Path(__file__).parents[1] / "bench/realtime_web_retrieval_dev_v2.jsonl"
DEV_V2_MANIFEST = (
    Path(__file__).parents[1]
    / "bench/realtime_web_retrieval_dev_v2_manifest.json"
)
FROZEN_V1_SHA256 = "6900404d43deac290b599f10ee3b1f6e2fb8d8db06f821b346809049ab2e57dc"


class RealtimeRetrievalBenchDataTest(unittest.TestCase):
    def test_dataset_is_balanced_and_covers_required_source_types(self):
        rows = load_cases(BENCH)
        self.assertEqual(len(rows), 50)
        self.assertEqual(Counter(row["language"] for row in rows), {"zh": 25, "en": 25})
        categories = {row["category"] for row in rows}
        self.assertTrue(
            {
                "academic_paper",
                "company_filing",
                "government_policy",
                "newsroom",
                "official_docs",
                "realtime_public_info",
                "repository_release",
                "software_release",
                "statistics",
            }.issubset(categories)
        )

    def test_every_case_has_reviewable_source_expectations(self):
        rows = load_cases(BENCH)
        for row in rows:
            self.assertTrue(row["expected_domains_any"], row["id"])
            self.assertTrue(row["target_url_patterns_any"], row["id"])
            self.assertIn("search_homepage", row["forbidden_result_types"])
            self.assertIn("error_page", row["forbidden_result_types"])

    def test_rejects_duplicate_ids(self):
        row = load_cases(BENCH)[0]
        with self.assertRaisesRegex(RetrievalCaseError, "duplicate id"):
            validate_cases([row, {**row, "query": row["query"] + " duplicate"}])

    def test_rejects_domain_with_scheme_or_path(self):
        row = load_cases(BENCH)[0]
        with self.assertRaisesRegex(RetrievalCaseError, "invalid normalized domain"):
            validate_cases([{**row, "expected_domains_any": ["https://python.org/downloads/"]}])

    def test_rejects_language_id_mismatch(self):
        row = load_cases(BENCH)[0]
        with self.assertRaisesRegex(RetrievalCaseError, "language does not match id"):
            validate_cases([{**row, "language": "en"}])

    def test_frozen_v1_was_not_modified_by_the_dev_v2_expansion(self):
        self.assertEqual(hashlib.sha256(BENCH.read_bytes()).hexdigest(), FROZEN_V1_SHA256)

    def test_dev_v2_has_100_balanced_bilingual_cases(self):
        rows = load_cases(DEV_V2)
        self.assertEqual(len(rows), 100)
        self.assertEqual(Counter(row["language"] for row in rows), {"zh": 50, "en": 50})
        self.assertEqual(len({row["id"] for row in rows}), 100)
        self.assertTrue(all(101 <= int(row["id"].rsplit("-", 1)[1]) <= 150 for row in rows))

        frozen_queries = {
            " ".join(str(row["query"]).casefold().split()) for row in load_cases(BENCH)
        }
        dev_queries = {" ".join(str(row["query"]).casefold().split()) for row in rows}
        self.assertFalse(frozen_queries.intersection(dev_queries))

    def test_dev_v2_covers_realistic_styles_and_new_source_shapes(self):
        rows = load_cases(DEV_V2)
        styles = Counter(row["query_style"] for row in rows)
        categories = Counter(row["category"] for row in rows)
        policies = Counter(row["source_policy"] for row in rows)
        self.assertGreaterEqual(styles["conversational"], 40)
        self.assertGreaterEqual(styles["terse"], 20)
        self.assertGreaterEqual(styles["noisy"], 4)
        self.assertGreaterEqual(categories["community_discussion"], 10)
        self.assertGreaterEqual(categories["security_advisory"], 16)
        self.assertGreaterEqual(categories["standards_specification"], 10)
        self.assertGreaterEqual(policies["community_required"], 10)
        self.assertIn(1, {row["gold_ttl_days"] for row in rows})
        self.assertIn(730, {row["gold_ttl_days"] for row in rows})
        for row in rows:
            self.assertEqual(row["annotation_status"], "source_policy_reviewed")
            self.assertEqual(row["origin"], "manually_curated_realistic")
            self.assertTrue(row["expected_domains_any"], row["id"])
            self.assertTrue(row["target_url_patterns_any"], row["id"])

    def test_dev_v2_is_reproducible_and_bilingual_labels_are_paired(self):
        rows = load_cases(DEV_V2)
        generated = build_rows()
        self.assertEqual(rows, generated)
        by_id = {row["id"]: row for row in rows}
        ignored = {"id", "query", "language"}
        for number in range(101, 151):
            zh = by_id[f"retrieval-zh-{number:03d}"]
            en = by_id[f"retrieval-en-{number:03d}"]
            self.assertEqual(
                {key: value for key, value in zh.items() if key not in ignored},
                {key: value for key, value in en.items() if key not in ignored},
            )

        raw_rows = [json.loads(line) for line in DEV_V2.read_text().splitlines()]
        self.assertEqual(raw_rows, generated)

        manifest = json.loads(DEV_V2_MANIFEST.read_text())
        self.assertEqual(manifest["case_count"], 100)
        self.assertEqual(manifest["paired_topic_count"], 50)
        self.assertEqual(manifest["language_counts"], {"zh": 50, "en": 50})
        self.assertEqual(manifest["sha256"], hashlib.sha256(DEV_V2.read_bytes()).hexdigest())
        self.assertFalse(manifest["is_user_log"])
        self.assertFalse(manifest["is_blind_test"])
        self.assertFalse(manifest["runtime_label_visibility"])

    def test_extended_metadata_is_all_or_nothing_and_validated(self):
        row = load_cases(DEV_V2)[0]
        with self.assertRaisesRegex(RetrievalCaseError, "extended metadata"):
            validate_cases([{key: value for key, value in row.items() if key != "origin"}])
        with self.assertRaisesRegex(RetrievalCaseError, "invalid gold_ttl_days"):
            validate_cases([{**row, "gold_ttl_days": 0}])
        with self.assertRaisesRegex(RetrievalCaseError, "invalid query_style"):
            validate_cases([{**row, "query_style": "synthetic"}])


if __name__ == "__main__":
    unittest.main()
