import unittest
from collections import Counter
from pathlib import Path

from bench.retrieval_schema import RetrievalCaseError, load_cases, validate_cases


BENCH = Path(__file__).parents[1] / "bench/realtime_web_retrieval.jsonl"


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


if __name__ == "__main__":
    unittest.main()
