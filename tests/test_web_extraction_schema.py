from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from bench.web_extraction_schema import (
    WebExtractionSchemaError,
    load_cases,
    validate_cases,
)


BENCH = Path(__file__).parents[1] / "bench/web_extraction_cases.jsonl"


class WebExtractionSchemaTest(unittest.TestCase):
    def test_dataset_is_balanced_and_covers_static_page_shapes(self) -> None:
        cases = load_cases(BENCH)
        self.assertEqual(len(cases), 30)
        self.assertEqual(
            Counter(case["language"] for case in cases), {"zh": 15, "en": 15}
        )
        self.assertTrue(
            {
                "article",
                "code",
                "documentation",
                "filing",
                "government",
                "js_app",
                "json",
                "paper",
                "release",
                "repository",
                "table",
            }.issubset({case["page_type"] for case in cases})
        )
        self.assertTrue(
            {"usable", "unsupported", "js_required"}.issubset(
                {case["expected_static_outcome"] for case in cases}
            )
        )

    def test_every_usable_case_has_reviewable_content_expectations(self) -> None:
        for case in load_cases(BENCH):
            if case["expected_static_outcome"] != "usable":
                continue
            self.assertTrue(case["content_contains_any"], case["id"])
            self.assertGreaterEqual(case["min_text_chars"], 80, case["id"])
            if case["content_kind"] == "html":
                self.assertTrue(case["require_title"], case["id"])
                self.assertTrue(case["title_contains_any"], case["id"])

    def test_json_case_can_disable_title_requirement(self) -> None:
        case = next(
            case for case in load_cases(BENCH) if case["content_kind"] == "json"
        )
        self.assertFalse(case["require_title"])
        self.assertEqual(case["title_contains_any"], [])

    def test_duplicate_urls_and_invalid_usable_cases_are_rejected(self) -> None:
        case = load_cases(BENCH)[0]
        with self.assertRaisesRegex(WebExtractionSchemaError, "duplicate url"):
            validate_cases([case, {**case, "id": "extract-zh-099"}])
        with self.assertRaisesRegex(
            WebExtractionSchemaError, "content_contains_any"
        ):
            validate_cases([{**case, "content_contains_any": []}])

    def test_table_and_code_require_markers(self) -> None:
        case = load_cases(BENCH)[0]
        with self.assertRaisesRegex(WebExtractionSchemaError, "table_text_any"):
            validate_cases(
                [{**case, "require_table": True, "table_text_any": []}]
            )
        with self.assertRaisesRegex(WebExtractionSchemaError, "code_text_any"):
            validate_cases([{**case, "require_code": True, "code_text_any": []}])


if __name__ == "__main__":
    unittest.main()
