from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bench.retrieval_failure_attribution import (
    aggregate_attributions,
    attribute_record,
)


TARGET_URL = "https://example.com/releases/current"
HOME_URL = "https://example.com/"


def _record(**updates: object) -> dict:
    value = {
        "id": "case-1",
        "language": "en",
        "category": "release",
        "expected_domains_any": ["example.com"],
        "target_url_patterns_any": ["/releases/"],
        "initial_candidates": [],
        "post_pivot_candidates": [],
        "candidates": [],
        "fetches": [],
        "results": [],
        "stats": {
            "discovery_request_count": 1,
            "attempted": 0,
            "usable": 0,
        },
        "total_elapsed_ms": 100.0,
    }
    value.update(updates)
    return value


def _item(url: str) -> dict:
    return {"url": url, "title": "Example"}


def _fetch(
    *,
    status: str = "succeeded",
    rejected: bool = False,
) -> dict:
    value = {
        "requested_url": TARGET_URL,
        "final_url": TARGET_URL if status == "succeeded" else "",
        "status": status,
        "error_type": "ExtractionError" if status != "succeeded" else "",
    }
    if rejected:
        value["admission_rejection_reasons"] = ["login_or_captcha"]
    return value


class RetrievalFailureAttributionTest(unittest.TestCase):
    def test_target_lifecycle_outcomes_are_mutually_attributed(self) -> None:
        cases = {
            "initial_domain_miss": _record(),
            "exact_page_not_discovered_after_precision": _record(
                initial_candidates=[_item(HOME_URL)],
                post_pivot_candidates=[_item(HOME_URL)],
                candidates=[_item(HOME_URL)],
            ),
            "target_not_scheduled": _record(
                initial_candidates=[_item(HOME_URL)],
                post_pivot_candidates=[_item(TARGET_URL)],
                candidates=[_item(TARGET_URL)],
            ),
            "target_fetch_or_extraction_failed": _record(
                initial_candidates=[_item(TARGET_URL)],
                post_pivot_candidates=[_item(TARGET_URL)],
                candidates=[_item(TARGET_URL)],
                fetches=[_fetch(status="failed")],
            ),
            "target_post_fetch_rejected": _record(
                initial_candidates=[_item(TARGET_URL)],
                post_pivot_candidates=[_item(TARGET_URL)],
                candidates=[_item(TARGET_URL)],
                fetches=[_fetch(rejected=True)],
            ),
            "target_final_ranking_drop": _record(
                initial_candidates=[_item(TARGET_URL)],
                post_pivot_candidates=[_item(TARGET_URL)],
                candidates=[_item(TARGET_URL)],
                fetches=[_fetch()],
            ),
            "success_result": _record(
                initial_candidates=[_item(TARGET_URL)],
                post_pivot_candidates=[_item(TARGET_URL)],
                candidates=[_item(TARGET_URL)],
                fetches=[_fetch()],
                results=[_item(TARGET_URL)],
            ),
        }
        for expected, record in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(attribute_record(record)["target_outcome"], expected)

    def test_stage_attribution_separates_pivot_and_one_hop(self) -> None:
        pivot = attribute_record(
            _record(
                initial_candidates=[_item(HOME_URL)],
                post_pivot_candidates=[_item(TARGET_URL)],
                candidates=[_item(TARGET_URL)],
            )
        )
        one_hop = attribute_record(
            _record(
                initial_candidates=[_item(HOME_URL)],
                post_pivot_candidates=[_item(HOME_URL)],
                candidates=[_item(TARGET_URL)],
            )
        )
        self.assertEqual(pivot["target_stage"], "domain_pivot")
        self.assertEqual(one_hop["target_stage"], "one_hop_link")

    def test_aggregate_preserves_each_run_and_reports_variation(self) -> None:
        first = _record(
            initial_candidates=[_item(TARGET_URL)],
            post_pivot_candidates=[_item(TARGET_URL)],
            candidates=[_item(TARGET_URL)],
            fetches=[_fetch()],
            results=[_item(TARGET_URL)],
        )
        second = _record()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, record in enumerate((first, second), 1):
                path = Path(directory) / f"run-{index}.jsonl"
                path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                paths.append(path)
            summary = aggregate_attributions(paths)
        self.assertEqual(summary["run_count"], 2)
        self.assertEqual(
            summary["runs"][0]["target_outcomes"], {"success_result": 1}
        )
        target_range = summary["across_runs"]["stage_recall_ranges"][
            "result_target"
        ]
        self.assertEqual(target_range, {"mean": 0.5, "min": 0.0, "max": 1.0})

    def test_fetch_failure_types_distinguish_extraction_from_network(self) -> None:
        attributed = attribute_record(
            _record(
                initial_candidates=[_item(TARGET_URL)],
                post_pivot_candidates=[_item(TARGET_URL)],
                candidates=[_item(TARGET_URL)],
                fetches=[_fetch(status="failed")],
            )
        )
        self.assertEqual(attributed["target_failure_types"], ["ExtractionError"])


if __name__ == "__main__":
    unittest.main()
