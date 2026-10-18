from __future__ import annotations

import unittest
from pathlib import Path

from bench.run_c_feedback_integration_smoke import _integration_checks
from rwkv_search.config import AppConfig


class CFeedbackIntegrationSmokeTests(unittest.TestCase):
    def test_isolated_config_uses_bing_cn_without_precision_extensions(self) -> None:
        path = Path("configs/bench-4080-c-feedback.json")
        config = AppConfig.load(path).realtime_search
        self.assertEqual(config.searxng_url, "")
        self.assertEqual(config.fallback_engines, ["bing"])
        self.assertEqual(config.bing_base_url, "https://cn.bing.com")
        self.assertFalse(config.source_channels_enabled)
        self.assertFalse(config.domain_pivot_enabled)
        self.assertFalse(config.one_hop_link_expansion_enabled)

    def test_integration_checks_enforce_budget_and_safe_trace(self) -> None:
        record = {
            "id": "ok",
            "stats": {
                "discovery_request_count": 2,
                "feedback_query_executed": True,
                "model_search_plans": [{"stage": "initial", "query": "q1"}],
            },
            "candidates": [{"discovery_stages": ["initial", "model_feedback"]}],
            "events": [
                {"type": "discovery_progress"},
                {"type": "fetch_progress"},
                {"type": "realtime_result"},
            ],
        }
        checks = _integration_checks([record])
        self.assertTrue(checks["passed"])
        self.assertEqual(checks["max_discovery_requests"], 2)
        self.assertEqual(checks["feedback_candidate_provenance_cases"], 1)

        record["stats"]["discovery_request_count"] = 3
        record["stats"]["model_search_plans"] = [{"reasoning": "private"}]
        checks = _integration_checks([record])
        self.assertFalse(checks["passed"])
        self.assertEqual(len(checks["violations"]), 2)

        record["stats"] = {
            "discovery_request_count": 1,
            "feedback_query_executed": False,
            "model_search_plans": [{"stage": "initial"}],
        }
        record["candidates"] = [{"discovery_stages": ["initial"]}]
        checks = _integration_checks([record])
        self.assertFalse(checks["passed"])
        self.assertEqual(len(checks["violations"]), 2)


if __name__ == "__main__":
    unittest.main()
