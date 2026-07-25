from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class AgentHybridShadowBaselineTests(unittest.TestCase):
    def test_public_manifest_is_hash_locked_and_sanitized(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "bench"
            / "baselines"
            / "long_knowledge"
            / "agent-hybrid-shadow-v1"
        )
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(
            manifest["schema_version"],
            "agent-knowledge-shadow-manifest.v1",
        )
        for item in manifest["files"]:
            path = root / item["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                item["sha256"],
            )
            value = path.read_text(encoding="utf-8")
            self.assertNotIn("/home/wzu/", value)
            self.assertNotIn("127.0.0.1", value)

    def test_public_decision_does_not_claim_production_switch(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "bench"
            / "baselines"
            / "long_knowledge"
            / "agent-hybrid-shadow-v1"
            / "comparison.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(value["decision"]["shadow_integration_gate_passed"])
        self.assertFalse(value["decision"]["production_default_switch_passed"])
        self.assertFalse(value["scope"]["answer_model_called"])
        self.assertFalse(value["scope"]["production_enabled"])
        self.assertEqual(value["metrics"]["overall"]["hybrid"]["hit_at_5"], 21)
        self.assertEqual(value["metrics"]["overall"]["legacy"]["hit_at_5"], 18)


if __name__ == "__main__":
    unittest.main()
