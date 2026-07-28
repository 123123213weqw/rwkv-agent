from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "serve_fitgen_hf_eval.py"
SPEC = importlib.util.spec_from_file_location("fitgen_hf_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PromptStateRuntime = MODULE.PromptStateRuntime


class PromptStateRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        def generate(prompt, stops, _max_tokens):
            self.assertTrue(prompt)
            self.assertIn("</tool_call>", stops)
            return {
                "text": "answer",
                "committed_text": "answer</tool_call>",
                "token_ids": [1, 2],
                "stop_reason": "</tool_call>",
            }

        self.runtime = PromptStateRuntime(
            encode=lambda value: list(value.encode()),
            generate=generate,
            capacity=4,
        )

    def test_prefill_fork_continue_release_are_owner_isolated(self) -> None:
        root = self.runtime.prefill(owner_id="owner", prompt="root", branch="root")
        children = self.runtime.fork(
            owner_id="owner",
            parent_state_id=root["state_id"],
            branches=["a", "b"],
        )
        result = self.runtime.continue_many(
            owner_id="owner",
            items=[{"state_id": children[0]["state_id"], "input": "+next"}],
            stops=["</tool_call>"],
            max_tokens=8,
        )[0]
        self.assertEqual(result["text"], "answer")
        self.assertTrue(
            self.runtime.records[children[0]["state_id"]].prompt.endswith(
                "answer</tool_call>"
            )
        )
        with self.assertRaises(PermissionError):
            self.runtime.release(
                owner_id="other", state_ids=[children[0]["state_id"]]
            )
        released = self.runtime.release(
            owner_id="owner",
            state_ids=[root["state_id"], *(item["state_id"] for item in children)],
        )
        self.assertEqual(released["released"], 3)
        self.assertEqual(self.runtime.health()["allocated"], 0)


if __name__ == "__main__":
    unittest.main()
