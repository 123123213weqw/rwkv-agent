from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.agent_benchmark_schema import (
    CASE_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    load_jsonl,
    validate_case,
    validate_result,
)


def tool_case() -> dict:
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "id": "bfcl-1",
        "dataset": "bfcl",
        "split": "test",
        "track": "tool_protocol",
        "language": "en",
        "prompt": "Find Python's current release.",
        "gold": {
            "should_call_tools": True,
            "tool_calls": [
                {
                    "name": "web_search",
                    "arguments": {"query": "Python current release"},
                }
            ],
        },
    }


class AgentBenchmarkSchemaTests(unittest.TestCase):
    def test_accepts_tool_case_and_result(self) -> None:
        self.assertEqual(validate_case(tool_case())["id"], "bfcl-1")
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "case_id": "bfcl-1",
            "status": "ok",
            "answer": "",
            "tool_calls": [
                {
                    "name": "web_search",
                    "arguments": {"query": "Python current release"},
                }
            ],
            "evidence": [],
        }
        self.assertEqual(validate_result(result)["case_id"], "bfcl-1")

    def test_tool_track_requires_explicit_call_policy(self) -> None:
        case = tool_case()
        del case["gold"]["should_call_tools"]
        with self.assertRaisesRegex(ValueError, "should_call_tools"):
            validate_case(case)

    def test_rejects_duplicate_evidence_ids(self) -> None:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "case_id": "x",
            "status": "ok",
            "answer": "answer",
            "evidence": [{"id": "W1"}, {"id": "W1"}],
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_result(result)

    def test_jsonl_loader_reports_line_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            row = json.dumps(tool_case())
            path.write_text(row + "\n" + row + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_jsonl(path, kind="case")


if __name__ == "__main__":
    unittest.main()
