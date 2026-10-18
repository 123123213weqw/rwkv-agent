from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.create_fitgen_sft import (
    SCHEMA_VERSION,
    build,
    canonical_bfcl_calls,
    case_records,
    concise_supported_alce_answer,
)
from rwkv_agent.claim_verifier import verify_answer_claims


class FitGenSFTTests(unittest.TestCase):
    def test_bfcl_targets_follow_schema_not_reference_sentinels(self) -> None:
        case = {
            "id": "bfcl-x",
            "dataset": "bfcl",
            "prompt": "do it",
            "available_tools": [
                {
                    "name": "demo.run",
                    "parameters": {
                        "type": "dict",
                        "required": ["location", "population"],
                        "properties": {
                            "location": {"type": "string"},
                            "optional_number": {"type": "float"},
                            "population": {"type": "dict"},
                        },
                    },
                }
            ],
            "gold": {
                "tool_calls": [
                    {
                        "name": "demo.run",
                        "arguments": {
                            "location": "LA",
                            "optional_number": "",
                            "population": {"adults": [2], "children": [1]},
                            "annotation_only": "drop",
                        },
                    }
                ]
            },
        }
        self.assertEqual(
            canonical_bfcl_calls(case),
            [
                {
                    "name": "demo.run",
                    "arguments": {
                        "location": "LA",
                        "population": {"adults": 2, "children": 1},
                    },
                }
            ],
        )

    def test_bfcl_parallel_target_contains_every_call(self) -> None:
        case = {
            "id": "b1",
            "dataset": "bfcl",
            "prompt": "Do both.",
            "available_tools": [],
            "gold": {
                "tool_calls": [
                    {"name": "a", "arguments": {"x": 1}},
                    {"name": "b", "arguments": {"y": 2}},
                ]
            },
        }
        row = case_records(case)[0]
        self.assertEqual(row["schema_version"], SCHEMA_VERSION)
        self.assertIn('"name":"a"', row["response"])
        self.assertIn('"name":"b"', row["response"])
        self.assertTrue(row["response"].endswith("</tool_call>"))

    def test_search_builds_tool_and_grounded_answer_examples(self) -> None:
        case = {
            "id": "w1",
            "dataset": "webwalkerqa",
            "prompt": "Who maintains ExampleDB?",
            "gold": {
                "answers": ["Example Foundation."],
                "source_uris": ["https://example.invalid/db"],
            },
        }
        rows = case_records(case)
        self.assertEqual([row["task"] for row in rows], ["web_search_call", "web_evidence_answer"])
        self.assertIn('"name":"web_search"', rows[0]["response"])
        self.assertIn("[W1]", rows[1]["response"])

    def test_alce_target_is_concise_and_supported_by_its_citation(self) -> None:
        evidence = [
            {
                "id": "W1",
                "title": "Mars",
                "content": "Mars has two moons named Phobos and Deimos.",
                "uri": "alce://mars",
            },
            {
                "id": "W2",
                "title": "Unsupported",
                "content": "Venus has no natural satellite.",
                "uri": "alce://venus",
            },
        ]
        answer = concise_supported_alce_answer(
            ["Mars has two moons named Phobos and Deimos, discovered in 1877."],
            evidence,
        )
        self.assertIn("[W1]", answer)
        self.assertNotIn("1877", answer)
        claims = verify_answer_claims(answer, evidence)
        self.assertTrue(claims)
        self.assertTrue(all(claim["supported"] for claim in claims))

    def test_alce_target_falls_back_to_verbatim_evidence(self) -> None:
        evidence = [
            {
                "id": "W1",
                "title": "Primary source",
                "content": "The observatory opened during the spring of 1984.",
                "uri": "alce://observatory",
            }
        ]
        answer = concise_supported_alce_answer(
            ["No reference vocabulary overlaps this passage."],
            evidence,
            extractive_words=6,
        )
        self.assertIn("The observatory opened during the spring [W1]", answer)
        self.assertTrue(all(claim["supported"] for claim in verify_answer_claims(answer, evidence)))

    def test_alce_list_target_keeps_complete_supported_items(self) -> None:
        evidence = [
            {
                "id": "W1",
                "title": "Towers",
                "content": "Buffalo City Hall and Seneca One Tower are in Buffalo.",
                "uri": "alce://towers",
            }
        ]
        answer = concise_supported_alce_answer(
            ["Seneca One Tower, Buffalo City Hall, Unsupported Building."],
            evidence,
            list_answer=True,
        )
        self.assertIn("Seneca One Tower [W1].", answer)
        self.assertIn("Buffalo City Hall [W1].", answer)
        self.assertNotIn("Unsupported Building", answer)
        self.assertTrue(all(claim["supported"] for claim in verify_answer_claims(answer, evidence)))

    def test_build_reads_only_training_splits_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "split"
            output = Path(temporary) / "out"
            for split in ("train", "dev"):
                directory = root / "training" / split
                directory.mkdir(parents=True)
                for dataset in ("bfcl", "webwalkerqa", "frames", "longbench_v2", "alce"):
                    case = {
                        "id": f"{split}-{dataset}",
                        "dataset": dataset,
                        "prompt": "Question?",
                        "gold": {"answers": ["A"], "source_uris": ["https://example.invalid/a"]},
                    }
                    if dataset == "bfcl":
                        case.update(available_tools=[])
                        case["gold"] = {"tool_calls": [{"name": "a", "arguments": {}}]}
                    elif dataset == "longbench_v2":
                        case.update(context="A relevant context.")
                    elif dataset == "alce":
                        case.update(evidence_context=[{"id": "W1", "content": "A", "uri": "alce://1"}])
                    (directory / f"{dataset}.jsonl").write_text(json.dumps(case) + "\n")
            (root / "locked").mkdir()
            (root / "locked" / "sentinel").write_text("must not be read")
            manifest = build(root, output)
            self.assertEqual(manifest["locked_files_read"], 0)
            self.assertTrue((output / "train.jsonl").is_file())
            with self.assertRaises(FileExistsError):
                build(root, output)


if __name__ == "__main__":
    unittest.main()
