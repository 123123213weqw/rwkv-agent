from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.agent_benchmark_schema import CASE_SCHEMA_VERSION
from benchmarks.create_unified_regression_set import (
    DATASETS,
    create_regression_set,
    load_dev_cases,
)


def fixture_case(dataset: str, index: int) -> dict:
    value = {
        "schema_version": CASE_SCHEMA_VERSION,
        "id": f"{dataset}-{index:04d}",
        "dataset": dataset,
        "split": "dev",
        "track": {
            "bfcl": "tool_protocol",
            "longbench_v2": "long_text",
            "alce": "citation_grounding",
        }.get(dataset, "web_research"),
        "language": "zh" if dataset == "webwalkerqa" and index % 2 else "en",
        "prompt": f"question {index}",
        "gold": {
            "answers": [f"answer {index}"],
            "answerable": True,
            "requires_citations": dataset not in {"bfcl", "longbench_v2"},
        },
        "limits": {
            "max_rounds": 2,
            "max_requests": 8,
            "max_latency_ms": 20000,
        },
        "metadata": {"license": "fixture", "revision": "fixture"},
    }
    if dataset == "bfcl":
        category = ("simple", "multiple", "parallel", "parallel_multiple")[
            index % 4
        ]
        value["metadata"]["category"] = category
        value["available_tools"] = [
            {
                "name": f"tool_{index}",
                "description": "fixture",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        value["gold"] = {
            "should_call_tools": True,
            "tool_calls": [{"name": f"tool_{index}", "arguments": {}}],
        }
    elif dataset == "webwalkerqa":
        value["metadata"].update(
            {
                "difficulty": ("easy", "medium", "hard")[index % 3],
                "domain": ("education", "conference", "game")[index % 3],
                "question_type": ("single_source", "multi_source")[index % 2],
            }
        )
        value["gold"]["source_uris"] = [f"https://example.org/{index}"]
    elif dataset == "frames":
        value["metadata"]["reasoning_types"] = (
            "Numerical reasoning | Multiple constraints"
            if index % 2
            else "Temporal reasoning | Tabular reasoning"
        )
        value["gold"]["source_uris"] = [f"https://example.org/{index}"]
    elif dataset == "longbench_v2":
        value["context"] = f"context {index}"
        value["metadata"].update(
            {
                "difficulty": "hard" if index % 2 else "easy",
                "domain": f"domain-{index % 5}",
                "sub_domain": f"sub-{index % 7}",
                "context_bucket": ("short", "medium", "long")[index % 3],
            }
        )
    elif dataset == "alce":
        value["metadata"]["subset"] = "asqa" if index % 2 else "qampari"
        value["gold"]["source_uris"] = [f"https://example.org/{index}"]
        value["evidence_context"] = []
    return value


def write_dev(root: Path, rows_per_dataset: int = 50) -> None:
    root.mkdir(parents=True)
    for dataset in DATASETS:
        rows = [fixture_case(dataset, index) for index in range(rows_per_dataset)]
        (root / f"{dataset}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )


class UnifiedRegressionSetTests(unittest.TestCase):
    def test_builds_deterministic_balanced_200_case_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev = root / "training" / "dev"
            write_dev(dev)
            first = create_regression_set(
                dev_dir=dev,
                output_dir=root / "first",
                seed="fixture",
            )
            second = create_regression_set(
                dev_dir=dev,
                output_dir=root / "second",
                seed="fixture",
            )
            self.assertEqual(first["total_cases"], 200)
            self.assertEqual(
                first["files"]["cases.jsonl"]["sha256"],
                second["files"]["cases.jsonl"]["sha256"],
            )
            for dataset in DATASETS:
                self.assertEqual(first["selected"][dataset]["rows"], 40)
                self.assertTrue((root / "first" / f"{dataset}.jsonl").is_file())
                self.assertEqual(first["files"][f"{dataset}.jsonl"]["rows"], 40)
            categories = first["selected"]["bfcl"]["distribution"]
            self.assertEqual(categories["category=simple"], 10)
            self.assertEqual(categories["category=multiple"], 10)
            self.assertEqual(categories["category=parallel"], 10)
            self.assertEqual(categories["category=parallel_multiple"], 10)
            subsets = first["selected"]["alce"]["distribution"]
            self.assertEqual(subsets["subset=asqa"], 20)
            self.assertEqual(subsets["subset=qampari"], 20)

    def test_refuses_locked_or_non_dev_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locked = root / "locked" / "fit_id"
            write_dev(locked)
            with self.assertRaisesRegex(ValueError, "locked test data"):
                load_dev_cases(locked)

            other = root / "training" / "fit_id"
            write_dev(other)
            with self.assertRaisesRegex(ValueError, "training/dev"):
                load_dev_cases(other)

    def test_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev = root / "training" / "dev"
            write_dev(dev)
            output = root / "output"
            create_regression_set(dev_dir=dev, output_dir=output, seed="fixture")
            with self.assertRaises(FileExistsError):
                create_regression_set(
                    dev_dir=dev,
                    output_dir=output,
                    seed="fixture",
                )


if __name__ == "__main__":
    unittest.main()
