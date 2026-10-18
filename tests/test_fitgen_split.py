from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.agent_benchmark_schema import CASE_SCHEMA_VERSION
from benchmarks.create_fitgen_split import (
    audit_split,
    build_dataset_split,
    create_split,
    isolation_keys,
)


def case(dataset: str, index: int, *, group: str, stratum: str) -> dict:
    value = {
        "schema_version": CASE_SCHEMA_VERSION,
        "id": f"{dataset}-{index:04d}",
        "dataset": dataset,
        "split": "fixture",
        "track": "tool_protocol" if dataset == "bfcl" else "web_research",
        "language": "en",
        "prompt": f"question {index}",
        "gold": {},
        "limits": {
            "max_rounds": 2,
            "max_requests": 8,
            "max_latency_ms": 20000,
        },
        "metadata": {},
    }
    if dataset == "bfcl":
        value["available_tools"] = [
            {
                "name": group,
                "description": "fixture",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        value["gold"] = {
            "should_call_tools": True,
            "tool_calls": [{"name": group, "arguments": {}}],
        }
        value["metadata"]["category"] = stratum
    elif dataset == "webwalkerqa":
        value["gold"] = {
            "answers": [f"answer {index}"],
            "answerable": True,
            "requires_citations": True,
            "source_uris": [f"https://{group}/page/{index}"],
        }
        value["metadata"].update(
            {
                "root_url": f"https://{group}/",
                "difficulty": stratum,
                "question_type": "single_source",
            }
        )
    else:
        raise ValueError(dataset)
    return value


class FitGenSplitTests(unittest.TestCase):
    def test_bfcl_ood_tool_components_do_not_leak(self) -> None:
        cases = [
            case(
                "bfcl",
                index,
                group=f"tool_{index // 2}",
                stratum="simple" if index % 2 else "parallel",
            )
            for index in range(40)
        ]
        targets = {"train": 20, "dev": 5, "fit_id": 5, "structural_ood": 10}
        value = build_dataset_split(
            "bfcl",
            cases,
            targets=targets,
            seed="fixture",
        )
        audit_split(value)
        ood = set().union(
            *(isolation_keys("bfcl", cases[index]) for index in value.indexes["structural_ood"])
        )
        rest = set().union(
            *(
                isolation_keys("bfcl", cases[index])
                for split in ("train", "dev", "fit_id")
                for index in value.indexes[split]
            )
        )
        self.assertFalse(ood & rest)

    def test_web_split_is_deterministic_and_domain_disjoint(self) -> None:
        cases = [
            case(
                "webwalkerqa",
                index,
                group=f"site{index // 3}.example",
                stratum=("easy", "medium", "hard")[index % 3],
            )
            for index in range(60)
        ]
        targets = {"train": 30, "dev": 6, "fit_id": 6, "structural_ood": 18}
        first = build_dataset_split(
            "webwalkerqa",
            copy.deepcopy(cases),
            targets=targets,
            seed="fixture",
        )
        second = build_dataset_split(
            "webwalkerqa",
            copy.deepcopy(cases),
            targets=targets,
            seed="fixture",
        )
        self.assertEqual(first.indexes, second.indexes)
        audit_split(first)

    def test_create_split_writes_manifest_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core = root / "core"
            output = root / "output"
            core.mkdir()
            datasets = {
                "bfcl": [case("bfcl", index, group=f"t{index}", stratum="simple") for index in range(4)],
                "webwalkerqa": [case("webwalkerqa", index, group=f"s{index}.example", stratum="easy") for index in range(4)],
            }
            # Exercise the file writer with fixture-sized temporary targets.
            from benchmarks import create_fitgen_split as module

            original = module.TARGETS
            module.TARGETS = {
                "bfcl": {"train": 1, "dev": 1, "fit_id": 1, "structural_ood": 1},
                "webwalkerqa": {"train": 1, "dev": 1, "fit_id": 1, "structural_ood": 1},
            }
            try:
                for name, rows in datasets.items():
                    (core / f"{name}.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8",
                    )
                manifest = create_split(core_dir=core, output_dir=output, seed="fixture")
                self.assertEqual(manifest["benchmark"], "RWKV-Agent-FitGen-v1")
                self.assertTrue((output / "manifest.json").is_file())
                self.assertTrue((output / "locked" / "fit_id" / "bfcl.jsonl").is_file())
                with self.assertRaises(FileExistsError):
                    create_split(core_dir=core, output_dir=output, seed="fixture")
            finally:
                module.TARGETS = original


if __name__ == "__main__":
    unittest.main()
