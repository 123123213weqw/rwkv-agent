#!/usr/bin/env python3
"""Build a compact, deterministic regression set from the frozen Dev split.

The builder deliberately accepts only ``training/dev``-style inputs. It never
reads Fit-ID, Structural-OOD, or Fresh-Web test files, so the resulting set is
safe for routine development and regression testing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
for value in (str(PROJECT), str(PROJECT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmarks.agent_benchmark_schema import validate_case  # noqa: E402


SCHEMA_VERSION = "rwkv-agent-unified-regression-manifest.v1"
DATASET_NAME = "RWKV-Agent-Unified-Regression-v1"
DEFAULT_SEED = "rwkv-agent-unified-regression-v1-20260728"
DATASETS = ("bfcl", "webwalkerqa", "frames", "longbench_v2", "alce")


def stable_hash(seed: str, *values: object) -> str:
    payload = "\x1f".join((seed, *(str(value) for value in values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            rows.append(validate_case(value))
    return rows


def _jsonl_dump(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metadata(case: Mapping[str, Any]) -> dict[str, Any]:
    return dict(case.get("metadata") or {})


def case_features(dataset: str, case: Mapping[str, Any]) -> tuple[str, ...]:
    """Return namespaced strata used to preserve Dev-set diversity."""

    metadata = _metadata(case)
    features = {f"language={case.get('language') or 'unknown'}"}
    if dataset == "bfcl":
        features.add(f"category={metadata.get('category') or 'unknown'}")
    elif dataset == "webwalkerqa":
        for key in ("difficulty", "domain", "question_type"):
            features.add(f"{key}={metadata.get(key) or 'unknown'}")
    elif dataset == "frames":
        raw = str(metadata.get("reasoning_types") or "unknown")
        for value in raw.split("|"):
            features.add(f"reasoning={value.strip() or 'unknown'}")
    elif dataset == "longbench_v2":
        for key in ("difficulty", "domain", "sub_domain", "context_bucket"):
            features.add(f"{key}={metadata.get(key) or 'unknown'}")
    elif dataset == "alce":
        features.add(f"subset={metadata.get('subset') or 'unknown'}")
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return tuple(sorted(features))


def _desired_feature_counts(
    dataset: str,
    cases: Sequence[Mapping[str, Any]],
    target: int,
) -> dict[str, float]:
    source = Counter(
        feature
        for case in cases
        for feature in case_features(dataset, case)
    )
    desired = {
        feature: count * target / len(cases)
        for feature, count in source.items()
    }
    # BFCL's difficult parallel categories are important release gates. Use an
    # equal category quota instead of allowing the larger simple bucket to
    # dominate a compact daily regression set.
    if dataset == "bfcl":
        categories = sorted(
            feature for feature in source if feature.startswith("category=")
        )
        if categories:
            quota = target / len(categories)
            desired.update({feature: quota for feature in categories})
    return desired


def select_balanced(
    dataset: str,
    cases: Sequence[Mapping[str, Any]],
    *,
    target: int,
    seed: str,
) -> list[dict[str, Any]]:
    """Select exactly ``target`` cases with deterministic stratum balancing."""

    if target <= 0:
        raise ValueError("target must be positive")
    if len(cases) < target:
        raise ValueError(f"{dataset}: only {len(cases)} cases for target {target}")
    desired = _desired_feature_counts(dataset, cases, target)
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    remaining = [dict(case) for case in cases]
    for step in range(target):
        progress = (step + 1) / target
        candidates: list[tuple[tuple[float, str], int, tuple[str, ...]]] = []
        for index, case in enumerate(remaining):
            features = case_features(dataset, case)
            after = selected_counts.copy()
            after.update(features)
            error = sum(
                abs(after.get(feature, 0) - expected * progress)
                / max(expected, 1.0)
                for feature, expected in desired.items()
            )
            identity = stable_hash(seed, dataset, case.get("id"))
            candidates.append(((round(error, 12), identity), index, features))
        _, index, features = min(candidates, key=lambda item: item[0])
        selected.append(remaining.pop(index))
        selected_counts.update(features)
    return sorted(selected, key=lambda case: str(case.get("id") or ""))


def _distribution(dataset: str, cases: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                feature
                for case in cases
                for feature in case_features(dataset, case)
            ).items()
        )
    )


def load_dev_cases(dev_dir: Path) -> dict[str, list[dict[str, Any]]]:
    resolved = dev_dir.expanduser().resolve()
    if "locked" in resolved.parts:
        raise ValueError("locked test data cannot be used for a regression set")
    if resolved.name != "dev" or resolved.parent.name != "training":
        raise ValueError("input must be the frozen training/dev directory")
    output: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for dataset in DATASETS:
        path = resolved / f"{dataset}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = _jsonl_load(path)
        for row in rows:
            if str(row.get("dataset") or "") != dataset:
                raise ValueError(f"{path}: dataset mismatch for {row.get('id')}")
            case_id = str(row.get("id") or "")
            if case_id in seen:
                raise ValueError(f"duplicate case id: {case_id}")
            seen.add(case_id)
        output[dataset] = rows
    return output


def create_regression_set(
    *,
    dev_dir: Path,
    output_dir: Path,
    seed: str = DEFAULT_SEED,
    per_dataset: int = 40,
) -> dict[str, Any]:
    """Create one combined JSONL, review index, README, and manifest."""

    dev_dir = dev_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    source = load_dev_cases(dev_dir)
    selected = {
        dataset: select_balanced(
            dataset,
            source[dataset],
            target=per_dataset,
            seed=seed,
        )
        for dataset in DATASETS
    }
    combined = [case for dataset in DATASETS for case in selected[dataset]]
    if len({str(case["id"]) for case in combined}) != len(combined):
        raise ValueError("selected case IDs are not globally unique")

    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        cases_path = temporary / "cases.jsonl"
        index_path = temporary / "index.jsonl"
        _jsonl_dump(cases_path, combined)
        dataset_paths: dict[str, Path] = {}
        for dataset in DATASETS:
            dataset_path = temporary / f"{dataset}.jsonl"
            _jsonl_dump(dataset_path, selected[dataset])
            dataset_paths[dataset] = dataset_path
        _jsonl_dump(
            index_path,
            (
                {
                    "id": case["id"],
                    "dataset": dataset,
                    "track": case["track"],
                    "language": case["language"],
                    "features": list(case_features(dataset, case)),
                }
                for dataset in DATASETS
                for case in selected[dataset]
            ),
        )
        source_files = {
            dataset: {
                "file": f"{dataset}.jsonl",
                "rows": len(source[dataset]),
                "sha256": file_sha256(dev_dir / f"{dataset}.jsonl"),
            }
            for dataset in DATASETS
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": DATASET_NAME,
            "purpose": "bounded daily development and end-to-end regression",
            "source_split": "dev",
            "locked_test_files_read": 0,
            "seed": seed,
            "per_dataset": per_dataset,
            "total_cases": len(combined),
            "dataset_order": list(DATASETS),
            "source_files": source_files,
            "selected": {
                dataset: {
                    "rows": len(selected[dataset]),
                    "ids": [str(case["id"]) for case in selected[dataset]],
                    "distribution": _distribution(dataset, selected[dataset]),
                }
                for dataset in DATASETS
            },
            "files": {
                "cases.jsonl": {
                    "rows": len(combined),
                    "bytes": cases_path.stat().st_size,
                    "sha256": file_sha256(cases_path),
                },
                "index.jsonl": {
                    "rows": len(combined),
                    "bytes": index_path.stat().st_size,
                    "sha256": file_sha256(index_path),
                },
                **{
                    f"{dataset}.jsonl": {
                        "rows": len(selected[dataset]),
                        "bytes": dataset_paths[dataset].stat().st_size,
                        "sha256": file_sha256(dataset_paths[dataset]),
                    }
                    for dataset in DATASETS
                },
            },
            "publication": {
                "raw_cases_public": False,
                "reason": "ALCE source metadata does not declare a publishable license",
                "safe_to_publish": ["manifest.json", "index.jsonl", "README.md"],
            },
        }
        _json_dump(temporary / "manifest.json", manifest)
        (temporary / "README.md").write_text(
            "# RWKV-Agent Unified Regression v1\n\n"
            "A deterministic 200-case daily regression set selected only from "
            "the frozen FitGen Dev split. It contains 40 cases each from BFCL, "
            "WebWalkerQA, FRAMES, LongBench v2, and ALCE. Locked Fit-ID, "
            "Structural-OOD, and Fresh-Web Gold are not read.\n\n"
            "`cases.jsonl` stays private because ALCE does not declare a "
            "publishable license. `manifest.json` and `index.jsonl` contain the "
            "reproducibility metadata. The five `<dataset>.jsonl` files expose "
            "the same selected rows in the layout expected by the benchmark "
            "runner.\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--per-dataset", type=int, default=40)
    args = parser.parse_args()
    manifest = create_regression_set(
        dev_dir=args.dev_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        per_dataset=args.per_dataset,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
