#!/usr/bin/env python3
"""Compose generated short and long Agent Policy splits without eval leakage."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "rwkv-agent-policy-curriculum.v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_generated(root: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = root.expanduser().resolve()
    manifest_path = resolved / "manifest.json"
    data_path = resolved / f"{split}.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("benchmark_inputs") != []:
        raise ValueError(f"generated source reads benchmark inputs: {resolved}")
    if manifest.get("failure_trace_inputs", []) != []:
        raise ValueError(f"generated source reads failure traces: {resolved}")
    expected = manifest.get(split)
    if not isinstance(expected, dict) or expected.get("sha256") != sha256(data_path):
        raise ValueError(f"source manifest hash mismatch: {data_path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(data_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict) or row.get("split") != split:
            raise ValueError(f"{data_path}:{line_number}: invalid split row")
        if not all(isinstance(row.get(key), str) and row[key] for key in ("id", "trajectory_id", "prompt", "response", "dataset")):
            raise ValueError(f"{data_path}:{line_number}: invalid policy row")
        source_dataset = str(row["dataset"])
        value = dict(row)
        value["source_id"] = value["id"]
        value["source_trajectory_id"] = value["trajectory_id"]
        value["id"] = f"{source_dataset}::{value['id']}"
        value["trajectory_id"] = f"{source_dataset}::{value['trajectory_id']}"
        rows.append(value)
    return rows, {
        "root": str(resolved),
        "manifest_sha256": sha256(manifest_path),
        "data_sha256": sha256(data_path),
        "schema_version": manifest.get("schema_version"),
        "rows": len(rows),
    }


def _validate(train: Sequence[dict[str, Any]], dev: Sequence[dict[str, Any]]) -> dict[str, Any]:
    train_ids = {str(row["id"]) for row in train}
    dev_ids = {str(row["id"]) for row in dev}
    if len(train_ids) != len(train) or len(dev_ids) != len(dev):
        raise ValueError("duplicate composed row id")
    train_trajectories = {str(row["trajectory_id"]) for row in train}
    dev_trajectories = {str(row["trajectory_id"]) for row in dev}
    if train_trajectories & dev_trajectories:
        raise ValueError("Train/Dev trajectory leakage")
    train_pairs = {(str(row["prompt"]), str(row["response"])) for row in train}
    dev_pairs = {(str(row["prompt"]), str(row["response"])) for row in dev}
    if train_pairs & dev_pairs:
        raise ValueError("Train/Dev exact example leakage")
    return {
        "trajectory_overlap": 0,
        "exact_example_overlap": 0,
        "train_trajectories": len(train_trajectories),
        "dev_trajectories": len(dev_trajectories),
    }


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "datasets": dict(sorted(Counter(str(row["dataset"]) for row in rows).items())),
        "tasks": dict(sorted(Counter(str(row.get("task") or "") for row in rows).items())),
        "families": dict(
            sorted(Counter(str(row.get("family") or "") for row in rows).items())
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-dir", required=True, type=Path)
    parser.add_argument("--long-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    short_train, short_train_meta = _load_generated(args.short_dir, "train")
    short_dev, short_dev_meta = _load_generated(args.short_dir, "dev")
    long_train, long_train_meta = _load_generated(args.long_dir, "train")
    long_dev, long_dev_meta = _load_generated(args.long_dir, "dev")
    train = short_train + long_train
    dev = short_dev + long_dev
    leakage = _validate(train, dev)
    train_path = output / "train.jsonl"
    dev_path = output / "dev.jsonl"
    _write(train_path, train)
    _write(dev_path, dev)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "composer": str(Path(__file__).resolve()),
        "composer_sha256": sha256(Path(__file__).resolve()),
        "benchmark_inputs": [],
        "failure_trace_inputs": [],
        "sources": {
            "short": {"train": short_train_meta, "dev": short_dev_meta},
            "long": {"train": long_train_meta, "dev": long_dev_meta},
        },
        "train": _summary(train) | {"path": str(train_path), "sha256": sha256(train_path)},
        "dev": _summary(dev) | {"path": str(dev_path), "sha256": sha256(dev_path)},
        "leakage_audit": leakage,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
