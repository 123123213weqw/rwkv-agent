#!/usr/bin/env python3
"""Create the deterministic, leakage-audited RWKV-Agent-FitGen-v1 split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


PROJECT = Path(__file__).resolve().parents[1]
for value in (str(PROJECT), str(PROJECT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmarks.agent_benchmark_schema import validate_case  # noqa: E402


SCHEMA_VERSION = "rwkv-agent-fitgen-split.v1"
DEFAULT_SEED = "rwkv-agent-fitgen-v1-20260726"
SPLITS = ("train", "dev", "fit_id", "structural_ood")
TARGETS = {
    "bfcl": {"train": 650, "dev": 100, "fit_id": 100, "structural_ood": 150},
    "webwalkerqa": {"train": 400, "dev": 80, "fit_id": 80, "structural_ood": 120},
    "frames": {"train": 480, "dev": 100, "fit_id": 100, "structural_ood": 144},
    "longbench_v2": {"train": 280, "dev": 50, "fit_id": 70, "structural_ood": 103},
    "alce": {"train": 650, "dev": 100, "fit_id": 100, "structural_ood": 150},
}


def stable_hash(seed: str, *values: object) -> str:
    payload = "\x1f".join((seed, *(str(value) for value in values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_load(path: Path) -> list[dict[str, Any]]:
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


def jsonl_dump(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalized_domain(value: str) -> str:
    hostname = (urlsplit(str(value or "")).hostname or "").casefold().strip(".")
    return hostname.removeprefix("www.")


def normalized_answer(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def tool_key(tool: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "name": str(tool.get("name") or ""),
            "parameters": tool.get("parameters") or {},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def case_stratum(dataset: str, case: Mapping[str, Any]) -> str:
    metadata = dict(case.get("metadata") or {})
    if dataset == "bfcl":
        return str(metadata.get("category") or "unknown")
    if dataset == "webwalkerqa":
        return "/".join(
            (
                str(case.get("language") or "unknown"),
                str(metadata.get("difficulty") or "unknown"),
                str(metadata.get("question_type") or "unknown"),
            )
        )
    if dataset == "frames":
        return str(metadata.get("reasoning_types") or "unknown")
    if dataset == "longbench_v2":
        return "/".join(
            (
                str(metadata.get("domain") or "unknown"),
                str(metadata.get("difficulty") or "unknown"),
                str(metadata.get("context_bucket") or "unknown"),
            )
        )
    if dataset == "alce":
        return str(metadata.get("subset") or "unknown")
    raise ValueError(f"unsupported dataset: {dataset}")


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def isolation_keys(dataset: str, case: Mapping[str, Any]) -> set[str]:
    gold = dict(case.get("gold") or {})
    metadata = dict(case.get("metadata") or {})
    if dataset == "bfcl":
        return {"tool:" + tool_key(tool) for tool in case.get("available_tools") or []}
    if dataset == "webwalkerqa":
        return {"root-domain:" + normalized_domain(str(metadata.get("root_url") or ""))}
    if dataset == "frames":
        keys = {"url:" + str(uri) for uri in gold.get("source_uris") or []}
        keys.update(
            "answer:" + normalized_answer(str(answer))
            for answer in gold.get("answers") or []
            if normalized_answer(str(answer))
        )
        return keys
    if dataset == "longbench_v2":
        source_id = str(metadata.get("source_id") or case.get("id") or "")
        return {"document:" + source_id}
    if dataset == "alce":
        return {"document:" + str(uri) for uri in gold.get("source_uris") or []}
    raise ValueError(f"unsupported dataset: {dataset}")


def isolation_groups(
    dataset: str,
    cases: Sequence[Mapping[str, Any]],
) -> list[list[int]]:
    disjoint = DisjointSet(len(cases))
    by_key: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        keys = isolation_keys(dataset, case) or {"case:" + str(case["id"])}
        for key in keys:
            by_key[key].append(index)
    for indexes in by_key.values():
        for index in indexes[1:]:
            disjoint.union(indexes[0], index)
    output: dict[int, list[int]] = defaultdict(list)
    for index in range(len(cases)):
        output[disjoint.find(index)].append(index)
    return list(output.values())


def _stratum_error(
    counts: Counter[str],
    desired: Mapping[str, float],
) -> float:
    return sum(abs(counts.get(key, 0) - value) for key, value in desired.items())


def select_ood_groups(
    dataset: str,
    cases: Sequence[Mapping[str, Any]],
    *,
    target: int,
    seed: str,
    tolerance: int = 5,
) -> set[int]:
    """Choose complete isolation groups near target with deterministic balance."""

    groups = isolation_groups(dataset, cases)
    total_strata = Counter(case_stratum(dataset, case) for case in cases)
    desired = {
        key: count * target / len(cases)
        for key, count in total_strata.items()
    }
    metadata = []
    for indexes in groups:
        strata = Counter(case_stratum(dataset, cases[index]) for index in indexes)
        identity = ",".join(sorted(str(cases[index]["id"]) for index in indexes))
        metadata.append((indexes, strata, stable_hash(seed, dataset, identity)))

    selected: set[int] = set()
    selected_groups: set[int] = set()
    selected_strata: Counter[str] = Counter()
    while len(selected) < target - tolerance:
        candidates = []
        for group_index, (indexes, strata, identity_hash) in enumerate(metadata):
            if group_index in selected_groups:
                continue
            new_total = len(selected) + len(indexes)
            if new_total > target + tolerance:
                continue
            new_strata = selected_strata + strata
            cost = (
                abs(target - new_total),
                round(_stratum_error(new_strata, desired), 9),
                identity_hash,
            )
            candidates.append((cost, group_index, indexes, new_strata))
        if not candidates:
            break
        _, group_index, indexes, new_strata = min(candidates, key=lambda item: item[0])
        selected_groups.add(group_index)
        selected.update(indexes)
        selected_strata = new_strata
    if abs(len(selected) - target) > tolerance:
        raise RuntimeError(
            f"{dataset} OOD split differs from target by more than {tolerance}: "
            f"{len(selected)} vs {target}"
        )
    return selected


def stratified_partition(
    dataset: str,
    cases: Sequence[Mapping[str, Any]],
    indexes: Sequence[int],
    *,
    targets: Mapping[str, int],
    seed: str,
) -> dict[str, set[int]]:
    """Assign remaining cases exactly while approximating stratum proportions."""

    if sum(targets.values()) != len(indexes):
        raise ValueError("partition targets must cover every remaining case")
    output = {name: set() for name in targets}
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for index in indexes:
        by_stratum[case_stratum(dataset, cases[index])].append(index)
    for stratum in by_stratum:
        by_stratum[stratum].sort(
            key=lambda index: stable_hash(seed, dataset, stratum, cases[index]["id"])
        )
    total = len(indexes)
    desired_by_stratum = {
        name: {
            stratum: len(values) * targets[name] / total
            for stratum, values in by_stratum.items()
        }
        for name in targets
    }
    ordered = []
    for stratum, values in sorted(by_stratum.items()):
        for position, index in enumerate(values):
            ordered.append(
                (
                    position / max(1, len(values)),
                    stable_hash(seed, dataset, "order", cases[index]["id"]),
                    stratum,
                    index,
                )
            )
    assigned_strata = {name: Counter() for name in targets}
    for _, _, stratum, index in sorted(ordered):
        available = [name for name in targets if len(output[name]) < targets[name]]
        name = max(
            available,
            key=lambda candidate: (
                desired_by_stratum[candidate][stratum]
                - assigned_strata[candidate][stratum],
                targets[candidate] - len(output[candidate]),
                stable_hash(seed, dataset, stratum, index, candidate),
            ),
        )
        output[name].add(index)
        assigned_strata[name][stratum] += 1
    return output


@dataclass(frozen=True)
class DatasetSplit:
    dataset: str
    cases: list[dict[str, Any]]
    indexes: dict[str, set[int]]


def build_dataset_split(
    dataset: str,
    cases: list[dict[str, Any]],
    *,
    targets: Mapping[str, int],
    seed: str,
) -> DatasetSplit:
    if set(targets) != set(SPLITS):
        raise ValueError(f"{dataset} targets must contain {SPLITS}")
    if sum(targets.values()) != len(cases):
        raise ValueError(f"{dataset} target total does not match cases")
    ood = select_ood_groups(
        dataset,
        cases,
        target=targets["structural_ood"],
        seed=seed,
    )
    remaining = [index for index in range(len(cases)) if index not in ood]
    non_ood_targets = {name: targets[name] for name in SPLITS[:-1]}
    difference = len(remaining) - sum(non_ood_targets.values())
    non_ood_targets["train"] += difference
    indexes = stratified_partition(
        dataset,
        cases,
        remaining,
        targets=non_ood_targets,
        seed=seed,
    )
    indexes["structural_ood"] = ood
    return DatasetSplit(dataset=dataset, cases=cases, indexes=indexes)


def audit_split(value: DatasetSplit) -> dict[str, Any]:
    all_indexes = set().union(*value.indexes.values())
    if all_indexes != set(range(len(value.cases))):
        raise RuntimeError(f"{value.dataset} split does not cover every case")
    total_members = sum(len(indexes) for indexes in value.indexes.values())
    if total_members != len(value.cases):
        raise RuntimeError(f"{value.dataset} split contains duplicate cases")
    ood_keys = set().union(
        *(
            isolation_keys(value.dataset, value.cases[index])
            for index in value.indexes["structural_ood"]
        )
    )
    train_side = set().union(
        *(
            isolation_keys(value.dataset, value.cases[index])
            for name in SPLITS[:-1]
            for index in value.indexes[name]
        )
    )
    overlap = sorted(ood_keys & train_side)
    if overlap:
        raise RuntimeError(
            f"{value.dataset} structural OOD key leakage: {overlap[:5]}"
        )
    return {
        "all_case_ids_unique": True,
        "all_cases_covered": True,
        "structural_ood_key_overlap": 0,
        "structural_ood_groups": len(
            isolation_groups(
                value.dataset,
                [value.cases[index] for index in value.indexes["structural_ood"]],
            )
        ),
    }


def write_split(
    value: DatasetSplit,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for split in SPLITS:
        rows = sorted(
            (value.cases[index] for index in value.indexes[split]),
            key=lambda case: str(case["id"]),
        )
        parent = "locked" if split in {"fit_id", "structural_ood"} else "training"
        path = output_dir / parent / split / f"{value.dataset}.jsonl"
        jsonl_dump(path, rows)
        relative = str(path.relative_to(output_dir))
        artifacts[relative] = {
            "bytes": path.stat().st_size,
            "cases": len(rows),
            "sha256": file_sha256(path),
        }
    return artifacts


def create_split(
    *,
    core_dir: Path,
    output_dir: Path,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "RWKV-Agent-FitGen-v1",
        "seed": seed,
        "targets": TARGETS,
        "group_rules": {
            "bfcl": "connected components of function name plus parameter schema",
            "webwalkerqa": "root hostname with leading www removed",
            "frames": "connected components of exact Gold URL or normalized Gold answer",
            "longbench_v2": "source document ID; strata retain domain/sub-domain coverage",
            "alce": "connected components of Gold document IDs",
        },
        "datasets": {},
        "artifacts": {},
    }
    for dataset, targets in TARGETS.items():
        source = core_dir / f"{dataset}.jsonl"
        cases = jsonl_load(source)
        value = build_dataset_split(
            dataset,
            cases,
            targets=targets,
            seed=seed,
        )
        audit = audit_split(value)
        artifacts = write_split(value, output_dir=output_dir)
        manifest["artifacts"].update(artifacts)
        manifest["datasets"][dataset] = {
            "source": str(source),
            "source_cases": len(cases),
            "source_sha256": file_sha256(source),
            "actual": {
                split: len(value.indexes[split])
                for split in SPLITS
            },
            "strata": {
                split: dict(
                    sorted(
                        Counter(
                            case_stratum(dataset, cases[index])
                            for index in value.indexes[split]
                        ).items()
                    )
                )
                for split in SPLITS
            },
            "audit": audit,
        }
    manifest_path = output_dir / "manifest.json"
    json_dump(manifest_path, manifest)
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    manifest = create_split(
        core_dir=args.core_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
