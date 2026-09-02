#!/usr/bin/env python3
"""Evaluate RWKV Agent measurements against versioned production/S-level gates.

Every metric is an independent hard gate.  Missing measurements fail closed;
there is deliberately no weighted or grand score.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


MEASUREMENT_SCHEMA = "rwkv-agent-s-level-measurements.v1"
REPORT_SCHEMA = "rwkv-agent-s-level-report.v1"
DEFAULT_TARGETS = Path(__file__).with_name("s_level_targets.v1.json")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _nested(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    track: str
    actual: float | None
    operator: str
    target: float
    passed: bool
    missing: bool
    gap: float | None


def _evaluate_gate(
    gate_id: str,
    track: str,
    actual: Any,
    operator: str,
    target: Any,
) -> GateResult:
    measured = _number(actual)
    wanted = _number(target)
    if wanted is None:
        raise ValueError(f"target for {gate_id} must be numeric")
    passed = False
    gap: float | None = None
    if measured is not None:
        if operator == ">=":
            passed = measured >= wanted
            gap = max(0.0, wanted - measured)
        elif operator == "<=":
            passed = measured <= wanted
            gap = max(0.0, measured - wanted)
        elif operator == "==":
            passed = measured == wanted
            gap = abs(measured - wanted)
        else:
            raise ValueError(f"unsupported operator for {gate_id}: {operator}")
    return GateResult(
        gate_id=gate_id,
        track=track,
        actual=measured,
        operator=operator,
        target=wanted,
        passed=passed,
        missing=measured is None,
        gap=gap,
    )


def _subgroup_target(operator: str, target: float, floor_ratio: float) -> float:
    if operator == ">=":
        return target * floor_ratio
    if operator == "<=":
        return 0.0 if target == 0 else target / floor_ratio
    return target


def evaluate(
    measurements: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    if measurements.get("schema_version") != MEASUREMENT_SCHEMA:
        raise ValueError(f"unsupported measurement schema: {measurements.get('schema_version')}")
    if profile not in {"production", "s_level"}:
        raise ValueError(f"unsupported target profile: {profile}")

    integrity = targets.get("integrity")
    definitions = targets.get("metrics")
    if not isinstance(integrity, Mapping) or not isinstance(definitions, Mapping):
        raise ValueError("targets must define integrity and metrics objects")
    values = measurements.get("metrics")
    values = values if isinstance(values, Mapping) else {}

    gates = [
        _evaluate_gate(
            "integrity.minimum_cases",
            "integrity",
            measurements.get("cases"),
            ">=",
            _nested(integrity, f"minimum_cases.{profile}"),
        ),
        _evaluate_gate(
            "integrity.minimum_live_repetitions",
            "integrity",
            measurements.get("live_repetitions"),
            ">=",
            _nested(integrity, f"minimum_live_repetitions.{profile}"),
        ),
    ]

    for name, raw_definition in definitions.items():
        if not isinstance(raw_definition, Mapping):
            raise ValueError(f"metric target must be an object: {name}")
        gates.append(
            _evaluate_gate(
                f"metric.{name}",
                str(raw_definition.get("track") or "unknown"),
                values.get(name),
                str(raw_definition.get("operator") or ""),
                raw_definition.get(profile),
            )
        )

    floor_ratio = _number(integrity.get("subgroup_floor_ratio"))
    floor_ratio = floor_ratio if floor_ratio is not None else 0.9
    subgroup_values = measurements.get("language_groups")
    subgroup_values = subgroup_values if isinstance(subgroup_values, Mapping) else {}
    required_groups = [str(value) for value in integrity.get("required_language_groups") or []]
    for group in required_groups:
        group_metrics = subgroup_values.get(group)
        group_metrics = group_metrics if isinstance(group_metrics, Mapping) else {}
        for name, raw_definition in definitions.items():
            if not isinstance(raw_definition, Mapping) or not raw_definition.get("subgroup_gate"):
                continue
            operator = str(raw_definition.get("operator") or "")
            target = _number(raw_definition.get(profile))
            if target is None:
                raise ValueError(f"subgroup target for {name} must be numeric")
            gates.append(
                _evaluate_gate(
                    f"language.{group}.{name}",
                    f"language:{group}",
                    group_metrics.get(name),
                    operator,
                    _subgroup_target(operator, target, floor_ratio),
                )
            )

    failed = [item for item in gates if not item.passed]
    known_failed = sorted(
        (item for item in failed if not item.missing),
        key=lambda item: item.gap or 0.0,
        reverse=True,
    )
    missing = [item for item in failed if item.missing]
    return {
        "schema_version": REPORT_SCHEMA,
        "benchmark": str(targets.get("benchmark") or "RWKV-Agent-S-Level-v1"),
        "profile": profile,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_passed": not failed,
        "grand_score": None,
        "gate_counts": {
            "total": len(gates),
            "passed": len(gates) - len(failed),
            "failed": len(failed),
            "missing": len(missing),
        },
        "gates": [asdict(item) for item in gates],
        "failed_gates": [asdict(item) for item in failed],
        "largest_known_gaps": [asdict(item) for item in known_failed[:10]],
        "missing_gate_ids": [item.gate_id for item in missing],
    }


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--profile", choices=("production", "s_level"), default="s_level")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    measurements = _load_object(args.measurements)
    report = evaluate(measurements, _load_object(args.targets), profile=args.profile)
    _atomic_write(args.output, report)
    print(
        json.dumps(
            {
                "release_passed": report["release_passed"],
                "profile": report["profile"],
                "gate_counts": report["gate_counts"],
                "output": str(args.output.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["release_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
