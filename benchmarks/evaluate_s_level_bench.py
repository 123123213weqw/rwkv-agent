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


def _unified_metric(summary: Mapping[str, Any], name: str, statistic: str | None = None) -> float | None:
    value = _nested(summary, f"unified_metrics.{name}")
    if not isinstance(value, Mapping):
        return None
    if statistic is not None:
        return _number(value.get(statistic))
    return _number(value.get("rate" if value.get("kind") == "rate" else "mean"))


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


def measurements_from_fitgen(
    summary: Mapping[str, Any],
    funnel: Mapping[str, Any] | None = None,
    *,
    live_repetitions: int = 1,
) -> dict[str, Any]:
    """Create a conservative partial S-level measurement from existing artifacts.

    Only metrics whose semantics match the S-level contract are mapped.  All
    others remain absent and therefore fail closed instead of being guessed.
    """

    metrics: dict[str, float] = {}
    cases = int(_number(summary.get("cases")) or 0)
    mappings = {
        "citation_validity_precision": ("citation_validity_precision", None),
        "citation_exact_page_recall": ("citation_exact_page_recall", None),
        "important_claim_citation_coverage": ("claim_citation_coverage", None),
        "unsupported_claim_rate": ("unsupported_claim_rate", None),
        "status_ok_rate": ("result_status_ok", None),
        "search_p50_latency_ms": ("latency_ms", "p50"),
        "search_p95_latency_ms": ("latency_ms", "p95"),
    }
    for output_name, (source_name, statistic) in mappings.items():
        value = _unified_metric(summary, source_name, statistic)
        if value is not None:
            metrics[output_name] = value
    state_leaks = _number(_nested(summary, "reliability.state_leak_count"))
    if state_leaks is not None:
        metrics["state_leak_count"] = state_leaks

    language_groups: dict[str, dict[str, float]] = {}
    if isinstance(funnel, Mapping):
        stage_rates = funnel.get("stage_hit_rates")
        macro_recalls = funnel.get("stage_macro_recalls")
        stage_rates = stage_rates if isinstance(stage_rates, Mapping) else {}
        macro_recalls = macro_recalls if isinstance(macro_recalls, Mapping) else {}
        discovery_domain = _number(stage_rates.get("domain_candidate_hit"))
        exact_discovery = _number(macro_recalls.get("raw_candidate_recall"))
        final_evidence = _number(macro_recalls.get("final_evidence_recall"))
        if discovery_domain is not None:
            metrics["domain_recall_at_10"] = discovery_domain
        if exact_discovery is not None:
            metrics["exact_page_recall_at_20"] = exact_discovery
        if final_evidence is not None:
            metrics["final_evidence_exact_recall"] = final_evidence
        raw_hit = _number(stage_rates.get("exact_raw_candidate_hit"))
        final_hit = _number(stage_rates.get("exact_final_evidence_hit"))
        if raw_hit is not None and raw_hit > 0 and final_hit is not None:
            metrics["discovered_to_evidence_retention_rate"] = min(1.0, final_hit / raw_hit)
        search_invoked = _number(stage_rates.get("search_invoked"))
        if search_invoked is not None:
            metrics["search_false_negative_rate"] = round(
                max(0.0, 1.0 - search_invoked), 12
            )

        by_language = funnel.get("by_language")
        by_language = by_language if isinstance(by_language, Mapping) else {}
        for language in ("zh", "en"):
            item = by_language.get(language)
            item = item if isinstance(item, Mapping) else {}
            group_rates = item.get("stage_hit_rates")
            group_rates = group_rates if isinstance(group_rates, Mapping) else {}
            group: dict[str, float] = {}
            for output_name, source_name in (
                ("domain_recall_at_10", "domain_candidate_hit"),
                ("exact_page_recall_at_20", "exact_raw_candidate_hit"),
                ("final_evidence_exact_recall", "exact_final_evidence_hit"),
                ("citation_exact_page_recall", "exact_citation_hit"),
            ):
                value = _number(group_rates.get(source_name))
                if value is not None:
                    group[output_name] = value
            language_groups[language] = group

    return {
        "schema_version": MEASUREMENT_SCHEMA,
        "benchmark": "RWKV-Agent-S-Level-v1",
        "source": "fitgen_partial_adapter",
        "cases": cases,
        "live_repetitions": live_repetitions,
        "metrics": metrics,
        "language_groups": language_groups,
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
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--measurements", type=Path)
    sources.add_argument("--fitgen-summary", type=Path)
    parser.add_argument("--retrieval-funnel", type=Path)
    parser.add_argument("--live-repetitions", type=int, default=1)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--profile", choices=("production", "s_level"), default="s_level")
    parser.add_argument("--write-measurements", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.measurements:
        measurements = _load_object(args.measurements)
    else:
        summary = _load_object(args.fitgen_summary)
        funnel = _load_object(args.retrieval_funnel) if args.retrieval_funnel else None
        measurements = measurements_from_fitgen(
            summary,
            funnel,
            live_repetitions=args.live_repetitions,
        )
    if args.write_measurements:
        _atomic_write(args.write_measurements, measurements)
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
