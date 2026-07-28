from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.agent_benchmark_metrics import (
    aggregate_evaluations,
    compare_evaluations,
    evaluate_agent_case,
)
from benchmarks.agent_benchmark_schema import load_jsonl


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluate(
    cases: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cases_by_id = {str(case["id"]): case for case in cases}
    results_by_id = {str(result["case_id"]): result for result in results}
    missing = sorted(set(cases_by_id) - set(results_by_id))
    extra = sorted(set(results_by_id) - set(cases_by_id))
    if missing or extra:
        raise ValueError(
            f"cases/results must have identical IDs; missing={missing}, extra={extra}"
        )
    return [
        evaluate_agent_case(cases_by_id[case_id], results_by_id[case_id])
        for case_id in sorted(cases_by_id)
    ]


def build_report(
    *,
    cases_path: str | Path,
    results_path: str | Path,
    baseline_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases = load_jsonl(cases_path, kind="case")
    results = load_jsonl(results_path, kind="result")
    evaluations = _evaluate(cases, results)
    report: dict[str, Any] = {
        "schema_version": "rwkv-agent-benchmark-report.v1",
        "inputs": {
            "cases": {
                "path": str(cases_path),
                "sha256": _sha256(cases_path),
                "rows": len(cases),
            },
            "results": {
                "path": str(results_path),
                "sha256": _sha256(results_path),
                "rows": len(results),
            },
        },
        "coverage": {
            "expected": len(cases),
            "evaluated": len(evaluations),
            "rate": 1.0,
        },
        "summary": aggregate_evaluations(evaluations),
    }
    if baseline_path is not None:
        baseline_results = load_jsonl(baseline_path, kind="result")
        baseline_evaluations = _evaluate(cases, baseline_results)
        report["inputs"]["baseline"] = {
            "path": str(baseline_path),
            "sha256": _sha256(baseline_path),
            "rows": len(baseline_results),
        }
        report["comparison"] = compare_evaluations(
            baseline_evaluations,
            evaluations,
        )
    return report, evaluations


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate normalized RWKV Agent benchmark result JSONL files."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows-output")
    args = parser.parse_args(argv)
    report, evaluations = build_report(
        cases_path=args.cases,
        results_path=args.results,
        baseline_path=args.baseline,
    )
    _write_json(args.output, report)
    if args.rows_output:
        _write_jsonl(args.rows_output, evaluations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
