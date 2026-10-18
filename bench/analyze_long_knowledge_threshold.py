from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


def maximum_score(
    row: Mapping[str, Any],
    *,
    score_field: str = "semantic_scores",
) -> float:
    values = [float(value) for value in row.get(score_field, ())]
    return max(values) if values else -math.inf


def evaluate_thresholds(
    rows: Iterable[Mapping[str, Any]],
    thresholds: Sequence[float],
    *,
    strategy: str = "semantic",
    score_field: str = "semantic_scores",
) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    positives = [
        row
        for row in records
        if str(row.get("expectation") or "relevant") != "missing"
    ]
    missing = [
        row
        for row in records
        if str(row.get("expectation") or "relevant") == "missing"
    ]
    baseline_hits = sum(
        float(row["strategies"][strategy]["hit_at_10"])
        for row in positives
    )
    results = []
    for threshold in thresholds:
        accepted = [
            row
            for row in positives
            if maximum_score(row, score_field=score_field) >= float(threshold)
        ]
        accepted_hits = sum(
            float(row["strategies"][strategy]["hit_at_10"])
            for row in accepted
        )
        results.append(
            {
                "threshold": float(threshold),
                "positive_accept_rate": len(accepted) / len(positives)
                if positives
                else None,
                "useful_hit_at_10": accepted_hits / len(positives)
                if positives
                else None,
                "hit_at_10_loss": (baseline_hits - accepted_hits) / len(positives)
                if positives
                else None,
                "lost_top10_hits": int(baseline_hits - accepted_hits),
                "missing_rejection_rate": statistics.fmean(
                    maximum_score(row, score_field=score_field) < float(threshold)
                    for row in missing
                )
                if missing
                else None,
            }
        )
    return {
        "schema_version": "long-knowledge-threshold-analysis.v1",
        "strategy": strategy,
        "score_field": score_field,
        "positive_cases": len(positives),
        "expected_missing_cases": len(missing),
        "baseline_hit_at_10": baseline_hits / len(positives)
        if positives
        else None,
        "missing_max_scores": [
            score if math.isfinite(score) else None
            for score in (
                maximum_score(row, score_field=score_field)
                for row in missing
            )
        ],
        "thresholds": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a generic reranker-score admission threshold against recall loss."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", default="semantic")
    parser.add_argument("--score-field", default="semantic_scores")
    parser.add_argument("--thresholds", default="-4,-2,0,2,4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    thresholds = [
        float(value.strip())
        for value in args.thresholds.split(",")
        if value.strip()
    ]
    result = evaluate_thresholds(
        rows,
        thresholds,
        strategy=args.strategy,
        score_field=args.score_field,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
