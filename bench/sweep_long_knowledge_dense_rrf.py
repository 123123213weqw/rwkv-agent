from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from bench.long_knowledge_hybrid import (
    evaluate_order,
    reciprocal_rank_fusion,
    summarize_rows,
)
from bench.long_knowledge_schema import LongKnowledgeCase, load_cases


def sweep_dense_weights(
    rows: Iterable[Mapping[str, Any]],
    cases: Mapping[str, LongKnowledgeCase],
    weights: Sequence[float],
    *,
    limit: int = 100,
) -> dict[str, Any]:
    records = [dict(row) for row in rows]
    trials = []
    for weight in weights:
        evaluated = []
        for row in records:
            case = cases[str(row["id"])]
            ranking = reciprocal_rank_fusion(
                (
                    list(row.get("lexical_hits") or ()),
                    list(row.get("dense_hits") or ()),
                ),
                weights=(1.0, float(weight)),
                limit=limit,
            )
            evaluated.append(
                {
                    "language": case.language,
                    "query_type": case.query_type,
                    "expectation": case.expectation,
                    "latency_ms": {"dense_rrf": 0.0},
                    "strategies": {
                        "dense_rrf": evaluate_order(
                            case,
                            ranking,
                            index_eligible=bool(row.get("index_eligible")),
                        )
                    },
                }
            )
        overall = summarize_rows(
            evaluated,
            strategies=("dense_rrf",),
        )["strategies"]["dense_rrf"]["overall"]
        trials.append(
            {
                "lexical_weight": 1.0,
                "dense_weight": float(weight),
                "hit_at_1": overall["hit_at_1"],
                "hit_at_10": overall["hit_at_10"],
                "recall_at_10": overall["recall_at_10"],
                "hit_at_100": overall["hit_at_100"],
                "failure_stages": overall["failure_stages"],
            }
        )
    selected = max(
        trials,
        key=lambda trial: (
            float(trial["hit_at_10"] or 0.0),
            float(trial["recall_at_10"] or 0.0),
            float(trial["hit_at_1"] or 0.0),
            -abs(float(trial["dense_weight"]) - 1.0),
        ),
    )
    return {
        "schema_version": "long-knowledge-dense-weight-sweep.v1",
        "selection_scope": "development_only_not_held_out",
        "record_count": len(records),
        "trials": trials,
        "selected": selected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute weighted lexical/dense RRF on a frozen dense run."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--language", default="")
    parser.add_argument("--weights", default="0.5,0.75,1,1.5,2,3")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = {
        case.id: case
        for case in load_cases(args.cases)
        if not args.language or case.language == args.language
    }
    if set(row["id"] for row in rows) != set(cases):
        raise SystemExit("run rows and filtered cases do not contain identical IDs")
    result = sweep_dense_weights(
        rows,
        cases,
        [float(value) for value in args.weights.split(",") if value.strip()],
        limit=args.limit,
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
