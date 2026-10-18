#!/usr/bin/env python3
"""Evaluate the semantic chat/tool gate through a real G1I Sidecar."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time
from typing import Any
from urllib.request import Request, urlopen


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))
    return ordered[index]


def call_gate(url: str, case: dict[str, Any], threshold: float) -> dict[str, Any]:
    payload = json.dumps(
        {
            "message": case["message"],
            "context": case.get("context", ""),
            "has_pasted_text": bool(case.get("has_pasted_text", False)),
            "threshold": threshold,
        },
        ensure_ascii=False,
    ).encode()
    request = Request(
        url.rstrip("/") + "/v1/gate/tool",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
    result["wall_elapsed_ms"] = round(
        (time.perf_counter() - started) * 1000.0,
        3,
    )
    return result


def summarize(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    by_group: dict[str, dict[str, int]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    for group, values in sorted(grouped.items()):
        by_group[group] = {
            "total": len(values),
            "correct": sum(bool(row["correct"]) for row in values),
        }
    chat = [row for row in rows if row["expected_label"] == "chat"]
    tool = [row for row in rows if row["expected_label"] == "tool"]
    latencies = [float(row["gate"]["elapsed_ms"]) for row in rows]
    return {
        "cases": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": round(sum(bool(row["correct"]) for row in rows) / len(rows), 4),
        "false_tool_rate": round(
            sum(row["selected_label"] == "tool" for row in chat) / len(chat),
            4,
        ),
        "missed_tool_rate": round(
            sum(row["selected_label"] == "chat" for row in tool) / len(tool),
            4,
        ),
        "mean_gate_ms": round(statistics.fmean(latencies), 3),
        "p95_gate_ms": round(percentile(latencies, 0.95), 3),
        "threshold": threshold,
        "by_group": by_group,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8417")
    parser.add_argument(
        "--dataset",
        default="benchmarks/semantic_tool_gate_v1.jsonl",
    )
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    for case in cases:
        gate = call_gate(args.url, case, args.threshold)
        selected = str(gate["label"])
        rows.append(
            {
                **case,
                "selected_label": selected,
                "correct": selected == case["expected_label"],
                "gate": gate,
            }
        )
    output = {
        "schema": "rwkv-agent-semantic-tool-gate-v1",
        "summary": summarize(rows, args.threshold),
        "rows": rows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
