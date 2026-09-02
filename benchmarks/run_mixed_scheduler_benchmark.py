#!/usr/bin/env python3
"""Frozen mixed-traffic benchmark for the RWKV Sidecar scheduler.

The workload deliberately overlaps ordinary completions, one-token
classifications, single-State chat continuations, and two-row Agent-State
continuations.  It talks only to an already-running isolated Sidecar and never
starts, stops, or reconfigures a service.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Protocol
from urllib.request import Request, urlopen
import uuid

SCHEMA = "rwkv_agent_mixed_scheduler_benchmark.v1"
DEFAULT_CONCURRENCY = (8, 16)
DEFAULT_MAX_TOKENS = 8
WORKLOAD_BLOCK = (
    "state_chat",
    "completion",
    "gate",
    "state_chat",
    "completion",
    "gate",
    "state_chat",
    "state_branch",
)
GATE_LABELS = {"tool": "search", "chat": "chat"}
GATE_SCORE_ATOL = 0.125
CHAT_STOPS = ("\n\nUser:", "\nUser:", "\nSystem:", "</s>")
DIRECT_SYSTEM_PROMPT = (
    "System: You are a helpful conversational assistant. Answer the user "
    "directly in the user's language. Do not claim to have searched, do not "
    "invent sources or citation IDs, and do not emit a tool call. Never output "
    "<think> tags or hidden reasoning.\n\n"
)


class SidecarTransport(Protocol):
    endpoint: str

    def get(self, path: str) -> dict[str, Any]: ...

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class HttpSidecar:
    def __init__(self, endpoint: str, *, timeout: float = 300.0) -> None:
        self.endpoint = str(endpoint or "").rstrip("/")
        if not self.endpoint:
            raise ValueError("endpoint must not be empty")
        self.timeout = float(timeout)

    def get(self, path: str) -> dict[str, Any]:
        with urlopen(self.endpoint + path, timeout=self.timeout) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise RuntimeError(f"{path} returned a non-object")
        return value

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.endpoint + path,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise RuntimeError(f"{path} returned a non-object")
        return value


def _shape_counts(health: dict[str, Any]) -> dict[str, int]:
    scheduler = dict(health.get("inference") or {}).get("scheduler", {})
    return {
        str(key): int(value)
        for key, value in dict(dict(scheduler).get("shape_counts") or {}).items()
    }


def summarize_shape_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    left = _shape_counts(before)
    right = _shape_counts(after)
    delta = {
        key: max(0, right.get(key, 0) - left.get(key, 0))
        for key in sorted(set(left) | set(right))
        if right.get(key, 0) - left.get(key, 0) > 0
    }
    decoded: list[tuple[int, int, int]] = []
    for key, calls in delta.items():
        if not key.startswith("B") or "T" not in key:
            continue
        batch_text, token_text = key[1:].split("T", 1)
        try:
            decoded.append((int(batch_text), int(token_text), int(calls)))
        except ValueError:
            continue

    def fill(*, token_length: int | None) -> float:
        rows = [
            (batch, calls)
            for batch, tokens, calls in decoded
            if (tokens == token_length if token_length is not None else tokens > 1)
        ]
        calls = sum(count for _batch, count in rows)
        return (
            sum(batch * count for batch, count in rows) / calls if calls else 0.0
        )

    return {
        "shape_counts": delta,
        "decode_average_batch_fill": round(fill(token_length=1), 4),
        "prefill_average_batch_fill": round(fill(token_length=None), 4),
        "model_forward_calls": sum(delta.values()),
    }


@dataclass(frozen=True, slots=True)
class WorkItem:
    item_id: str
    kind: str
    state_rows: int
    prompt: str
    state_input: str


@dataclass(slots=True)
class StateBinding:
    owner_id: str
    state_ids: list[str]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[rank]


def build_work_items(concurrency: int) -> list[WorkItem]:
    value = int(concurrency)
    if value < 1 or value % len(WORKLOAD_BLOCK):
        raise ValueError(
            f"concurrency must be a positive multiple of {len(WORKLOAD_BLOCK)}"
        )
    output = []
    for index in range(value):
        kind = WORKLOAD_BLOCK[index % len(WORKLOAD_BLOCK)]
        note = (
            f"Scheduler case {index + 1}. RWKV recurrent state preserves token "
            "order and lets independent requests share exact-shape GPU batches. "
            "A fair serving loop must bound queueing while retaining greedy "
            "token equivalence. "
        ) * 4
        prompt = (
            DIRECT_SYSTEM_PROMPT
            + f"User: Read the technical note and answer in one sentence:\n{note}"
            + "\n\nAssistant:"
        )
        output.append(
            WorkItem(
                item_id=f"mixed-{value:02d}-{index + 1:02d}",
                kind=kind,
                state_rows=2 if kind == "state_branch" else (
                    1 if kind == "state_chat" else 0
                ),
                prompt=prompt,
                state_input=(
                    f"User: Read the technical note and answer in one sentence:\n"
                    f"{note}\n\nAssistant:"
                ),
            )
        )
    return output


def required_state_capacity(items: list[WorkItem]) -> int:
    return sum(item.state_rows for item in items)


def _owner_id(item_id: str) -> str:
    return "mixed-" + hashlib.sha256(item_id.encode()).hexdigest()[:24]


def allocate_states(
    transport: SidecarTransport,
    items: list[WorkItem],
) -> dict[str, StateBinding]:
    bindings: dict[str, StateBinding] = {}
    for item in items:
        if not item.state_rows:
            continue
        owner_id = _owner_id(item.item_id)
        state_ids = []
        for row in range(item.state_rows):
            response = transport.post(
                "/v1/states/prefill",
                {
                    "owner_id": owner_id,
                    "prompt": (
                        DIRECT_SYSTEM_PROMPT
                        + f"System: Active branch {row + 1}.\n\n"
                    ),
                    "branch": f"{item.kind}-{row + 1}",
                },
            )
            state_ids.append(str(response["state"]["state_id"]))
        bindings[item.item_id] = StateBinding(
            owner_id=owner_id,
            state_ids=state_ids,
        )
    return bindings


def release_states(
    transport: SidecarTransport,
    bindings: dict[str, StateBinding],
) -> int:
    released = 0
    for binding in bindings.values():
        response = transport.post(
            "/v1/states/release",
            {
                "owner_id": binding.owner_id,
                "state_ids": binding.state_ids,
            },
        )
        released += int(response.get("released") or 0)
    return released


def _fingerprint(kind: str, response: dict[str, Any]) -> dict[str, Any]:
    if kind == "completion":
        result = dict(response["g1i"])
        return {
            "text": str(result.get("text") or ""),
            "token_ids": [int(value) for value in result.get("token_ids") or []],
            "stop_reason": str(result.get("stop_reason") or ""),
        }
    if kind == "gate":
        scores = {
            str(key): float(value)
            for key, value in sorted(dict(response["scores"]).items())
        }
        return {
            "label": max(scores, key=scores.__getitem__),
            "scores": scores,
        }
    return {
        "rows": [
            {
                "branch": str(row.get("branch") or ""),
                "text": str(row.get("text") or ""),
                "token_ids": [
                    int(value) for value in row.get("token_ids") or []
                ],
                "stop_reason": str(row.get("stop_reason") or ""),
            }
            for row in response["results"]
        ]
    }


def execute_item(
    transport: SidecarTransport,
    item: WorkItem,
    bindings: dict[str, StateBinding],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    if item.kind == "completion":
        response = transport.post(
            "/v1/completions",
            {
                "prompt": item.prompt,
                "stop": list(CHAT_STOPS),
                "max_tokens": int(max_tokens),
            },
        )
    elif item.kind == "gate":
        response = transport.post(
            "/v1/classify",
            {
                "prompt": (
                    "System: Select the next action.\n\n"
                    + item.prompt
                    + "\n\nAction:"
                ),
                "labels": GATE_LABELS,
            },
        )
    else:
        binding = bindings[item.item_id]
        response = transport.post(
            "/v1/states/batch_continue",
            {
                "owner_id": binding.owner_id,
                "items": [
                    {
                        "state_id": state_id,
                        "input": item.state_input,
                    }
                    for state_id in binding.state_ids
                ],
                "stop": list(CHAT_STOPS),
                "max_tokens": int(max_tokens),
            },
        )
    return {
        "item_id": item.item_id,
        "kind": item.kind,
        "state_rows": item.state_rows,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "fingerprint": _fingerprint(item.kind, response),
    }


def run_isolated_reference(
    transport: SidecarTransport,
    items: list[WorkItem],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    bindings = allocate_states(transport, items)
    started = time.perf_counter()
    try:
        rows = [
            execute_item(
                transport,
                item,
                bindings,
                max_tokens=max_tokens,
            )
            for item in items
        ]
    finally:
        released = release_states(transport, bindings)
    return summarize_rows(
        rows,
        wall_seconds=time.perf_counter() - started,
        released_states=released,
    )


def run_mixed(
    transport: SidecarTransport,
    items: list[WorkItem],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    bindings = allocate_states(transport, items)
    barrier = threading.Barrier(len(items))

    def run(item: WorkItem) -> dict[str, Any]:
        barrier.wait()
        return execute_item(
            transport,
            item,
            bindings,
            max_tokens=max_tokens,
        )

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=len(items)) as executor:
            rows = list(executor.map(run, items))
    finally:
        released = release_states(transport, bindings)
    return summarize_rows(
        rows,
        wall_seconds=time.perf_counter() - started,
        released_states=released,
    )


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    wall_seconds: float,
    released_states: int,
) -> dict[str, Any]:
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({str(row["kind"]) for row in rows}):
        selected = [row for row in rows if row["kind"] == kind]
        latencies = [float(row["latency_ms"]) for row in selected]
        by_kind[kind] = {
            "requests": len(selected),
            "state_rows": sum(int(row["state_rows"]) for row in selected),
            "latency_ms": {
                "mean": round(sum(latencies) / len(latencies), 3),
                "p50": round(percentile(latencies, 0.50), 3),
                "p95": round(percentile(latencies, 0.95), 3),
                "max": round(max(latencies), 3),
            },
        }
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "requests": len(rows),
        "state_rows": sum(int(row["state_rows"]) for row in rows),
        "wall_seconds": round(float(wall_seconds), 6),
        "requests_per_second": round(
            len(rows) / wall_seconds if wall_seconds > 0 else 0.0,
            4,
        ),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3),
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3),
        },
        "by_kind": by_kind,
        "released_states": int(released_states),
        "rows": rows,
    }


def compare_reference(
    reference: dict[str, Any],
    mixed: dict[str, Any],
) -> dict[str, Any]:
    left = {
        str(row["item_id"]): row["fingerprint"]
        for row in reference["rows"]
    }
    right = {
        str(row["item_id"]): row["fingerprint"]
        for row in mixed["rows"]
    }
    keys = sorted(set(left) | set(right))
    mismatches = []
    gate_label_mismatches = []
    gate_score_max_abs_delta = 0.0
    for key in keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if (
            isinstance(left_value, dict)
            and isinstance(right_value, dict)
            and "label" in left_value
            and "label" in right_value
        ):
            if left_value["label"] != right_value["label"]:
                gate_label_mismatches.append(key)
                mismatches.append(key)
                continue
            left_scores = dict(left_value.get("scores") or {})
            right_scores = dict(right_value.get("scores") or {})
            if set(left_scores) != set(right_scores):
                mismatches.append(key)
                continue
            delta = max(
                (
                    abs(float(left_scores[name]) - float(right_scores[name]))
                    for name in left_scores
                ),
                default=0.0,
            )
            gate_score_max_abs_delta = max(gate_score_max_abs_delta, delta)
            if delta > GATE_SCORE_ATOL:
                mismatches.append(key)
        elif left_value != right_value:
            mismatches.append(key)
    slowdowns = {}
    for kind, mixed_kind in mixed["by_kind"].items():
        baseline_kind = reference["by_kind"][kind]
        baseline_p95 = float(baseline_kind["latency_ms"]["p95"])
        mixed_p95 = float(mixed_kind["latency_ms"]["p95"])
        slowdowns[kind] = round(
            mixed_p95 / baseline_p95 if baseline_p95 > 0 else 0.0,
            4,
        )
    return {
        "compared": len(keys),
        "exact": len(keys) - len(mismatches),
        "all_exact": bool(keys) and not mismatches,
        "mismatches": mismatches,
        "gate_label_mismatches": gate_label_mismatches,
        "gate_score_max_abs_delta": round(gate_score_max_abs_delta, 6),
        "gate_score_atol": GATE_SCORE_ATOL,
        "mixed_p95_over_isolated": slowdowns,
    }


def _metric_counter(health: dict[str, Any], path: tuple[str, ...]) -> dict[str, int]:
    value: Any = health
    for key in path:
        value = dict(value or {}).get(key)
    return {
        str(key): int(number)
        for key, number in dict(value or {}).items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def counter_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    path: tuple[str, ...],
) -> dict[str, int]:
    left = _metric_counter(before, path)
    right = _metric_counter(after, path)
    return {
        key: right.get(key, 0) - left.get(key, 0)
        for key in sorted(set(left) | set(right))
        if right.get(key, 0) != left.get(key, 0)
    }


def validate_profile(profile: dict[str, Any]) -> list[str]:
    errors = []
    if not profile["comparison"]["all_exact"]:
        errors.append("output_mismatch")
    if int(profile["state_leak_count"]):
        errors.append("state_leak")
    if int(profile["mixed"]["released_states"]) != int(
        profile["required_state_capacity"]
    ):
        errors.append("release_mismatch")
    if float(profile["mixed"]["latency_ms"]["max"]) > 20_000:
        errors.append("starvation_budget_exceeded")
    inference = profile["queue_metric_delta"]["inference"]
    state = profile["queue_metric_delta"]["state"]
    if any(
        int(inference.get(name, 0)) or int(state.get(name, 0))
        for name in (
            "rejected_queue_full",
            "request_timeouts",
            "failed",
            "failed_jobs",
        )
    ):
        errors.append("queue_failure")
    return errors


def run_profile(
    transport: SidecarTransport,
    *,
    concurrency: int,
    max_tokens: int,
) -> dict[str, Any]:
    items = build_work_items(concurrency)
    required = required_state_capacity(items)
    initial = transport.get("/health")
    reference = run_isolated_reference(
        transport,
        items,
        max_tokens=max_tokens,
    )
    before = transport.get("/health")
    mixed = run_mixed(
        transport,
        items,
        max_tokens=max_tokens,
    )
    after = transport.get("/health")
    comparison = compare_reference(reference, mixed)
    profile = {
        "concurrency": int(concurrency),
        "max_tokens": int(max_tokens),
        "required_state_capacity": required,
        "reference": reference,
        "mixed": mixed,
        "comparison": comparison,
        "scheduler": summarize_shape_delta(before, after),
        "queue_metric_delta": {
            "inference": counter_delta(
                before,
                after,
                ("inference", "metrics"),
            ),
            "state": counter_delta(
                before,
                after,
                ("persistent_states", "batching", "metrics"),
            ),
        },
        "state_leak_count": max(
            0,
            int(dict(after.get("persistent_states") or {}).get("allocated") or 0)
            - int(dict(initial.get("persistent_states") or {}).get("allocated") or 0),
        ),
    }
    profile["errors"] = validate_profile(profile)
    return profile


def require_capacity(
    health: dict[str, Any],
    concurrencies: tuple[int, ...],
) -> int:
    persistent = dict(health.get("persistent_states") or {})
    capacity = int(persistent.get("capacity") or 0)
    required = max(
        required_state_capacity(build_work_items(value))
        for value in concurrencies
    )
    if int(persistent.get("allocated") or 0):
        raise RuntimeError("persistent state pool must be empty before benchmark")
    if capacity < required:
        raise RuntimeError(
            f"persistent state capacity {capacity} is below required {required}"
        )
    return capacity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--concurrency",
        default=",".join(str(value) for value in DEFAULT_CONCURRENCY),
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    concurrency = tuple(
        int(value.strip())
        for value in args.concurrency.split(",")
        if value.strip()
    )
    if not concurrency:
        parser.error("--concurrency must not be empty")
    try:
        for value in concurrency:
            build_work_items(value)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.max_tokens <= 64:
        parser.error("--max-tokens must be 1..64")

    transport = HttpSidecar(args.endpoint, timeout=args.timeout)
    initial = transport.get("/health")
    capacity = require_capacity(initial, concurrency)
    profiles = []
    for value in concurrency:
        profile = run_profile(
            transport,
            concurrency=value,
            max_tokens=args.max_tokens,
        )
        profiles.append(profile)
        print(
            json.dumps(
                {
                    "concurrency": value,
                    "requests_per_second": profile["mixed"][
                        "requests_per_second"
                    ],
                    "all_exact": profile["comparison"]["all_exact"],
                    "state_leak_count": profile["state_leak_count"],
                    "errors": profile["errors"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    final = transport.get("/health")
    result = {
        "schema": SCHEMA,
        "run_id": "mixed-scheduler-" + uuid.uuid4().hex,
        "endpoint": transport.endpoint,
        "model": initial.get("model"),
        "context": initial.get("context"),
        "persistent_capacity": capacity,
        "protocol": {
            "concurrency": list(concurrency),
            "max_tokens": args.max_tokens,
            "greedy": True,
            "workload_block": list(WORKLOAD_BLOCK),
            "state_branch_rows": 2,
            "gate_score_atol": GATE_SCORE_ATOL,
            "starvation_budget_ms": 20_000,
            "batch_fill_targets": {"8": 3.0, "16": 5.0},
            "comparison": "sequential isolated reference vs simultaneous mixed",
            "prompt_set_sha256": hashlib.sha256(
                json.dumps(
                    [
                        (
                            item.item_id,
                            item.kind,
                            item.state_rows,
                            item.prompt,
                            item.state_input,
                        )
                        for value in concurrency
                        for item in build_work_items(value)
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "profiles": profiles,
        "initial_persistent_allocated": int(
            dict(initial.get("persistent_states") or {}).get("allocated") or 0
        ),
        "final_persistent_allocated": int(
            dict(final.get("persistent_states") or {}).get("allocated") or 0
        ),
        "all_profiles_valid": all(not profile["errors"] for profile in profiles),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "profiles": len(profiles),
                "all_profiles_valid": result["all_profiles_valid"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    if not result["all_profiles_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
