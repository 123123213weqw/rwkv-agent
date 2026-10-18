#!/usr/bin/env python3
"""Frozen Gate 3 benchmark for independent persistent RWKV States.

Each logical job owns a distinct prompt, owner ID and recurrent State.  The
scaling arm submits jobs simultaneously; a serial arm replays the same jobs
for an exact greedy A/B.  Raw rows, Scheduler shapes and ROCm telemetry are
retained as auditable artifacts.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import threading
import time
from typing import Any, Iterable

from benchmarks.run_chat_state_throughput_ab import (
    CHAT_STOPS,
    DIRECT_SYSTEM_PROMPT,
    HttpSidecar,
    SidecarTransport,
    summarize_shape_delta,
)

SCHEMA = "rwkv_agent_gate3_state_scaling.v1"
DEFAULT_LEVELS = (1, 4, 8, 16, 32, 64, 100)


@dataclass(frozen=True, slots=True)
class StateTask:
    index: int
    owner_id: str
    marker: str
    prompt: str
    continuation: str


@dataclass(frozen=True, slots=True)
class StateBinding:
    task: StateTask
    state_id: str
    seen_tokens: int
    setup_ms: float


def percentile(values: Iterable[float], quantile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    return rows[max(0, math.ceil(quantile * len(rows)) - 1)]


def build_tasks(count: int) -> list[StateTask]:
    if not 1 <= int(count) <= 100:
        raise ValueError("task count must be 1..100")
    tasks = []
    for index in range(1, int(count) + 1):
        marker = f"RAVEN-{index:03d}"
        owner = "gate3-" + hashlib.sha256(
            f"independent-owner-{index:03d}".encode()
        ).hexdigest()[:24]
        prompt = (
            DIRECT_SYSTEM_PROMPT
            + "This is one isolated benchmark session. Its private reference "
            + f"code is {marker}. Never use a code from another session."
        )
        continuation = (
            "\n\nUser: Return this session's private reference code only."
            "\n\nAssistant:"
        )
        tasks.append(StateTask(index, owner, marker, prompt, continuation))
    return tasks


def _rocm_sample() -> dict[str, Any]:
    completed = subprocess.run(
        ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--csv"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = list(csv.DictReader(io.StringIO(completed.stdout.strip())))
    if not rows:
        raise RuntimeError("rocm-smi returned no GPU rows")
    row = rows[0]
    return {
        "gpu_busy_pct": float(row["GPU use (%)"]),
        "vram_total_bytes": int(row["VRAM Total Memory (B)"]),
        "vram_used_bytes": int(row["VRAM Total Used Memory (B)"]),
    }


class TelemetrySampler:
    def __init__(
        self,
        transport: SidecarTransport,
        *,
        interval_seconds: float,
    ) -> None:
        self.transport = transport
        self.interval_seconds = max(0.05, float(interval_seconds))
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._phase = "idle"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = time.time()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry sampler already started")
        self._thread = threading.Thread(
            target=self._run,
            name="gate3-rocm-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.time()
            with self._lock:
                phase = self._phase
            sample: dict[str, Any] = {
                "timestamp_unix": round(started, 6),
                "elapsed_seconds": round(started - self._started, 6),
                "phase": phase,
            }
            try:
                sample.update(_rocm_sample())
                health = self.transport.get("/health")
                inference = dict(health.get("inference") or {})
                persistent = dict(health.get("persistent_states") or {})
                batching = dict(persistent.get("batching") or {})
                sample.update(
                    {
                        "resident_states": int(persistent.get("allocated") or 0),
                        "waiting_jobs": int(inference.get("waiting") or 0),
                        "prefilling_rows": int(inference.get("prefilling") or 0),
                        "decoding_rows": int(inference.get("decoding") or 0),
                        "active_state_rows": int(batching.get("active_rows") or 0),
                    }
                )
                self.samples.append(sample)
            except Exception as exc:  # telemetry cannot invalidate inference
                self.errors.append(f"{type(exc).__name__}: {exc}"[:300])
            self._stop.wait(
                max(0.0, self.interval_seconds - (time.time() - started))
            )


def telemetry_summary(
    samples: list[dict[str, Any]],
    *,
    phase: str,
) -> dict[str, Any]:
    selected = [row for row in samples if row.get("phase") == phase]
    busy = [float(row["gpu_busy_pct"]) for row in selected]
    vram = [int(row["vram_used_bytes"]) for row in selected]
    return {
        "phase": phase,
        "samples": len(selected),
        "gpu_busy_pct": {
            "mean": round(statistics.fmean(busy), 3) if busy else 0.0,
            "p50": round(percentile(busy, 0.50), 3),
            "p95": round(percentile(busy, 0.95), 3),
            "peak": round(max(busy), 3) if busy else 0.0,
        },
        "vram_used_bytes": {
            "mean": round(statistics.fmean(vram)) if vram else 0,
            "peak": max(vram) if vram else 0,
        },
        "resident_states_peak": max(
            (int(row.get("resident_states") or 0) for row in selected), default=0
        ),
        "active_state_rows_peak": max(
            (int(row.get("active_state_rows") or 0) for row in selected), default=0
        ),
        "decoding_rows_peak": max(
            (int(row.get("decoding_rows") or 0) for row in selected), default=0
        ),
        "waiting_jobs_peak": max(
            (int(row.get("waiting_jobs") or 0) for row in selected), default=0
        ),
    }


def allocate_states(
    transport: SidecarTransport,
    tasks: list[StateTask],
) -> list[StateBinding]:
    bindings = []
    for task in tasks:
        started = time.perf_counter()
        response = transport.post(
            "/v1/states/prefill",
            {
                "owner_id": task.owner_id,
                "prompt": task.prompt,
                "branch": f"gate3-{task.index:03d}",
            },
        )
        state = dict(response["state"])
        bindings.append(
            StateBinding(
                task=task,
                state_id=str(state["state_id"]),
                seen_tokens=int(state.get("seen_tokens") or 0),
                setup_ms=(time.perf_counter() - started) * 1000.0,
            )
        )
    return bindings


def release_states(
    transport: SidecarTransport,
    bindings: list[StateBinding],
) -> tuple[int, list[str]]:
    released = 0
    errors = []
    for binding in bindings:
        try:
            response = transport.post(
                "/v1/states/release",
                {
                    "owner_id": binding.task.owner_id,
                    "state_ids": [binding.state_id],
                },
            )
            released += int(response.get("released") or 0)
        except Exception as exc:
            errors.append(
                f"{binding.task.index}:{type(exc).__name__}:{exc}"[:300]
            )
    return released, errors


def _advance_one(
    transport: SidecarTransport,
    binding: StateBinding,
    *,
    max_tokens: int,
    barrier: threading.Barrier | None,
) -> dict[str, Any]:
    if barrier is not None:
        barrier.wait(timeout=30)
    started = time.perf_counter()
    try:
        response = transport.post(
            "/v1/states/batch_continue",
            {
                "owner_id": binding.task.owner_id,
                "items": [
                    {
                        "state_id": binding.state_id,
                        "input": binding.task.continuation,
                    }
                ],
                "stop": list(CHAT_STOPS),
                "max_tokens": int(max_tokens),
            },
        )
        result = dict(response["results"][0])
        return {
            "task": binding.task.index,
            "owner_id": binding.task.owner_id,
            "marker": binding.task.marker,
            "state_id": binding.state_id,
            "status": "ok",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "queue_ms": float(result.get("queue_ms") or 0.0),
            "model_elapsed_ms": float(result.get("elapsed_ms") or 0.0),
            "text": str(result.get("text") or ""),
            "token_ids": [int(value) for value in result.get("token_ids") or []],
            "stop_reason": str(result.get("stop_reason") or ""),
            "seen_tokens": int(result.get("seen_tokens") or 0),
        }
    except Exception as exc:
        return {
            "task": binding.task.index,
            "owner_id": binding.task.owner_id,
            "marker": binding.task.marker,
            "state_id": binding.state_id,
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "error": f"{type(exc).__name__}: {exc}"[:500],
            "text": "",
            "token_ids": [],
        }


def execute_rows(
    transport: SidecarTransport,
    bindings: list[StateBinding],
    *,
    mode: str,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    if mode == "serial":
        rows = [
            _advance_one(
                transport, binding, max_tokens=max_tokens, barrier=None
            )
            for binding in bindings
        ]
    elif mode == "concurrent":
        barrier = threading.Barrier(len(bindings))

        def run(binding: StateBinding) -> dict[str, Any]:
            return _advance_one(
                transport, binding, max_tokens=max_tokens, barrier=barrier
            )

        with ThreadPoolExecutor(max_workers=len(bindings)) as executor:
            rows = list(executor.map(run, bindings))
    else:
        raise ValueError("mode must be serial or concurrent")
    return rows, time.perf_counter() - started


def _physical_batch(shape_summary: dict[str, Any]) -> int:
    observed = 0
    for key, calls in dict(shape_summary.get("shape_counts") or {}).items():
        if int(calls) < 1 or not str(key).startswith("B") or "T" not in str(key):
            continue
        try:
            observed = max(observed, int(str(key)[1:].split("T", 1)[0]))
        except ValueError:
            continue
    return observed


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    wall_seconds: float,
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok"]
    latencies = [float(row["latency_ms"]) for row in successful]
    queues = [float(row.get("queue_ms") or 0.0) for row in successful]
    output_tokens = sum(len(row.get("token_ids") or []) for row in successful)
    per_state_tps = [
        len(row.get("token_ids") or []) / (float(row["latency_ms"]) / 1000.0)
        for row in successful
        if float(row["latency_ms"]) > 0
    ]
    return {
        "requests": len(rows),
        "successes": len(successful),
        "failures": len(rows) - len(successful),
        "wall_seconds": round(float(wall_seconds), 6),
        "output_tokens": output_tokens,
        "aggregate_output_tokens_per_second": round(
            output_tokens / wall_seconds if wall_seconds > 0 else 0.0, 4
        ),
        "per_state_output_tokens_per_second": {
            "mean": round(statistics.fmean(per_state_tps), 4)
            if per_state_tps
            else 0.0,
            "p50": round(percentile(per_state_tps, 0.50), 4),
            "p95": round(percentile(per_state_tps, 0.95), 4),
        },
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "queue_ms": {
            "mean": round(statistics.fmean(queues), 3) if queues else 0.0,
            "p50": round(percentile(queues, 0.50), 3),
            "p95": round(percentile(queues, 0.95), 3),
            "max": round(max(queues), 3) if queues else 0.0,
        },
        "ttft": {
            "available": False,
            "reason": "Sidecar State HTTP endpoint is non-streaming",
        },
    }


def validate_rows(rows: list[dict[str, Any]], tasks: list[StateTask]) -> list[str]:
    errors = []
    if len(rows) != len(tasks):
        errors.append("request_count_mismatch")
    if any(row.get("status") != "ok" for row in rows):
        errors.append("request_failure")
    state_ids = [str(row.get("state_id") or "") for row in rows]
    if len(set(state_ids)) != len(state_ids) or any(not value for value in state_ids):
        errors.append("state_identity_failure")
    known = {task.marker for task in tasks}
    for row in rows:
        foreign = known - {str(row.get("marker") or "")}
        if any(marker in str(row.get("text") or "") for marker in foreign):
            errors.append(f"session_mix_task_{row.get('task')}")
    return errors


def compare_arms(serial: dict[str, Any], concurrent: dict[str, Any]) -> dict[str, Any]:
    left = {int(row["task"]): row for row in serial["rows"]}
    right = {int(row["task"]): row for row in concurrent["rows"]}
    keys = sorted(set(left) | set(right))
    mismatches = [
        key
        for key in keys
        if key not in left
        or key not in right
        or left[key].get("token_ids") != right[key].get("token_ids")
        or left[key].get("text") != right[key].get("text")
    ]
    serial_wall = float(serial["metrics"]["wall_seconds"])
    concurrent_wall = float(concurrent["metrics"]["wall_seconds"])
    serial_tps = float(
        serial["metrics"]["aggregate_output_tokens_per_second"]
    )
    concurrent_tps = float(
        concurrent["metrics"]["aggregate_output_tokens_per_second"]
    )
    return {
        "compared": len(keys),
        "exact": len(keys) - len(mismatches),
        "all_exact": bool(keys) and not mismatches,
        "mismatches": mismatches,
        "concurrent_speedup": round(
            serial_wall / concurrent_wall if concurrent_wall > 0 else 0.0, 4
        ),
        "aggregate_tps_ratio": round(
            concurrent_tps / serial_tps if serial_tps > 0 else 0.0, 4
        ),
    }


def run_arm(
    transport: SidecarTransport,
    sampler: TelemetrySampler,
    tasks: list[StateTask],
    *,
    mode: str,
    max_tokens: int,
    label: str,
) -> dict[str, Any]:
    initial = transport.get("/health")
    bindings: list[StateBinding] = []
    rows: list[dict[str, Any]] = []
    before = initial
    after = initial
    resident = initial
    setup_seconds = 0.0
    wall_seconds = 0.0
    sampler.set_phase(label + "-setup")
    setup_started = time.perf_counter()
    try:
        bindings = allocate_states(transport, tasks)
        setup_seconds = time.perf_counter() - setup_started
        resident = transport.get("/health")
        before = transport.get("/health")
        sampler.set_phase(label + "-decode")
        rows, wall_seconds = execute_rows(
            transport, bindings, mode=mode, max_tokens=max_tokens
        )
        after = transport.get("/health")
    finally:
        sampler.set_phase(label + "-release")
        release_started = time.perf_counter()
        released, release_errors = release_states(transport, bindings)
        release_seconds = time.perf_counter() - release_started
        final = transport.get("/health")
        sampler.set_phase("idle")

    shape = summarize_shape_delta(before, after)
    errors = validate_rows(rows, tasks)
    if released != len(bindings):
        errors.append("release_mismatch")
    if release_errors:
        errors.append("release_error")
    initial_allocated = int(
        dict(initial.get("persistent_states") or {}).get("allocated") or 0
    )
    final_allocated = int(
        dict(final.get("persistent_states") or {}).get("allocated") or 0
    )
    if final_allocated != initial_allocated:
        errors.append("state_leak")
    return {
        "label": label,
        "mode": mode,
        "concurrency": len(tasks),
        "max_tokens": max_tokens,
        "setup_seconds": round(setup_seconds, 6),
        "release_seconds": round(release_seconds, 6),
        "resident_states": int(
            dict(resident.get("persistent_states") or {}).get("allocated") or 0
        ),
        "released_states": released,
        "release_errors": release_errors,
        "initial_allocated": initial_allocated,
        "final_allocated": final_allocated,
        "metrics": summarize_rows(rows, wall_seconds=wall_seconds),
        "scheduler": {**shape, "physical_batch_observed": _physical_batch(shape)},
        "telemetry_phase": label + "-decode",
        "errors": sorted(set(errors)),
        "rows": rows,
    }


def write_telemetry_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    columns = [
        "timestamp_unix",
        "elapsed_seconds",
        "phase",
        "gpu_busy_pct",
        "vram_total_bytes",
        "vram_used_bytes",
        "resident_states",
        "waiting_jobs",
        "prefilling_rows",
        "decoding_rows",
        "active_state_rows",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: sample.get(key, "") for key in columns})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--levels", default=",".join(str(value) for value in DEFAULT_LEVELS)
    )
    parser.add_argument("--ab-level", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--telemetry-interval", type=float, default=0.2)
    args = parser.parse_args()
    levels = tuple(
        int(value.strip()) for value in args.levels.split(",") if value.strip()
    )
    if not levels or any(value < 1 or value > 100 for value in levels):
        parser.error("--levels values must be 1..100")
    if len(set(levels)) != len(levels):
        parser.error("--levels values must be unique")
    if args.ab_level not in levels:
        parser.error("--ab-level must be included in --levels")
    if not 1 <= args.max_tokens <= 64:
        parser.error("--max-tokens must be 1..64")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transport = HttpSidecar(args.endpoint, timeout=args.timeout)
    initial = transport.get("/health")
    persistent = dict(initial.get("persistent_states") or {})
    if int(persistent.get("allocated") or 0):
        raise RuntimeError("persistent state pool must be empty before Gate 3")
    if int(persistent.get("capacity") or 0) < max(levels):
        raise RuntimeError(
            f"persistent capacity {persistent.get('capacity')} is below "
            f"required {max(levels)}"
        )

    tasks = build_tasks(max(levels))
    sampler = TelemetrySampler(
        transport, interval_seconds=args.telemetry_interval
    )
    sampler.start()
    scaling = []
    serial_arm: dict[str, Any] | None = None
    try:
        for level in levels:
            arm = run_arm(
                transport,
                sampler,
                tasks[:level],
                mode="concurrent",
                max_tokens=args.max_tokens,
                label=f"scale-b{level}",
            )
            scaling.append(arm)
            print(
                json.dumps(
                    {
                        "level": level,
                        "wall_seconds": arm["metrics"]["wall_seconds"],
                        "aggregate_tps": arm["metrics"][
                            "aggregate_output_tokens_per_second"
                        ],
                        "physical_batch": arm["scheduler"][
                            "physical_batch_observed"
                        ],
                        "errors": arm["errors"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        serial_arm = run_arm(
            transport,
            sampler,
            tasks[: args.ab_level],
            mode="serial",
            max_tokens=args.max_tokens,
            label=f"serial-b{args.ab_level}",
        )
    finally:
        sampler.stop()

    for arm in [*scaling, serial_arm]:
        if arm is not None:
            arm["telemetry"] = telemetry_summary(
                sampler.samples, phase=str(arm["telemetry_phase"])
            )
    concurrent_ab = next(
        arm for arm in scaling if int(arm["concurrency"]) == args.ab_level
    )
    if serial_arm is None:
        raise RuntimeError("serial reference did not run")
    ab = compare_arms(serial_arm, concurrent_ab)
    final = transport.get("/health")
    final_allocated = int(
        dict(final.get("persistent_states") or {}).get("allocated") or 0
    )
    hard_errors = [
        {"label": arm["label"], "errors": arm["errors"]}
        for arm in [*scaling, serial_arm]
        if arm["errors"]
    ]
    if not ab["all_exact"]:
        hard_errors.append(
            {"label": "serial_concurrent_ab", "errors": ["output_mismatch"]}
        )
    if final_allocated:
        hard_errors.append({"label": "final", "errors": ["state_leak"]})

    raw_rows = output_dir / "gate3-state-scaling-rows.jsonl"
    with raw_rows.open("w") as stream:
        for arm in [*scaling, serial_arm]:
            for row in arm["rows"]:
                stream.write(
                    json.dumps(
                        {"arm": arm["label"], **row},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    telemetry_path = output_dir / "gate3-state-scaling-rocm.csv"
    write_telemetry_csv(telemetry_path, sampler.samples)
    summary = {
        "schema": SCHEMA,
        "status": "pass" if not hard_errors else "fail",
        "endpoint": args.endpoint,
        "model": initial.get("model"),
        "backend": initial.get("backend"),
        "context": initial.get("context"),
        "protocol": {
            "levels": list(levels),
            "ab_level": args.ab_level,
            "max_tokens": args.max_tokens,
            "greedy": True,
            "independent_owners": True,
            "independent_states": True,
            "shared_root_fork": False,
            "ttft_available": False,
        },
        "hashes": {
            "input_sha256": hashlib.sha256(
                json.dumps(
                    [
                        {
                            "index": task.index,
                            "owner_id": task.owner_id,
                            "marker": task.marker,
                            "prompt": task.prompt,
                            "continuation": task.continuation,
                        }
                        for task in tasks
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "raw_rows_sha256": hashlib.sha256(raw_rows.read_bytes()).hexdigest(),
            "telemetry_sha256": hashlib.sha256(telemetry_path.read_bytes()).hexdigest(),
        },
        "initial_health": initial,
        "scaling": scaling,
        "serial_reference": serial_arm,
        "serial_concurrent_ab": ab,
        "telemetry_errors": sampler.errors,
        "final_health": final,
        "final_persistent_allocated": final_allocated,
        "hard_errors": hard_errors,
        "artifacts": {
            "raw_rows": str(raw_rows),
            "telemetry_csv": str(telemetry_path),
        },
    }
    summary_path = output_dir / "gate3-state-scaling-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(summary_path),
                "ab": ab,
                "final_persistent_allocated": final_allocated,
            },
            ensure_ascii=False,
        )
    )
    if hard_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
