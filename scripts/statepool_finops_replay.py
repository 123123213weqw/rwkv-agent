#!/usr/bin/env python3
"""Replay an explicit 100-Session A/B/C StatePool capacity model.

This is an analytical simulation fed by archived measured primitives. It never
labels derived GPU-hours, cost, or token savings as live benchmark results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
KEDA_SUMMARY = (
    ROOT / "bench" / "artifacts" / "statepool-keda-kind-20260827" / "verified-summary.json"
)
GPU_RESULT = (
    ROOT
    / "bench"
    / "artifacts"
    / "statepool-4080-worker-kill-20260827"
    / "worker"
    / "result.json"
)
CONTRACT_ROOT = ROOT / "bench" / "artifacts" / "long-lived-phase0-rust-contract-v1"


@dataclass(frozen=True)
class ReplayConfig:
    sessions: int = 100
    sessions_per_worker: int = 8
    active_seconds_per_window: float = 60.0
    idle_seconds_between_windows: float = 300.0
    windows: int = 2
    second_window_history_tokens_per_session: int = 512
    gpu_price_cny_per_hour: float = 2.0

    def validate(self) -> None:
        if self.sessions <= 0 or self.sessions_per_worker <= 0:
            raise ValueError("sessions and sessions_per_worker must be positive")
        if self.windows < 2:
            raise ValueError("windows must be at least two")
        if self.active_seconds_per_window <= 0 or self.idle_seconds_between_windows < 0:
            raise ValueError("active/idle durations are invalid")
        if self.second_window_history_tokens_per_session < 0:
            raise ValueError("history token count must not be negative")
        if self.gpu_price_cny_per_hour < 0:
            raise ValueError("GPU price must not be negative")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_identity() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    return commit, dirty


def contract_result(policy: str) -> tuple[Path, dict[str, Any]]:
    path = CONTRACT_ROOT / f"dev-n100-{policy}.json"
    value = load_json(path)
    correctness = value.get("correctness", {})
    lifecycle = value.get("lifecycle", {})
    if (
        correctness.get("cross_talk") != 0
        or correctness.get("failed_events") != 0
        or lifecycle.get("state_leak_count") != 0
    ):
        raise ValueError(f"contract evidence failed for {policy}")
    return path, value


def build_replay(config: ReplayConfig) -> dict[str, Any]:
    config.validate()
    keda = load_json(KEDA_SUMMARY)
    gpu = load_json(GPU_RESULT)
    if not (keda.get("success") and keda.get("all_prestop_safe")):
        raise ValueError("KEDA evidence is not a verified passing run")
    if gpu.get("status") != "passed":
        raise ValueError("GPU lifecycle evidence is not a passing run")

    contract_sources: dict[str, dict[str, Any]] = {}
    for scenario, policy in {
        "sticky_worker": "keep-hot",
        "stateless_reprefill": "drop-reprefill",
        "statepool": "move-cpu",
    }.items():
        path, value = contract_result(policy)
        contract_sources[scenario] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "policy": value["placement_policy"],
            "events": value["correctness"]["events"],
            "cross_talk": value["correctness"]["cross_talk"],
            "failed_events": value["correctness"]["failed_events"],
            "state_leak_count": value["lifecycle"]["state_leak_count"],
        }

    timings = keda["timings_seconds"]
    ready_seconds = float(timings["ready_replicas_3"])
    shutdown_seconds = float(timings["last_worker_pod_gone"]) - float(
        timings["plugin_pending_cleared"]
    )
    if ready_seconds <= 0 or shutdown_seconds <= 0:
        raise ValueError("invalid KEDA timing primitives")

    workers = math.ceil(config.sessions / config.sessions_per_worker)
    active_seconds = config.windows * config.active_seconds_per_window
    sticky_seconds = workers * (
        active_seconds + (config.windows - 1) * config.idle_seconds_between_windows
    )
    elastic_seconds = workers * config.windows * (
        ready_seconds + config.active_seconds_per_window + shutdown_seconds
    )
    reprefill_tokens = (
        config.sessions
        * (config.windows - 1)
        * config.second_window_history_tokens_per_session
    )
    state_size = int(gpu["state_ref"]["size_bytes"])
    snapshots = config.sessions * config.windows
    restores = config.sessions * (config.windows - 1)

    def row(
        scenario: str,
        label: str,
        allocated_seconds: float,
        idle_minutes: float,
        repeated_prefill: int,
        avoided_prefill: int,
        state_bytes_written: int,
        state_bytes_read: int,
        hit_rate: float,
    ) -> dict[str, Any]:
        gpu_hours = allocated_seconds / 3600.0
        return {
            "scenario": scenario,
            "label": label,
            "classification": "simulation_replay",
            "sessions": config.sessions,
            "workers_at_peak": workers,
            "allocated_gpu_seconds": round(allocated_seconds, 6),
            "gpu_hours_per_100_sessions": round(gpu_hours, 6),
            "modeled_idle_gpu_minutes": round(idle_minutes, 6),
            "repeated_prefill_tokens": repeated_prefill,
            "prefill_tokens_avoided_vs_stateless": avoided_prefill,
            "state_bytes_written": state_bytes_written,
            "state_bytes_read": state_bytes_read,
            "eligible_continue_state_hit_rate": hit_rate,
            "estimated_gpu_cost_cny": round(
                gpu_hours * config.gpu_price_cny_per_hour, 6
            ),
        }

    rows = [
        row(
            "A",
            "sticky_worker",
            sticky_seconds,
            workers * (config.windows - 1) * config.idle_seconds_between_windows / 60,
            0,
            reprefill_tokens,
            0,
            0,
            1.0,
        ),
        row(
            "B",
            "stateless_reprefill",
            elastic_seconds,
            workers * config.windows * shutdown_seconds / 60,
            reprefill_tokens,
            0,
            0,
            0,
            0.0,
        ),
        row(
            "C",
            "statepool",
            elastic_seconds,
            workers * config.windows * shutdown_seconds / 60,
            0,
            reprefill_tokens,
            snapshots * state_size,
            restores * state_size,
            1.0,
        ),
    ]
    sticky_hours = rows[0]["gpu_hours_per_100_sessions"]
    statepool_hours = rows[2]["gpu_hours_per_100_sessions"]

    return {
        "schema_version": "statepool-finops-replay.v1",
        "classification": "simulation_replay_from_measured_primitives",
        "config": asdict(config),
        "measured_inputs": {
            "keda": {
                "path": KEDA_SUMMARY.relative_to(ROOT).as_posix(),
                "sha256": sha256(KEDA_SUMMARY),
                "environment": "kind 0.30.0 / Kubernetes 1.34.0 / KEDA 2.20.1 / simulated Worker",
                "scale_from_demand_to_three_ready_seconds": ready_seconds,
                "pending_clear_to_last_pod_gone_seconds": round(shutdown_seconds, 6),
            },
            "gpu_state": {
                "path": GPU_RESULT.relative_to(ROOT).as_posix(),
                "sha256": sha256(GPU_RESULT),
                "environment": "one RTX 4080; exact-compatible sequential Worker processes",
                "state_size_bytes": state_size,
                "worker_snapshot_ms_single_sample": gpu["timings_ms"]["worker_snapshot_ms"],
                "worker_restore_ms_single_sample": gpu["timings_ms"]["worker_restore_ms"],
                "fresh_worker_ready_ms_warm_single_sample": gpu["timings_ms"]["target_worker_ready_ms"],
            },
            "contract_correctness": contract_sources,
        },
        "derived_results": rows,
        "comparisons": {
            "statepool_gpu_hour_reduction_vs_sticky_percent": round(
                (1.0 - statepool_hours / sticky_hours) * 100.0, 3
            ),
            "statepool_estimated_gpu_cost_reduction_vs_sticky_percent": round(
                (1.0 - rows[2]["estimated_gpu_cost_cny"] / rows[0]["estimated_gpu_cost_cny"])
                * 100.0,
                3,
            ),
            "statepool_prefill_tokens_avoided_vs_stateless": reprefill_tokens,
            "statepool_total_state_transfer_bytes": snapshots * state_size
            + restores * state_size,
        },
        "not_measured": [
            "GPU average utilization for these three scenarios",
            "production restore P50/P95",
            "prefill GPU seconds or throughput penalty in scenario B",
            "storage, request and egress charges",
            "real GPU Kubernetes Pod scaling",
        ],
        "interpretation_limits": [
            "Derived GPU-hours are a capacity-allocation model, not a cloud bill.",
            "B and C have equal modeled allocation because the unknown B re-prefill compute penalty is deliberately omitted.",
            "The KEDA timing comes from a non-GPU kind Worker simulation.",
            "The State size and local snapshot/restore timings are single RTX 4080 measurements.",
            "The 2 CNY/GPU-hour input is an explicit scenario parameter, not a market price claim.",
        ],
    }


def write_bundle(bundle: dict[str, Any], output_dir: Path) -> None:
    commit, dirty = git_identity()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = bundle["derived_results"]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "statepool-finops-replay-manifest.v1",
        "classification": bundle["classification"],
        "git_commit": commit,
        "git_worktree_dirty_at_start": dirty,
        "generator": {
            "path": "scripts/statepool_finops_replay.py",
            "sha256": sha256(Path(__file__)),
        },
        "source_files": [
            {
                "path": value["path"],
                "sha256": value["sha256"],
            }
            for value in (
                bundle["measured_inputs"]["keda"],
                bundle["measured_inputs"]["gpu_state"],
                *bundle["measured_inputs"]["contract_correctness"].values(),
            )
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=100)
    parser.add_argument("--sessions-per-worker", type=int, default=8)
    parser.add_argument("--active-seconds", type=float, default=60.0)
    parser.add_argument("--idle-seconds", type=float, default=300.0)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--history-tokens", type=int, default=512)
    parser.add_argument("--gpu-price-cny-per-hour", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ReplayConfig(
        sessions=args.sessions,
        sessions_per_worker=args.sessions_per_worker,
        active_seconds_per_window=args.active_seconds,
        idle_seconds_between_windows=args.idle_seconds,
        windows=args.windows,
        second_window_history_tokens_per_session=args.history_tokens,
        gpu_price_cny_per_hour=args.gpu_price_cny_per_hour,
    )
    bundle = build_replay(config)
    write_bundle(bundle, args.output_dir)
    print(json.dumps(bundle["comparisons"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
