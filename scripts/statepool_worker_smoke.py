#!/usr/bin/env python3
"""Exercise Worker registration, heartbeat and sticky drain against a plugin."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any
from urllib.request import Request, urlopen

from rwkv_agent.statepool_worker import StatePoolWorkerAgent, WorkerSettings


def request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urlopen(request, timeout=5) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} returned non-object JSON")
    return value


def run(plugin_url: str) -> dict[str, Any]:
    load = {"allocated": 1, "busy": 0}

    def health() -> dict[str, Any]:
        allocated = load["allocated"]
        return {
            "inference": {"waiting": 0, "prefilling": 0, "decoding": 0},
            "persistent_states": {
                "capacity": 4,
                "allocated": allocated,
                "free": 4 - allocated,
                "busy": load["busy"],
                "batching": {
                    "waiting": 0,
                    "active_rows": load["busy"],
                    "max_batch_size": 4,
                },
            },
        }

    settings = WorkerSettings(
        statepool_url=plugin_url.rstrip("/"),
        worker_id="worker-smoke",
        endpoint="http://worker-smoke.invalid:8118",
        zone="cloud",
        model_ref={
            "model_id": "rwkv-smoke",
            "revision": "immutable-smoke-revision",
            "tokenizer": "rwkv-smoke-tokenizer",
            "state_abi": "rwkv-smoke-state-v1",
        },
        state_slots=4,
        max_batch=4,
        heartbeat_seconds=60,
    )
    agent = StatePoolWorkerAgent(settings, health)
    agent.start()
    registration_deadline = time.monotonic() + 5
    while not agent.status()["registered"]:
        if time.monotonic() >= registration_deadline:
            raise RuntimeError(f"Worker did not register: {agent.status()}")
        time.sleep(0.01)
    registered = request_json(plugin_url, "/plugin/v1/workers")
    workers = registered.get("workers")
    if not isinstance(workers, list) or len(workers) != 1:
        raise RuntimeError("Worker registration was not visible")
    draining = agent.begin_draining(timeout_seconds=30)
    if draining.get("status") != "draining" or agent.enter_request():
        raise RuntimeError("dirty Worker drain did not close admission")

    load["allocated"] = 0
    agent.heartbeat_once()
    safe = request_json(
        plugin_url,
        "/plugin/v1/workers/worker-smoke/drain",
        {
            "contract_version": "statepool-drain-request.v1",
            "deadline_ms": int(time.time() * 1000) + 30_000,
        },
    )
    if safe.get("status") != "safe_to_stop":
        raise RuntimeError(f"clean Worker did not become safe: {safe}")
    result = {
        "schema_version": "rwkv-statepool-worker-smoke.v1",
        "status": "passed",
        "registered_lifecycle": workers[0].get("lifecycle"),
        "dirty_drain_status": draining.get("status"),
        "clean_drain_status": safe.get("status"),
        "new_admission_after_drain": False,
    }
    agent.stop()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-url", default="http://127.0.0.1:8130")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    result = run(args.plugin_url)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        print(encoded, end="")
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
