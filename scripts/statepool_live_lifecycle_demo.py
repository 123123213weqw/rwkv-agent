#!/usr/bin/env python3
"""Drive the live RWKV Worker -> StatePool -> compatible Worker lifecycle."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class DemoError(RuntimeError):
    pass


def request_json(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    data = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            if not body:
                return {"status": "ok", "http_status": response.status}
            value = json.loads(body)
            if not isinstance(value, dict):
                raise DemoError(f"{path} returned non-object JSON")
            return value
    except HTTPError as exc:
        body = exc.read().decode(errors="replace")[:2000]
        raise DemoError(f"{method} {path}: HTTP {exc.code}: {body}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise DemoError(f"{method} {path}: {exc}") from exc


def acquire_lease(
    plugin_url: str,
    *,
    session_id: str,
    owner_id: str,
    holder_id: str,
    expected_version: int,
    ttl_ms: int,
) -> dict[str, Any]:
    return request_json(
        plugin_url,
        "/plugin/v1/leases/acquire",
        payload={
            "contract_version": "statepool-acquire-lease-request.v1",
            "session_id": session_id,
            "owner_id": owner_id,
            "holder_id": holder_id,
            "expected_state_version": expected_version,
            "ttl_ms": ttl_ms,
        },
    )


def release_lease(plugin_url: str, lease: dict[str, Any]) -> None:
    request_json(
        plugin_url,
        "/plugin/v1/leases/release",
        payload={
            "contract_version": "statepool-release-lease-request.v1",
            "lease": lease,
        },
    )


def exact_model_ref(worker_url: str, *, timeout: float = 30.0) -> dict[str, str]:
    health = request_json(worker_url, "/health", timeout=timeout)
    value = health.get("model_ref")
    required = ("model_id", "revision", "tokenizer", "state_abi")
    if not isinstance(value, dict) or any(
        not isinstance(value.get(key), str) or not value[key] for key in required
    ):
        raise DemoError(f"Worker {worker_url} does not report an exact model_ref")
    return {key: value[key] for key in required}


def wait_for_exact_model_ref(
    worker_url: str,
    *,
    timeout_seconds: float,
) -> dict[str, str]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "Worker has not started"
    while time.monotonic() < deadline:
        try:
            return exact_model_ref(worker_url, timeout=2.0)
        except DemoError as exc:
            last_error = str(exc)
            time.sleep(1.0)
    raise DemoError(
        f"Worker {worker_url} was not ready within {timeout_seconds}s: {last_error}"
    )


def wait_for_worker_down(worker_url: str, *, timeout_seconds: float) -> None:
    """Prove an old process no longer owns a reused Worker endpoint."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            exact_model_ref(worker_url, timeout=1.0)
        except DemoError:
            return
        time.sleep(0.25)
    raise DemoError(
        f"Worker {worker_url} was still reachable {timeout_seconds}s after stop"
    )


def terminate_started_process(process: subprocess.Popen[Any]) -> None:
    """Best-effort cleanup when a driver-started replacement cannot be used."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def continue_state(
    worker_url: str,
    *,
    owner_id: str,
    state_id: str,
    text: str,
    max_tokens: int,
) -> dict[str, Any]:
    response = request_json(
        worker_url,
        "/v1/states/batch_continue",
        payload={
            "owner_id": owner_id,
            "items": [{"state_id": state_id, "input": text}],
            "stop": [],
            "max_tokens": max_tokens,
        },
    )
    rows = response.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise DemoError("Worker continuation did not return exactly one result")
    return rows[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    target_start_command = getattr(args, "target_start_command", None)
    target_ready_timeout_seconds = float(
        getattr(args, "target_ready_timeout_seconds", 600.0)
    )
    source_down_timeout_seconds = float(
        getattr(args, "source_down_timeout_seconds", 30.0)
    )
    source_model = exact_model_ref(args.source_worker_url)
    target_model = (
        None
        if target_start_command
        else exact_model_ref(args.target_worker_url)
    )
    if target_model is not None and source_model != target_model:
        raise DemoError("source and target Workers have different exact model_ref values")

    lease = acquire_lease(
        args.plugin_url,
        session_id=args.session_id,
        owner_id=args.owner_id,
        holder_id=f"{args.source_worker_id}/snapshot",
        expected_version=0,
        ttl_ms=args.lease_ttl_ms,
    )
    prefill = request_json(
        args.source_worker_url,
        "/v1/states/prefill",
        payload={
            "owner_id": args.owner_id,
            "prompt": args.prompt,
            "branch": "cloud-lite-demo",
        },
    )
    source_state = prefill.get("state")
    if not isinstance(source_state, dict) or not source_state.get("state_id"):
        raise DemoError("source Worker prefill returned no State")
    source_state_id = str(source_state["state_id"])
    before = continue_state(
        args.source_worker_url,
        owner_id=args.owner_id,
        state_id=source_state_id,
        text=args.before_input,
        max_tokens=args.max_tokens,
    )

    mark = time.perf_counter()
    worker_snapshot = request_json(
        args.source_worker_url,
        f"/v1/states/{source_state_id}/snapshot",
        payload={
            "owner_id": args.owner_id,
            "model_ref": source_model,
            "target_tier": "cpu",
        },
    )
    timings["worker_snapshot_ms"] = (time.perf_counter() - mark) * 1000
    checkpoint = worker_snapshot.get("checkpoint")
    encoded_state = worker_snapshot.get("payload_base64")
    if not isinstance(checkpoint, dict) or not isinstance(encoded_state, str):
        raise DemoError("source Worker returned an invalid snapshot envelope")

    mark = time.perf_counter()
    state_ref = request_json(
        args.plugin_url,
        "/plugin/v1/states/snapshot",
        payload={
            "contract_version": "statepool-snapshot-request.v1",
            "provider_mode": "rwkv_recurrent",
            "model_ref": source_model,
            "target_tier": args.target_tier,
            "lease": lease,
            "expected_state_version": 0,
            "payload_base64": encoded_state,
            "expected_checksum": checkpoint.get("checksum"),
        },
    )
    timings["statepool_commit_ms"] = (time.perf_counter() - mark) * 1000

    source_transition = "released"
    if args.source_stop_command:
        command = shlex.split(args.source_stop_command)
        if not command:
            raise DemoError("source stop command is empty after parsing")
        subprocess.run(command, check=True)
        source_transition = "forcibly_stopped"
    else:
        request_json(
            args.source_worker_url,
            "/v1/states/release",
            payload={"owner_id": args.owner_id, "state_ids": [source_state_id]},
        )
    release_lease(args.plugin_url, lease)

    target_process_pid = None
    if target_start_command:
        if args.source_worker_url.rstrip("/") == args.target_worker_url.rstrip("/"):
            mark = time.perf_counter()
            wait_for_worker_down(
                args.source_worker_url,
                timeout_seconds=source_down_timeout_seconds,
            )
            timings["source_worker_down_ms"] = (time.perf_counter() - mark) * 1000
        command = shlex.split(target_start_command)
        if not command:
            raise DemoError("target start command is empty after parsing")
        mark = time.perf_counter()
        process = subprocess.Popen(command, start_new_session=True)
        target_process_pid = process.pid
        try:
            target_model = wait_for_exact_model_ref(
                args.target_worker_url,
                timeout_seconds=target_ready_timeout_seconds,
            )
        except DemoError as exc:
            exit_code = process.poll()
            detail = f"; target start process exited {exit_code}" if exit_code is not None else ""
            terminate_started_process(process)
            raise DemoError(f"{exc}{detail}") from exc
        timings["target_worker_ready_ms"] = (time.perf_counter() - mark) * 1000
        if source_model != target_model:
            terminate_started_process(process)
            raise DemoError(
                "source and post-restart target Workers have different exact model_ref values"
            )
    assert target_model is not None

    restore_lease = acquire_lease(
        args.plugin_url,
        session_id=args.session_id,
        owner_id=args.owner_id,
        holder_id=f"{args.target_worker_id}/restore",
        expected_version=int(state_ref["version"]),
        ttl_ms=args.lease_ttl_ms,
    )
    mark = time.perf_counter()
    stored = request_json(
        args.plugin_url,
        "/plugin/v1/states/restore",
        payload={
            "contract_version": "statepool-restore-request.v1",
            "state_ref": state_ref,
            "expected_model_ref": target_model,
            "target_worker_id": args.target_worker_id,
            "lease": restore_lease,
        },
    )
    timings["statepool_read_ms"] = (time.perf_counter() - mark) * 1000
    mark = time.perf_counter()
    restored = request_json(
        args.target_worker_url,
        "/v1/states/restore",
        payload={
            "owner_id": args.owner_id,
            "model_ref": target_model,
            "checksum": state_ref["checksum"],
            "payload_base64": stored["payload_base64"],
            "branch": "cloud-lite-restored",
        },
    )
    timings["worker_restore_ms"] = (time.perf_counter() - mark) * 1000
    target_state = restored.get("state")
    if not isinstance(target_state, dict) or not target_state.get("state_id"):
        raise DemoError("target Worker restore returned no State")
    target_state_id = str(target_state["state_id"])
    after = continue_state(
        args.target_worker_url,
        owner_id=args.owner_id,
        state_id=target_state_id,
        text=args.after_input,
        max_tokens=args.max_tokens,
    )
    request_json(
        args.target_worker_url,
        "/v1/states/release",
        payload={"owner_id": args.owner_id, "state_ids": [target_state_id]},
    )
    release_lease(args.plugin_url, restore_lease)
    timings["total_ms"] = (time.perf_counter() - started) * 1000
    return {
        "schema_version": "rwkv-statepool-live-lifecycle-result.v1",
        "status": "passed",
        "model_ref": source_model,
        "session_id": args.session_id,
        "owner_id": args.owner_id,
        "source_worker_id": args.source_worker_id,
        "target_worker_id": args.target_worker_id,
        "target_worker_started_by_driver": bool(target_start_command),
        "target_worker_process_pid": target_process_pid,
        "source_transition": source_transition,
        "source_state_id": source_state_id,
        "target_state_id": target_state_id,
        "state_ref": state_ref,
        "before": before,
        "after": after,
        "timings_ms": {key: round(value, 3) for key, value in timings.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-url", default="http://127.0.0.1:8130")
    parser.add_argument("--source-worker-url", required=True)
    parser.add_argument("--target-worker-url", required=True)
    parser.add_argument("--source-worker-id", default="worker-source")
    parser.add_argument("--target-worker-id", default="worker-target")
    parser.add_argument("--session-id", default="statepool-live-demo")
    parser.add_argument("--owner-id", default="demo:statepool-live")
    parser.add_argument("--prompt", default="System: Preserve this exact recurrent context.")
    parser.add_argument("--before-input", default="\nUser: remember marker ALPHA\nAssistant:")
    parser.add_argument("--after-input", default="\nUser: what marker did I give you?\nAssistant:")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--lease-ttl-ms", type=int, default=120_000)
    parser.add_argument("--target-tier", choices=("warm", "cold"), default="cold")
    parser.add_argument(
        "--source-stop-command",
        help="explicit command run after commit to prove forced Worker loss",
    )
    parser.add_argument(
        "--target-start-command",
        help=(
            "optional command spawned after source loss; use it to start a fresh "
            "compatible Worker when source and target share one GPU/endpoint"
        ),
    )
    parser.add_argument(
        "--target-ready-timeout-seconds",
        type=float,
        default=600.0,
    )
    parser.add_argument(
        "--source-down-timeout-seconds",
        type=float,
        default=30.0,
        help="time allowed to prove a stopped source released its reused endpoint",
    )
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    if (
        args.max_tokens < 1
        or args.lease_ttl_ms < 1000
        or args.target_ready_timeout_seconds <= 0
        or args.source_down_timeout_seconds <= 0
    ):
        parser.error("token, Lease and target-ready limits must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except (DemoError, subprocess.CalledProcessError) as exc:
        print(f"StatePool live lifecycle failed: {exc}", file=sys.stderr)
        return 1
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(encoded)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
