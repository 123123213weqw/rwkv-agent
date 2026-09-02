"""Kubernetes preStop client for the Sidecar StatePool drain gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            value = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"Worker HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Worker drain request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Worker drain response is not a JSON object")
    return value


def drain(worker_url: str, deadline_seconds: float, poll_seconds: float) -> dict[str, Any]:
    if deadline_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("drain deadline and poll interval must be positive")
    base = worker_url.rstrip("/")
    deadline = time.monotonic() + deadline_seconds
    status = _request(
        base + "/v1/statepool/drain",
        {"timeout_seconds": deadline_seconds},
    )
    while status.get("status") != "safe_to_stop":
        if status.get("status") == "deadline_exceeded" or time.monotonic() >= deadline:
            raise TimeoutError(
                "Worker drain deadline expired with "
                f"active_requests={status.get('active_requests')} "
                f"unpersisted_states={status.get('unpersisted_states')}"
            )
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
        status = _request(base + "/v1/statepool/drain")
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker-url",
        default=os.getenv("RWKV_WORKER_SELF_URL", "http://127.0.0.1:8118"),
    )
    parser.add_argument("--deadline-seconds", type=float, default=120)
    parser.add_argument("--poll-seconds", type=float, default=1)
    args = parser.parse_args()
    try:
        status = drain(args.worker_url, args.deadline_seconds, args.poll_seconds)
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(f"StatePool Worker drain failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
