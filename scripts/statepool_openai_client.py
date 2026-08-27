#!/usr/bin/env python3
"""Plan through StatePool, then call the selected OpenAI-compatible Worker."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any
from urllib.request import Request, urlopen
import uuid


def post_json(url: str, payload: dict[str, Any], api_key: str = "") -> tuple[dict, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        value = json.loads(response.read())
        return value, response.headers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statepool-url", default="http://127.0.0.1:8130")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--state-abi", default="context-replay.v1")
    parser.add_argument("--session-id", default="openai-demo")
    parser.add_argument("--owner-id", default="demo-owner")
    parser.add_argument("--affinity-worker-id", default="")
    parser.add_argument("--prompt", default="Reply with exactly: statepool-ok")
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    plan_request = {
        "contract_version": "statepool-plan-request.v1",
        "request_id": f"openai-{uuid.uuid4().hex}",
        "session_id": args.session_id,
        "owner_id": args.owner_id,
        "model_ref": {
            "model_id": args.model_id,
            "revision": args.model_revision,
            "tokenizer": args.tokenizer,
            "state_abi": args.state_abi,
        },
        "privacy": "cloud_allowed",
        "latency_slo_ms": 30_000,
        "estimated_input_tokens": max(1, len(args.prompt) // 3),
        "estimated_output_tokens": 32,
    }
    if args.affinity_worker_id:
        plan_request["affinity_worker_id"] = args.affinity_worker_id

    started = time.perf_counter()
    plan, _ = post_json(
        args.statepool_url.rstrip("/") + "/plugin/v1/plan",
        plan_request,
    )
    endpoint = plan.get("endpoint")
    if plan.get("mode") not in {"local", "remote"} or not endpoint:
        raise SystemExit(f"StatePool did not select a Worker: {json.dumps(plan)}")

    completion, headers = post_json(
        endpoint.rstrip("/") + "/v1/chat/completions",
        {
            "model": args.model_id,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0,
            "max_tokens": 32,
        },
        args.api_key,
    )
    result = {
        "plan": plan,
        "selected_worker_id": headers.get("X-StatePool-Worker-Id"),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "completion": completion,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
