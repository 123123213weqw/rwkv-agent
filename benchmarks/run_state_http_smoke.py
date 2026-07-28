#!/usr/bin/env python3
"""Real Sidecar smoke for persistent Root/Fork/Resume Agent states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

from rwkv_agent.controller import parse_tool_call
from rwkv_agent.state_agent import (
    BRANCH_MISSIONS,
    render_branch_step,
    render_root_final_input,
    render_root_prompt,
    reconstruct_tool_call,
)


def post(
    endpoint: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float = 300.0,
) -> dict[str, Any]:
    request = Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} returned a non-object")
    return value


def get(endpoint: str, path: str) -> dict[str, Any]:
    with urlopen(endpoint.rstrip("/") + path, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} returned a non-object")
    return value


def completion(
    endpoint: str,
    prompt: str,
    *,
    stops: list[str],
    max_tokens: int,
) -> dict[str, Any]:
    value = post(
        endpoint,
        "/v1/completions",
        {"prompt": prompt, "stop": stops, "max_tokens": max_tokens},
    )
    g1i = value["g1i"]
    return {
        "text": str(g1i.get("text") or ""),
        "stop_reason": str(g1i.get("stop_reason") or ""),
        "token_ids": [int(token) for token in g1i.get("token_ids") or []],
    }


def state_text(result: dict[str, Any]) -> str:
    text = str(result.get("text") or "")
    stop = str(result.get("stop_reason") or "")
    if stop and stop not in {"max_tokens", "</s>"}:
        text += stop
    return text


def exact_output(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left.get(key) == right.get(key)
        for key in ("text", "stop_reason", "token_ids")
    )


def release(
    endpoint: str,
    owner_id: str,
    state_ids: list[str],
) -> dict[str, Any]:
    if not state_ids:
        return {"status": "ok", "released": 0}
    return post(
        endpoint,
        "/v1/states/release",
        {"owner_id": owner_id, "state_ids": state_ids},
    )


def run_equivalence(endpoint: str) -> dict[str, Any]:
    owner = "smoke-equivalence-" + uuid.uuid4().hex
    root_prompt = (
        "System: Continue the supplied branch exactly and answer briefly. Do not "
        "call tools.\n\nUser: Shared state prefix for exact continuation testing."
    )
    suffixes = [
        f"\n\nUser: Branch {index}: reply with a short distinct sentence.\n\nAssistant:"
        for index in range(1, 5)
    ]
    state_ids: list[str] = []
    started = time.perf_counter()
    try:
        root = post(
            endpoint,
            "/v1/states/prefill",
            {"owner_id": owner, "prompt": root_prompt, "branch": "root"},
        )["state"]
        root_id = str(root["state_id"])
        state_ids.append(root_id)
        branches = post(
            endpoint,
            f"/v1/states/{root_id}/fork",
            {
                "owner_id": owner,
                "branches": [f"equivalence-{index}" for index in range(1, 5)],
            },
        )["states"]
        state_ids.extend(str(branch["state_id"]) for branch in branches)
        first = post(
            endpoint,
            "/v1/states/batch_continue",
            {
                "owner_id": owner,
                "items": [
                    {"state_id": branch["state_id"], "input": suffix}
                    for branch, suffix in zip(branches, suffixes, strict=True)
                ],
                "stop": ["\n\nUser:"],
                "max_tokens": 16,
            },
        )["results"]
        first_reference = [
            completion(
                endpoint,
                root_prompt + suffix,
                stops=["\n\nUser:"],
                max_tokens=16,
            )
            for suffix in suffixes
        ]
        first_exact = [
            exact_output(actual, reference)
            for actual, reference in zip(first, first_reference, strict=True)
        ]

        observations = [
            (
                "\n\nUser: Observation "
                f"{index}: now reply with one different short sentence."
                "\n\nAssistant:"
            )
            for index in range(1, 5)
        ]
        second = post(
            endpoint,
            "/v1/states/batch_continue",
            {
                "owner_id": owner,
                "items": [
                    {"state_id": branch["state_id"], "input": observation}
                    for branch, observation in zip(
                        branches,
                        observations,
                        strict=True,
                    )
                ],
                "stop": ["\n\nUser:"],
                "max_tokens": 16,
            },
        )["results"]
        second_reference = [
            completion(
                endpoint,
                root_prompt
                + suffix
                + state_text(first_result)
                + observation,
                stops=["\n\nUser:"],
                max_tokens=16,
            )
            for suffix, first_result, observation in zip(
                suffixes,
                first,
                observations,
                strict=True,
            )
        ]
        second_exact = [
            exact_output(actual, reference)
            for actual, reference in zip(second, second_reference, strict=True)
        ]
        return {
            "first_exact": first_exact,
            "second_exact": second_exact,
            "passed": all(first_exact) and all(second_exact),
            "first_token_sha256": hashlib.sha256(
                json.dumps(
                    [item["token_ids"] for item in first],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "second_token_sha256": hashlib.sha256(
                json.dumps(
                    [item["token_ids"] for item in second],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "elapsed_s": round(time.perf_counter() - started, 6),
        }
    finally:
        release(endpoint, owner, state_ids)


def run_owner_isolation(endpoint: str) -> dict[str, Any]:
    owner = "smoke-owner-" + uuid.uuid4().hex
    state_ids: list[str] = []
    try:
        root = post(
            endpoint,
            "/v1/states/prefill",
            {"owner_id": owner, "prompt": "User: hello\n\nAssistant:"},
        )["state"]
        root_id = str(root["state_id"])
        state_ids.append(root_id)
        status = 0
        try:
            post(
                endpoint,
                f"/v1/states/{root_id}/fork",
                {"owner_id": owner + "-other", "branches": ["forbidden"]},
            )
        except HTTPError as exc:
            status = int(exc.code)
        return {"http_status": status, "passed": status == 403}
    finally:
        release(endpoint, owner, state_ids)


def parse_state_tool(result: dict[str, Any]) -> dict[str, Any]:
    raw = reconstruct_tool_call(result)
    parsed = parse_tool_call(raw)
    return {"raw": raw, "parsed": parsed}


def run_two_round_protocol(endpoint: str, question: str) -> dict[str, Any]:
    owner = "smoke-protocol-" + uuid.uuid4().hex
    state_ids: list[str] = []
    started = time.perf_counter()
    try:
        root = post(
            endpoint,
            "/v1/states/prefill",
            {
                "owner_id": owner,
                "prompt": render_root_prompt(question),
                "branch": "root",
            },
        )["state"]
        root_id = str(root["state_id"])
        state_ids.append(root_id)
        branches = post(
            endpoint,
            f"/v1/states/{root_id}/fork",
            {
                "owner_id": owner,
                "branches": [f"protocol-{index}" for index in range(1, 5)],
            },
        )["states"]
        state_ids.extend(str(branch["state_id"]) for branch in branches)

        first = post(
            endpoint,
            "/v1/states/batch_continue",
            {
                "owner_id": owner,
                "items": [
                    {
                        "state_id": branch["state_id"],
                        "input": render_branch_step(
                            question=question,
                            mission=BRANCH_MISSIONS[index],
                            round_index=1,
                            observation=None,
                        ),
                    }
                    for index, branch in enumerate(branches)
                ],
                "stop": ["</tool_call>"],
                "max_tokens": 96,
            },
        )["results"]
        first_parsed = [parse_state_tool(item) for item in first]
        first_strict = [
            bool(item["parsed"].get("strict"))
            and item["parsed"].get("tool") == "web_search"
            for item in first_parsed
        ]

        observations = [
            {
                "status": "ok",
                "evidence": [
                    {
                        "id": "W1",
                        "title": f"Evidence for branch {index}",
                        "content": (
                            "The first result is partial and must be verified "
                            "with a different focused query."
                        ),
                        "uri": f"https://example.invalid/branch-{index}",
                    }
                ],
            }
            for index in range(1, 5)
        ]
        second = post(
            endpoint,
            "/v1/states/batch_continue",
            {
                "owner_id": owner,
                "items": [
                    {
                        "state_id": branch["state_id"],
                        "input": render_branch_step(
                            question=question,
                            mission=BRANCH_MISSIONS[index],
                            round_index=2,
                            observation=observations[index],
                        ),
                    }
                    for index, branch in enumerate(branches)
                ],
                "stop": ["</tool_call>"],
                "max_tokens": 96,
            },
        )["results"]
        second_parsed = [parse_state_tool(item) for item in second]
        second_strict = [
            bool(item["parsed"].get("strict"))
            and item["parsed"].get("tool") == "web_search"
            for item in second_parsed
        ]
        first_queries = [
            item["parsed"].get("arguments", {}).get("query", "")
            for item in first_parsed
        ]
        second_queries = [
            item["parsed"].get("arguments", {}).get("query", "")
            for item in second_parsed
        ]
        final_evidence = [
            {
                "id": f"W{index}",
                "title": f"Verified source {index}",
                "content": (
                    "RWKV is an open source recurrent neural network project; "
                    "this synthetic evidence checks only the final Root resume "
                    "protocol, not factual answer quality."
                ),
                "uri": f"https://example.invalid/final-{index}",
            }
            for index in range(1, 5)
        ]
        final = post(
            endpoint,
            "/v1/states/batch_continue",
            {
                "owner_id": owner,
                "items": [
                    {
                        "state_id": root_id,
                        "input": render_root_final_input(
                            question,
                            final_evidence,
                        ),
                    }
                ],
                "stop": ["\n\nUser:", "\nSystem:"],
                "max_tokens": 96,
            },
        )["results"][0]
        final_text = str(final.get("text") or "").strip()
        final_valid = bool(final_text) and "<tool_call>" not in final_text
        return {
            "first_strict": first_strict,
            "second_strict": second_strict,
            "first_raw": [item["raw"] for item in first_parsed],
            "second_raw": [item["raw"] for item in second_parsed],
            "first_queries": first_queries,
            "second_queries": second_queries,
            "query_changed": [
                first_query != second_query
                for first_query, second_query in zip(
                    first_queries,
                    second_queries,
                    strict=True,
                )
            ],
            "final_valid": final_valid,
            "final_text": final_text,
            "passed": (
                all(first_strict)
                and all(second_strict)
                and final_valid
            ),
            "elapsed_s": round(time.perf_counter() - started, 6),
        }
    finally:
        release(endpoint, owner, state_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8218")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    before = get(args.endpoint, "/health")
    equivalence = run_equivalence(args.endpoint)
    isolation = run_owner_isolation(args.endpoint)
    protocols = {
        "en": run_two_round_protocol(
            args.endpoint,
            "Who created RWKV and what organization maintains the project?",
        ),
        "zh": run_two_round_protocol(
            args.endpoint,
            "搜索 RWKV 是由谁创建、由哪个组织维护的，请给出来源。",
        ),
    }
    after = get(args.endpoint, "/health")
    persistent = after.get("persistent_states") or {}
    pool = ((after.get("inference") or {}).get("scheduler") or {}).get("pool") or {}
    state_zero = persistent.get("allocated") == 0 and pool.get("allocated") == 0
    result = {
        "schema": "rwkv-agent-state-http-smoke-v2",
        "endpoint": args.endpoint,
        "model": after.get("model"),
        "before": {
            "persistent_allocated": (before.get("persistent_states") or {}).get(
                "allocated"
            ),
            "pool_allocated": (
                ((before.get("inference") or {}).get("scheduler") or {}).get("pool")
                or {}
            ).get("allocated"),
        },
        "equivalence": equivalence,
        "owner_isolation": isolation,
        "two_round_protocol": protocols,
        "after": {
            "persistent_allocated": persistent.get("allocated"),
            "pool_allocated": pool.get("allocated"),
            "shape_counts": (
                (after.get("inference") or {}).get("scheduler") or {}
            ).get("shape_counts"),
        },
        "state_zero": state_zero,
    }
    result["passed"] = bool(
        equivalence["passed"]
        and isolation["passed"]
        and all(protocol["passed"] for protocol in protocols.values())
        and state_zero
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(output)
    print(payload, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
