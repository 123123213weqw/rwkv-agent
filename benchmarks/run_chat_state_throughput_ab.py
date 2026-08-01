#!/usr/bin/env python3
"""Transcript re-prefill versus recurrent Session State throughput A/B.

This runner talks only to an already-started isolated G1I Sidecar.  It does not
start, stop or reconfigure a service.  Both arms use the same greedy prompts,
turns and token budget; the only changed variable is full-transcript prefill
versus incremental persistent-State continuation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Protocol
from urllib.request import Request, urlopen
import uuid

SCHEMA = "rwkv_agent_chat_state_throughput_ab.v1"
DEFAULT_CONCURRENCY = (1, 4, 8, 16)
DEFAULT_TURNS = 3
DEFAULT_MAX_TOKENS = 8
THEMES = (
    "recurrent state batching",
    "chunked prefill scheduling",
    "function call routing",
    "evidence-grounded answers",
    "persistent session context",
    "parallel long-text reading",
    "active-row greedy decode",
    "bounded Agent research",
    "local knowledge retrieval",
    "web evidence selection",
    "citation validation",
    "state ownership isolation",
    "GPU state slab reuse",
    "decode batch fairness",
    "tool result continuation",
    "session transcript recovery",
)
CHAT_STOPS = (
    "\n\nUser:",
    "\nUser:",
    "\nSystem:",
    "</s>",
)
CHAT_USER_STOPS = frozenset({"\n\nUser:", "\nUser:"})
DIRECT_SYSTEM_PROMPT = (
    "System: You are a helpful conversational assistant. Answer the user "
    "directly in the user's language. Do not claim to have searched, do not "
    "invent sources or citation IDs, and do not emit a tool call. Use the "
    "supplied conversation when relevant. The conversation is the only memory "
    "available; there is no extracted long-term profile. Do not mention memory "
    "machinery unless the user asks about it. Never output <think> tags or "
    "hidden reasoning.\n\n"
)


def render_direct_chat_prefix() -> str:
    return DIRECT_SYSTEM_PROMPT


def render_direct_chat_turn(
    message: str,
    *,
    continuation: bool,
    previous_stop: str = "",
) -> str:
    clean = str(message or "").strip()
    if not clean:
        raise ValueError("message must not be empty")
    if continuation and previous_stop in CHAT_USER_STOPS:
        return f" {clean}\n\nAssistant:"
    prefix = "\n\n" if continuation else ""
    return prefix + f"User: {clean}\n\nAssistant:"


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


@dataclass(frozen=True, slots=True)
class Conversation:
    session: str
    messages: tuple[str, ...]


def build_conversations(
    count: int,
    *,
    turns: int = DEFAULT_TURNS,
) -> list[Conversation]:
    if not 1 <= int(count) <= len(THEMES):
        raise ValueError(f"conversation count must be 1..{len(THEMES)}")
    if not 2 <= int(turns) <= 6:
        raise ValueError("turns must be 2..6")
    output = []
    for index, theme in enumerate(THEMES[: int(count)], start=1):
        material = (
            f"Technical note {index}: {theme}. "
            "RWKV retains a fixed-size recurrent state while new tokens are "
            "processed incrementally. Exact scheduling must preserve each "
            "request's state, token order, and stop boundary. "
        ) * 10
        followups = (
            "Summarize the main mechanism in one sentence.",
            "What invariant must the scheduler preserve?",
            "Give one concise implementation consequence.",
            "State one likely throughput bottleneck.",
            "Give one correctness check for this design.",
        )
        messages = (
            "Read this note and answer briefly:\n" + material,
            *followups[: int(turns) - 1],
        )
        output.append(
            Conversation(
                session=f"chat-state-ab-{index:02d}",
                messages=tuple(messages),
            )
        )
    return output


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[rank]


def _shape_counts(health: dict[str, Any]) -> dict[str, int]:
    scheduler = (
        dict(health.get("inference") or {})
        .get("scheduler", {})
    )
    return {
        str(key): int(value)
        for key, value in dict(
            dict(scheduler).get("shape_counts") or {}
        ).items()
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
            sum(batch * count for batch, count in rows) / calls
            if calls
            else 0.0
        )

    return {
        "shape_counts": delta,
        "decode_average_batch_fill": round(fill(token_length=1), 4),
        "prefill_average_batch_fill": round(fill(token_length=None), 4),
        "model_forward_calls": sum(delta.values()),
    }


def _parallel(
    concurrency: int,
    function: Any,
    items: list[Any],
) -> list[Any]:
    with ThreadPoolExecutor(max_workers=int(concurrency)) as executor:
        return list(executor.map(function, items))


def _completion_row(
    transport: SidecarTransport,
    *,
    prompt: str,
    max_tokens: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = transport.post(
        "/v1/completions",
        {
            "prompt": prompt,
            "stop": list(CHAT_STOPS),
            "max_tokens": int(max_tokens),
        },
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return dict(response["g1i"]), elapsed_ms


def run_transcript_workload(
    transport: SidecarTransport,
    conversations: list[Conversation],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    concurrency = len(conversations)
    transcripts = {
        item.session: render_direct_chat_prefix()
        for item in conversations
    }
    previous_stops = {item.session: "" for item in conversations}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for turn_index in range(len(conversations[0].messages)):
        prompts = []
        for conversation in conversations:
            prompt = transcripts[conversation.session] + render_direct_chat_turn(
                conversation.messages[turn_index],
                continuation=turn_index > 0,
                previous_stop=previous_stops[conversation.session],
            )
            prompts.append((conversation, prompt))

        def complete(value: tuple[Conversation, str]) -> dict[str, Any]:
            conversation, prompt = value
            result, elapsed_ms = _completion_row(
                transport,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            return {
                "session": conversation.session,
                "turn": turn_index + 1,
                "text": str(result.get("text") or ""),
                "token_ids": list(result.get("token_ids") or []),
                "stop_reason": str(result.get("stop_reason") or ""),
                "latency_ms": elapsed_ms,
                "queue_ms": float(result.get("queue_ms") or 0.0),
            }

        turn_rows = _parallel(concurrency, complete, prompts)
        rows.extend(turn_rows)
        for (_conversation, prompt), row in zip(prompts, turn_rows, strict=True):
            session = str(row["session"])
            transcripts[session] = prompt + str(row["text"])
            if str(row["stop_reason"]) in {"\n\nUser:", "\nUser:"}:
                transcripts[session] += str(row["stop_reason"])
            previous_stops[session] = str(row["stop_reason"])
    wall_seconds = time.perf_counter() - started
    return summarize_workload(
        "transcript_reprefill",
        rows,
        wall_seconds=wall_seconds,
        setup_seconds=0.0,
        release_seconds=0.0,
    )


def run_state_workload(
    transport: SidecarTransport,
    conversations: list[Conversation],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    concurrency = len(conversations)
    owners = {
        item.session: "bench-" + hashlib.sha256(item.session.encode()).hexdigest()[:24]
        for item in conversations
    }
    total_started = time.perf_counter()

    def prefill(conversation: Conversation) -> dict[str, Any]:
        response = transport.post(
            "/v1/states/prefill",
            {
                "owner_id": owners[conversation.session],
                "prompt": render_direct_chat_prefix(),
                "branch": "chat",
            },
        )
        return dict(response["state"])

    setup_started = time.perf_counter()
    states = _parallel(concurrency, prefill, conversations)
    setup_seconds = time.perf_counter() - setup_started
    by_session = {
        conversation.session: state
        for conversation, state in zip(conversations, states, strict=True)
    }
    previous_stops = {item.session: "" for item in conversations}
    rows: list[dict[str, Any]] = []
    try:
        for turn_index in range(len(conversations[0].messages)):
            inputs = [
                (
                    conversation,
                    render_direct_chat_turn(
                        conversation.messages[turn_index],
                        continuation=turn_index > 0,
                        previous_stop=previous_stops[conversation.session],
                    ),
                )
                for conversation in conversations
            ]

            def advance(value: tuple[Conversation, str]) -> dict[str, Any]:
                conversation, input_text = value
                state = by_session[conversation.session]
                started = time.perf_counter()
                response = transport.post(
                    "/v1/states/batch_continue",
                    {
                        "owner_id": owners[conversation.session],
                        "items": [
                            {
                                "state_id": str(state["state_id"]),
                                "input": input_text,
                            }
                        ],
                        "stop": list(CHAT_STOPS),
                        "max_tokens": int(max_tokens),
                    },
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                result = dict(response["results"][0])
                return {
                    "session": conversation.session,
                    "turn": turn_index + 1,
                    "text": str(result.get("text") or ""),
                    "token_ids": list(result.get("token_ids") or []),
                    "stop_reason": str(result.get("stop_reason") or ""),
                    "latency_ms": elapsed_ms,
                    "queue_ms": float(result.get("queue_ms") or 0.0),
                    "seen_tokens": int(result.get("seen_tokens") or 0),
                }

            turn_rows = _parallel(concurrency, advance, inputs)
            rows.extend(turn_rows)
            for row in turn_rows:
                previous_stops[str(row["session"])] = str(row["stop_reason"])
    finally:
        release_started = time.perf_counter()

        def release(conversation: Conversation) -> dict[str, Any]:
            state = by_session[conversation.session]
            return transport.post(
                "/v1/states/release",
                {
                    "owner_id": owners[conversation.session],
                    "state_ids": [str(state["state_id"])],
                },
            )

        releases = _parallel(concurrency, release, conversations)
        release_seconds = time.perf_counter() - release_started
    wall_seconds = time.perf_counter() - total_started
    output = summarize_workload(
        "recurrent_session_state",
        rows,
        wall_seconds=wall_seconds,
        setup_seconds=setup_seconds,
        release_seconds=release_seconds,
    )
    output["released_states"] = sum(
        int(value.get("released") or 0)
        for value in releases
    )
    return output


def summarize_workload(
    mode: str,
    rows: list[dict[str, Any]],
    *,
    wall_seconds: float,
    setup_seconds: float,
    release_seconds: float,
) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    output_tokens = sum(len(row["token_ids"]) for row in rows)
    return {
        "mode": mode,
        "requests": len(rows),
        "output_tokens": output_tokens,
        "wall_seconds": round(float(wall_seconds), 6),
        "setup_seconds": round(float(setup_seconds), 6),
        "release_seconds": round(float(release_seconds), 6),
        "output_tokens_per_second": round(
            output_tokens / wall_seconds if wall_seconds > 0 else 0.0,
            4,
        ),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "ttft": {
            "available": False,
            "reason": "Sidecar HTTP endpoints are non-streaming",
        },
        "rows": rows,
    }


def compare_outputs(
    transcript: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    left = {
        (str(row["session"]), int(row["turn"])): row
        for row in transcript["rows"]
    }
    right = {
        (str(row["session"]), int(row["turn"])): row
        for row in state["rows"]
    }
    keys = sorted(set(left) | set(right))
    exact = [
        key in left
        and key in right
        and left[key]["token_ids"] == right[key]["token_ids"]
        and left[key]["text"] == right[key]["text"]
        for key in keys
    ]
    return {
        "compared_turns": len(keys),
        "exact_turns": sum(exact),
        "all_exact": bool(keys) and all(exact),
        "mismatches": [
            {"session": key[0], "turn": key[1]}
            for key, matched in zip(keys, exact, strict=True)
            if not matched
        ],
    }


def run_profile(
    transport: SidecarTransport,
    *,
    concurrency: int,
    turns: int,
    max_tokens: int,
    state_first: bool,
) -> dict[str, Any]:
    conversations = build_conversations(concurrency, turns=turns)
    runners = (
        (("state", run_state_workload), ("transcript", run_transcript_workload))
        if state_first
        else (("transcript", run_transcript_workload), ("state", run_state_workload))
    )
    arms: dict[str, dict[str, Any]] = {}
    for name, runner in runners:
        before = transport.get("/health")
        result = runner(
            transport,
            conversations,
            max_tokens=max_tokens,
        )
        after = transport.get("/health")
        result["scheduler"] = summarize_shape_delta(before, after)
        result["persistent_allocated_before"] = int(
            dict(before.get("persistent_states") or {}).get("allocated") or 0
        )
        result["persistent_allocated_after"] = int(
            dict(after.get("persistent_states") or {}).get("allocated") or 0
        )
        arms[name] = result
    comparison = compare_outputs(arms["transcript"], arms["state"])
    state_wall = float(arms["state"]["wall_seconds"])
    transcript_wall = float(arms["transcript"]["wall_seconds"])
    return {
        "concurrency": int(concurrency),
        "turns": int(turns),
        "max_tokens": int(max_tokens),
        "order": [name for name, _runner in runners],
        "transcript": arms["transcript"],
        "state": arms["state"],
        "comparison": comparison,
        "state_speedup": round(
            transcript_wall / state_wall if state_wall > 0 else 0.0,
            4,
        ),
        "state_leak_count": max(
            0,
            int(arms["state"]["persistent_allocated_after"])
            - int(arms["state"]["persistent_allocated_before"]),
        ),
    }


def validate_profile(profile: dict[str, Any]) -> list[str]:
    errors = []
    if not profile["comparison"]["all_exact"]:
        errors.append("output_mismatch")
    if int(profile["state_leak_count"]) != 0:
        errors.append("state_leak")
    if int(profile["state"].get("released_states") or 0) != int(
        profile["concurrency"]
    ):
        errors.append("release_mismatch")
    if int(profile["transcript"]["requests"]) != int(profile["state"]["requests"]):
        errors.append("request_count_mismatch")
    return errors


def require_persistent_capacity(
    health: dict[str, Any],
    concurrency: tuple[int, ...],
) -> int:
    persistent = dict(health.get("persistent_states") or {})
    capacity = int(persistent.get("capacity") or 0)
    required = max(concurrency)
    if capacity < required:
        raise RuntimeError(
            f"persistent state capacity {capacity} is below required "
            f"concurrency {required}; use an isolated Sidecar with capacity "
            f"at least {required}"
        )
    if int(persistent.get("allocated") or 0) != 0:
        raise RuntimeError("persistent state pool must be empty before the A/B")
    return capacity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--concurrency",
        default=",".join(str(value) for value in DEFAULT_CONCURRENCY),
    )
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    concurrency = tuple(
        int(value.strip())
        for value in args.concurrency.split(",")
        if value.strip()
    )
    if not concurrency or any(
        value < 1 or value > len(THEMES)
        for value in concurrency
    ):
        parser.error(f"--concurrency values must be 1..{len(THEMES)}")
    if args.max_tokens < 1 or args.max_tokens > 64:
        parser.error("--max-tokens must be 1..64")

    transport = HttpSidecar(args.endpoint, timeout=args.timeout)
    initial_health = transport.get("/health")
    persistent_capacity = require_persistent_capacity(
        initial_health,
        concurrency,
    )
    run_id = "chat-state-ab-" + uuid.uuid4().hex
    profiles = []
    for index, value in enumerate(concurrency):
        profile = run_profile(
            transport,
            concurrency=value,
            turns=args.turns,
            max_tokens=args.max_tokens,
            state_first=bool(index % 2),
        )
        profile["errors"] = validate_profile(profile)
        profiles.append(profile)
        print(
            json.dumps(
                {
                    "concurrency": value,
                    "state_speedup": profile["state_speedup"],
                    "all_exact": profile["comparison"]["all_exact"],
                    "state_leak_count": profile["state_leak_count"],
                    "errors": profile["errors"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    final_health = transport.get("/health")
    result = {
        "schema": SCHEMA,
        "run_id": run_id,
        "endpoint": transport.endpoint,
        "protocol": {
            "concurrency": list(concurrency),
            "turns": args.turns,
            "max_tokens": args.max_tokens,
            "greedy": True,
            "chat_stops": list(CHAT_STOPS),
            "order": "alternating",
            "prompt_set_sha256": hashlib.sha256(
                json.dumps(
                    [
                        conversation.messages
                        for conversation in build_conversations(
                            max(concurrency),
                            turns=args.turns,
                        )
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "model": initial_health.get("model"),
        "context": initial_health.get("context"),
        "persistent_capacity": persistent_capacity,
        "profiles": profiles,
        "initial_persistent_allocated": int(
            dict(initial_health.get("persistent_states") or {}).get("allocated") or 0
        ),
        "final_persistent_allocated": int(
            dict(final_health.get("persistent_states") or {}).get("allocated") or 0
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
