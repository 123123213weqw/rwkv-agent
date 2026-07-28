#!/usr/bin/env python3
"""Real G1I serial-vs-concurrent smoke for the integrated Sidecar batcher."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from rwkv_agent.sidecar import NativeG1I


PROMPTS = [
    "System: Answer briefly.\n\nUser: Explain recurrent state batching. "
    * 24
    + "\n\nAssistant:",
    "System: 简短回答。\n\nUser: 解释RWKV状态复用。"
    * 32
    + "\n\nAssistant:",
    "System: Continue the technical note.\n\nUser: Dynamic microbatch scheduling "
    * 28
    + "\n\nAssistant:",
    "System: 续写技术说明。\n\nUser: 长文本分块预填充与无填充尾块。"
    * 28
    + "\n\nAssistant:",
    "System: Be concise.\n\nUser: Why must recurrent state rows stay isolated? "
    * 25
    + "\n\nAssistant:",
    "System: 简洁作答。\n\nUser: 为什么自定义CUDA核要求连续状态？"
    * 30
    + "\n\nAssistant:",
    "System: Give one paragraph.\n\nUser: Describe active-row greedy decoding. "
    * 26
    + "\n\nAssistant:",
    "System: 用一段话回答。\n\nUser: 描述连续批处理的公平性。"
    * 34
    + "\n\nAssistant:",
]


def digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [row["token_ids"] for row in rows],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()

    service = NativeG1I()
    try:
        started = time.perf_counter()
        serial = [
            service.complete(prompt, stops=[], max_tokens=args.max_tokens)
            for prompt in PROMPTS
        ]
        serial_seconds = time.perf_counter() - started

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(PROMPTS)) as executor:
            concurrent = list(
                executor.map(
                    lambda prompt: service.complete(
                        prompt,
                        stops=[],
                        max_tokens=args.max_tokens,
                    ),
                    PROMPTS,
                )
            )
        concurrent_seconds = time.perf_counter() - started
        exact = [
            left["token_ids"] == right["token_ids"]
            and left["text"] == right["text"]
            for left, right in zip(serial, concurrent, strict=True)
        ]
        result = {
            "schema": "rwkv_agent_sidecar_continuous_batch_smoke_v1",
            "model_load_seconds": service.loaded_seconds,
            "requests": len(PROMPTS),
            "max_tokens": args.max_tokens,
            "serial_seconds": serial_seconds,
            "concurrent_seconds": concurrent_seconds,
            "speedup": serial_seconds / concurrent_seconds,
            "all_exact": all(exact),
            "exact": exact,
            "serial_digest": digest(serial),
            "concurrent_digest": digest(concurrent),
            "serial": serial,
            "concurrent": concurrent,
            "health": service.health(),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    "requests": result["requests"],
                    "serial_seconds": result["serial_seconds"],
                    "concurrent_seconds": result["concurrent_seconds"],
                    "speedup": result["speedup"],
                    "all_exact": result["all_exact"],
                    "serial_digest": result["serial_digest"],
                    "concurrent_digest": result["concurrent_digest"],
                    "shape_counts": result["health"]["inference"]["scheduler"][
                        "shape_counts"
                    ],
                },
                ensure_ascii=False,
            )
        )
        if not result["all_exact"]:
            raise SystemExit(2)
    finally:
        service.close()


if __name__ == "__main__":
    main()
