#!/usr/bin/env python3
"""Gate-1 smoke for a converted RWKV-7 model on the ROCm HF backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_greedy(scheduler, request_ids: list[str], max_new_tokens: int, eos: int):
    outputs = {request_id: [] for request_id in request_ids}
    active = list(request_ids)
    for _step in range(max_new_tokens):
        sampled = scheduler.sample_next(active)
        advance = {}
        next_active = []
        for request_id in active:
            token = int(sampled[request_id])
            if token == eos:
                continue
            outputs[request_id].append(token)
            advance[request_id] = token
            next_active.append(request_id)
        if advance:
            scheduler.advance_tokens(advance)
        active = next_active
        if not active:
            break
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="rwkv7-g1i-preview4922-13.3b")
    parser.add_argument("--model-sha256", default="")
    parser.add_argument("--context", type=int, default=12288)
    parser.add_argument("--prefill-chunk-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).resolve()

    os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rwkv7_scheduler import HFRecurrentScheduler, SchedulerConfig
    from rwkv_agent.state_runtime import PersistentStateRuntime

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("Gate 1 requires a visible ROCm GPU")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    loaded_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    torch.cuda.synchronize()
    loaded_seconds = time.perf_counter() - loaded_started
    model_allocated_bytes = torch.cuda.memory_allocated()

    scheduler = HFRecurrentScheduler(
        model,
        config=SchedulerConfig(
            prefill_chunk_size=args.prefill_chunk_size,
            max_batch_size=2,
            max_queue_size=8,
            max_input_tokens=args.context,
            eos_token_id=0,
        ),
        device="cuda",
        capacity=8,
    )

    prompt = "System: You are a concise assistant.\n\nUser: What is 2+2?\n\nAssistant:"

    def one_shot(request_id: str) -> tuple[list[int], str, float]:
        shot_started = time.perf_counter()
        scheduler.admit(request_id, tokenizer.encode(prompt))
        scheduler.prefill([request_id])
        token_ids = decode_greedy(
            scheduler,
            [request_id],
            args.max_new_tokens,
            0,
        )[request_id]
        text = tokenizer.decode(token_ids)
        scheduler.release(request_id)
        torch.cuda.synchronize()
        return token_ids, text, time.perf_counter() - shot_started

    first_ids, first_text, first_seconds = one_shot("repeat-a")
    second_ids, second_text, second_seconds = one_shot("repeat-b")
    repeatable = first_ids == second_ids and first_text == second_text
    if not repeatable:
        raise AssertionError("greedy completion was not repeatable")

    runtime = PersistentStateRuntime(
        tokenizer=tokenizer,
        scheduler=scheduler,
        context_limit=args.context,
        eos_token_id=0,
        capacity=4,
        ttl_seconds=600.0,
    )
    prompt_a = "System: Answer briefly.\n\nUser: Continue sequence A: alpha\n\nAssistant:"
    prompt_b = "System: Answer briefly.\n\nUser: Continue sequence B: beta\n\nAssistant:"
    continuation_a = "\nObservation: branch A is independent.\nAssistant:"
    continuation_b = "\nObservation: branch B is independent.\nAssistant:"
    state_a = runtime.prefill(
        owner_id="owner-a",
        prompt=prompt_a,
        branch="a",
    )
    state_b = runtime.prefill(
        owner_id="owner-b",
        prompt=prompt_b,
        branch="b",
    )
    reference_a = runtime.prefill(
        owner_id="reference-a",
        prompt=prompt_a,
        branch="reference-a",
    )
    reference_b = runtime.prefill(
        owner_id="reference-b",
        prompt=prompt_b,
        branch="reference-b",
    )
    owner_isolation = False
    try:
        runtime.release(
            owner_id="owner-b",
            state_ids=[state_a["state_id"]],
        )
    except PermissionError:
        owner_isolation = True
    if not owner_isolation:
        raise AssertionError("cross-owner release was not rejected")

    before_seen = {
        state_a["state_id"]: scheduler.request(state_a["state_id"]).seen_tokens,
        state_b["state_id"]: scheduler.request(state_b["state_id"]).seen_tokens,
    }
    batch_started = time.perf_counter()
    scheduler.continue_many(
        [
            (state_a["state_id"], tokenizer.encode(continuation_a)),
            (state_b["state_id"], tokenizer.encode(continuation_b)),
        ]
    )
    batched_ids = decode_greedy(
        scheduler,
        [state_a["state_id"], state_b["state_id"]],
        args.max_new_tokens,
        0,
    )
    torch.cuda.synchronize()
    batch_seconds = time.perf_counter() - batch_started
    batched_text = {
        request_id: tokenizer.decode(token_ids)
        for request_id, token_ids in batched_ids.items()
    }
    scheduler.continue_many(
        [(reference_a["state_id"], tokenizer.encode(continuation_a))]
    )
    reference_a_ids = decode_greedy(
        scheduler,
        [reference_a["state_id"]],
        args.max_new_tokens,
        0,
    )[reference_a["state_id"]]
    scheduler.continue_many(
        [(reference_b["state_id"], tokenizer.encode(continuation_b))]
    )
    reference_b_ids = decode_greedy(
        scheduler,
        [reference_b["state_id"]],
        args.max_new_tokens,
        0,
    )[reference_b["state_id"]]
    if batched_ids[state_a["state_id"]] != reference_a_ids:
        raise AssertionError("batched state A differs from its isolated reference")
    if batched_ids[state_b["state_id"]] != reference_b_ids:
        raise AssertionError("batched state B differs from its isolated reference")
    for request_id in batched_ids:
        request = scheduler.request(request_id)
        if not bool(torch.isfinite(request.logits).all().item()):
            raise AssertionError(f"non-finite logits for {request_id}")
        if request.seen_tokens <= before_seen[request_id]:
            raise AssertionError(f"state {request_id} did not advance")

    resident_metrics = scheduler.metrics()
    forward_seconds = resident_metrics.get("forward_time_us", 0) / 1_000_000.0
    forward_tokens_per_second = (
        resident_metrics.get("forward_tokens", 0) / forward_seconds
        if forward_seconds > 0
        else 0.0
    )
    runtime.release(owner_id="owner-a", state_ids=[state_a["state_id"]])
    runtime.release(owner_id="owner-b", state_ids=[state_b["state_id"]])
    runtime.release(
        owner_id="reference-a",
        state_ids=[reference_a["state_id"]],
    )
    runtime.release(
        owner_id="reference-b",
        state_ids=[reference_b["state_id"]],
    )
    released_metrics = scheduler.metrics()
    if scheduler.pool.allocated != 0 or scheduler.pool.free != scheduler.pool.capacity:
        raise AssertionError("state counter did not return to zero")

    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    vocab_path = model_path / "rwkv_vocab_v20230424.txt"
    result: dict[str, Any] = {
        "status": "pass",
        "model": {
            "id": args.model_id,
            "path": str(model_path),
            "sha256": args.model_sha256,
            "dtype": args.dtype,
            "context": args.context,
            "class": type(model).__name__,
            "parameters": sum(int(value.numel()) for value in model.parameters()),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "path": str(vocab_path),
            "sha256": sha256_file(vocab_path) if vocab_path.exists() else "",
        },
        "runtime": {
            "backend": "hf_recurrent",
            "greedy": True,
            "prefill_chunk_size": args.prefill_chunk_size,
            "loaded_seconds": round(loaded_seconds, 3),
            "wall_seconds": round(time.perf_counter() - started, 3),
        },
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "total_vram_bytes": total_bytes,
            "model_allocated_bytes": model_allocated_bytes,
            "free_vram_bytes_after": free_bytes,
            "allocated_bytes_after": torch.cuda.memory_allocated(),
            "reserved_bytes_after": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "checks": {
            "single_prompt_greedy": True,
            "repeatable": repeatable,
            "prefill_continue": True,
            "two_independent_states_batched": True,
            "owner_isolation": owner_isolation,
            "finite_logits": True,
            "state_counter_zero": True,
        },
        "single_prompt": {
            "token_ids": first_ids,
            "text": first_text,
            "cold_seconds": round(first_seconds, 6),
            "warm_seconds": round(second_seconds, 6),
        },
        "batch": {
            "state_ids": [state_a["state_id"], state_b["state_id"]],
            "before_seen_tokens": before_seen,
            "token_ids": batched_ids,
            "text": batched_text,
            "isolated_reference_token_ids": {
                "a": reference_a_ids,
                "b": reference_b_ids,
            },
            "continuation_decode_wall_seconds": round(batch_seconds, 6),
        },
        "performance": {
            "aggregate_model_tokens_per_second": round(
                forward_tokens_per_second,
                3,
            ),
            "model_forward_tokens": resident_metrics.get("forward_tokens", 0),
            "model_forward_seconds": round(forward_seconds, 6),
            "b2_output_tokens": sum(len(values) for values in batched_ids.values()),
        },
        "resident_scheduler_metrics": resident_metrics,
        "released_scheduler_metrics": released_metrics,
        "persistent_state_metrics": runtime.health(),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
