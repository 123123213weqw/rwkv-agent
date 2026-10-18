#!/usr/bin/env python3
"""G1I correctness and production prefill-throughput matrix on Albatross."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from rwkv7_scheduler import (
    AlbatrossChunkScheduler,
    AlbatrossStatePool,
    SchedulerConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--pool-capacity", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--quantums", default="64,128,256,512")
    parser.add_argument(
        "--token-source",
        choices=("natural", "synthetic"),
        default="natural",
    )
    return parser.parse_args()


def setup_model(args: argparse.Namespace) -> tuple[Any, Any, str]:
    runtime_dir = str(Path(args.runtime_dir).resolve())
    sys.path.insert(0, runtime_dir)
    os.chdir(runtime_dir)
    import rwkv7_fast_v3a as v3a
    from rwkv.utils import PIPELINE

    v3a.MODEL_PATH = str(Path(args.model).resolve())
    v3a.WKV_MODE = "fp32io16"
    v3a.EMB_DEVICE = "cpu"
    v3a.RKV_MODE = "off"
    v3a.CMIX_SPARSE = "no-fc"
    v3a.LOWRANK_WEIGHT = "both"
    v3a.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
    torch.set_grad_enabled(False)
    v3a.load_extensions(v3a.WKV_MODE)
    model = v3a.RWKV7()
    tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")
    return model, tokenizer, "cpu" if model.emb_cpu else "cuda"


def make_tokens(
    lengths: list[int],
    salt: int,
    *,
    tokenizer: Any,
    source: str,
) -> list[list[int]]:
    if source == "synthetic":
        return [
            [
                ((salt + row * 997 + index * 37) % 65000) + 1
                for index in range(length)
            ]
            for row, length in enumerate(lengths)
        ]
    templates = [
        (
            "System: 你是一个严谨、简洁的中文助手。"
            "请阅读上下文并只回答用户当前的问题。\n\n"
            "Context: RWKV是一种循环神经网络语言模型架构。"
            "它使用固定大小的递归状态处理任意长度的序列，"
            "并可以把长输入拆成多个连续区块计算。"
        ),
        (
            "System: You are a precise assistant. Read the supplied context "
            "and answer only the current question.\n\n"
            "Context: A recurrent language model carries a fixed-size state "
            "between consecutive chunks. Chunk boundaries must preserve the "
            "same token order and state transition semantics."
        ),
    ]
    rows = []
    for row, length in enumerate(lengths):
        text = (
            templates[row % len(templates)]
            + f"\nDocument section {salt}-{row}: "
            + ("这是用于生产级长上下文调度测试的自然语言段落。" * 64)
            + "\nUser: 请根据以上上下文简短回答。\n\nAssistant:"
        )
        base = [0] + tokenizer.encode(text)
        if len(base) < length:
            filler = tokenizer.encode(
                " 后续区块继续保留相同主题、事实顺序和上下文关系。"
            )
            while len(base) < length:
                base.extend(filler)
        rows.append(base[:length])
    return rows
    return [
        [
            ((salt + row * 997 + index * 37) % 65000) + 1
            for index in range(length)
        ]
        for row, length in enumerate(lengths)
    ]


@torch.inference_mode()
def serial_chunked(
    model: Any,
    token_lists: list[list[int]],
    *,
    quantum: int,
    decode_tokens: int,
    token_device: str,
) -> tuple[list[list[int]], float]:
    outputs = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for values in token_lists:
        state = model.zero_state(1)
        logits = None
        for offset in range(0, len(values), quantum):
            tokens = torch.tensor(
                values[offset : offset + quantum],
                dtype=torch.long,
                device=token_device,
            ).view(1, -1)
            logits = model.forward(tokens, state)
        assert logits is not None
        output = []
        for _ in range(decode_tokens):
            token = int(torch.argmax(logits[0]).item())
            if token == 0:
                break
            output.append(token)
            logits = model.forward(
                torch.tensor([[token]], dtype=torch.long, device=token_device),
                state,
            )
        outputs.append(output)
    torch.cuda.synchronize()
    return outputs, time.perf_counter() - started


@torch.inference_mode()
def scheduled(
    scheduler: AlbatrossChunkScheduler,
    token_lists: list[list[int]],
    *,
    decode_tokens: int,
    run_id: str,
) -> tuple[list[list[int]], float, dict[str, Any]]:
    ids = [f"{run_id}-{index}" for index in range(len(token_lists))]
    torch.cuda.synchronize()
    started = time.perf_counter()
    scheduler.admit_many(zip(ids, token_lists, strict=True))
    scheduler.prefill(ids)
    output_by_id = scheduler.greedy_decode(
        ids,
        max_new_tokens=decode_tokens,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    outputs = [output_by_id[request_id] for request_id in ids]
    metrics = scheduler.metrics()
    for request_id in ids:
        scheduler.release(request_id)
    return outputs, elapsed, metrics


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":")).encode()
    ).hexdigest()


def run_pair(
    model: Any,
    pool: AlbatrossStatePool,
    token_lists: list[list[int]],
    *,
    quantum: int,
    decode_tokens: int,
    token_device: str,
    repeats: int,
    label: str,
) -> dict[str, Any]:
    scheduler = AlbatrossChunkScheduler(
        model,
        pool=pool,
        config=SchedulerConfig(
            prefill_chunk_size=quantum,
            max_batch_size=pool.max_batch_size,
            max_queue_size=pool.capacity,
            max_input_tokens=12288,
        ),
        token_device=token_device,
    )
    serial_times = []
    scheduled_times = []
    exact = []
    mismatch_counts = []
    mismatch_examples = []
    reference_digest = ""
    scheduled_digest = ""
    last_metrics = {}
    for repeat in range(repeats):
        reference, serial_seconds = serial_chunked(
            model,
            token_lists,
            quantum=quantum,
            decode_tokens=decode_tokens,
            token_device=token_device,
        )
        result, batch_seconds, last_metrics = scheduled(
            scheduler,
            token_lists,
            decode_tokens=decode_tokens,
            run_id=f"{label}-q{quantum}-r{repeat}",
        )
        serial_times.append(serial_seconds)
        scheduled_times.append(batch_seconds)
        exact.append(result == reference)
        mismatches = [
            index
            for index, (left, right) in enumerate(
                zip(result, reference, strict=True)
            )
            if left != right
        ]
        mismatch_counts.append(len(mismatches))
        if mismatches:
            mismatch_examples.append(
                {
                    "repeat": repeat,
                    "rows": mismatches,
                    "reference": {
                        str(index): reference[index] for index in mismatches
                    },
                    "scheduled": {
                        str(index): result[index] for index in mismatches
                    },
                }
            )
        reference_digest = digest(reference)
        scheduled_digest = digest(result)
    serial_median = statistics.median(serial_times)
    scheduled_median = statistics.median(scheduled_times)
    total_input = sum(map(len, token_lists))
    total_output = sum(map(len, reference))
    return {
        "label": label,
        "batch_size": len(token_lists),
        "lengths": list(map(len, token_lists)),
        "quantum": quantum,
        "repeats": repeats,
        "all_exact": all(exact),
        "exact_by_repeat": exact,
        "mismatch_counts": mismatch_counts,
        "mismatch_examples": mismatch_examples,
        "reference_digest": reference_digest,
        "scheduled_digest": scheduled_digest,
        "serial_seconds": serial_times,
        "scheduled_seconds": scheduled_times,
        "serial_median_seconds": serial_median,
        "scheduled_median_seconds": scheduled_median,
        "speedup": serial_median / scheduled_median,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "scheduled_input_tokens_per_second": total_input / scheduled_median,
        "scheduled_total_tokens_per_second": (
            total_input + total_output
        )
        / scheduled_median,
        "scheduler_metrics": last_metrics,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    load_started = time.perf_counter()
    model, tokenizer, token_device = setup_model(args)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    model_allocated = torch.cuda.memory_allocated()
    pool = AlbatrossStatePool(
        model,
        capacity=args.pool_capacity,
        max_batch_size=args.max_batch_size,
    )
    pool.prewarm(
        value
        for value in (1, 2, 4, 8)
        if value <= args.max_batch_size
    )

    # Warm representative prefill and decode kernels.
    warm_scheduler = AlbatrossChunkScheduler(
        model,
        pool=pool,
        config=SchedulerConfig(
            prefill_chunk_size=128,
            max_batch_size=args.max_batch_size,
            max_queue_size=args.pool_capacity,
        ),
        token_device=token_device,
    )
    scheduled(
        warm_scheduler,
        make_tokens(
            [256] * min(2, args.max_batch_size),
            11,
            tokenizer=tokenizer,
            source=args.token_source,
        ),
        decode_tokens=2,
        run_id="warmup",
    )

    quantums = [int(value) for value in args.quantums.split(",") if value.strip()]
    tuning_lengths = [384, 512, 640, 768, 896, 1024, 1152, 1280]
    tuning = []
    for quantum in quantums:
        tuning.append(
            run_pair(
                model,
                pool,
                make_tokens(
                    tuning_lengths,
                    100 + quantum,
                    tokenizer=tokenizer,
                    source=args.token_source,
                ),
                quantum=quantum,
                decode_tokens=args.decode_tokens,
                token_device=token_device,
                repeats=args.repeats,
                label="autotune_variable_medium_b8",
            )
        )
    valid = [row for row in tuning if row["all_exact"]]
    if not valid:
        raise RuntimeError("no exact quantum candidate")
    chosen = max(valid, key=lambda row: row["speedup"])["quantum"]

    cases = {
        "equal_512_b8": [512] * 8,
        "equal_2048_b8": [2048] * 8,
        "variable_short_b8": [41, 47, 53, 65, 83, 107, 137, 179],
        "variable_medium_b8": tuning_lengths,
        "variable_long_b8": [
            2048,
            2560,
            3072,
            3584,
            4096,
            4608,
            5120,
            6144,
        ],
    }
    matrix = []
    for index, (label, lengths) in enumerate(cases.items()):
        matrix.append(
            run_pair(
                model,
                pool,
                make_tokens(
                    lengths,
                    1000 + index,
                    tokenizer=tokenizer,
                    source=args.token_source,
                ),
                quantum=chosen,
                decode_tokens=args.decode_tokens,
                token_device=token_device,
                repeats=args.repeats,
                label=label,
            )
        )

    torch.cuda.synchronize()
    result = {
        "schema": "rwkv7_albatross_production_scheduler_bench_v1",
        "model": str(Path(args.model).resolve()),
        "runtime_dir": str(Path(args.runtime_dir).resolve()),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "load_seconds": load_seconds,
        "model_allocated_bytes": model_allocated,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "pool": {
            "capacity": pool.capacity,
            "max_batch_size": pool.max_batch_size,
            "bytes_per_slot": pool.bytes_per_slot,
            "slab_bytes": pool.slab_bytes,
            "workspace_bytes": pool.workspace_bytes,
        },
        "config": {
            "decode_tokens": args.decode_tokens,
            "repeats": args.repeats,
            "quantum_candidates": quantums,
            "chosen_quantum": chosen,
            "token_source": args.token_source,
            "wkv": "fp32io16",
            "emb": "cpu",
            "batched_rkv": "off",
            "cmix_sparse": "no-fc",
            "lowrank_weight": "both",
        },
        "autotune": tuning,
        "matrix": matrix,
        "all_exact": all(row["all_exact"] for row in tuning + matrix),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "chosen_quantum": chosen,
                "autotune": [
                    {
                        "quantum": row["quantum"],
                        "speedup": row["speedup"],
                        "all_exact": row["all_exact"],
                    }
                    for row in tuning
                ],
                "matrix": [
                    {
                        "label": row["label"],
                        "speedup": row["speedup"],
                        "all_exact": row["all_exact"],
                        "scheduled_input_tokens_per_second": row[
                            "scheduled_input_tokens_per_second"
                        ],
                    }
                    for row in matrix
                ],
                "peak_allocated_bytes": result["peak_allocated_bytes"],
                "all_exact": result["all_exact"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
