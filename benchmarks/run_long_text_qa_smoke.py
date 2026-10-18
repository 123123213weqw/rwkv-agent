#!/usr/bin/env python3
"""Real G1I smoke for generic parallel long-text chunk QA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from rwkv_agent.controller import ModelClient
from rwkv_agent.tools.long_text import LongTextQAAdapter


CASES = [
    {
        "id": "red_coast_launch_number",
        "question": "红岸工程这次常规发射是第几次？",
        "answer": "147",
    },
    {
        "id": "ozma_frequency_mhz",
        "question": "OZMA计划的单通道接收频率是多少兆赫？",
        "answer": "1420",
    },
    {
        "id": "guzheng_target_ship",
        "question": "古筝行动要夺取信息的目标船叫什么？",
        "answer": "审判日",
    },
    {
        "id": "first_alien_warning",
        "question": "来自另一个世界的第一条警告核心内容是什么？",
        "answer": "不要回答",
    },
    {
        "id": "human_computer_os",
        "question": "三体世界的人列计算机运行的操作系统叫什么？",
        "answer": "秦1.0",
    },
    {
        "id": "nanomaterial_codename",
        "question": "汪淼团队制造的超强度纳米材料代号叫什么？",
        "answer": "飞刃",
    },
]


def canonical_evidence(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item["chunk_id"],
            "answer_candidate": item["answer_candidate"],
            "content": item["content"],
        }
        for item in result.get("evidence", [])
    ]


def run_mode(
    *,
    client: ModelClient,
    document_text: str,
    document_name: str,
    concurrency: int,
    top_k: int,
) -> dict[str, Any]:
    adapter = LongTextQAAdapter(
        client.complete,
        concurrency=concurrency,
        top_k=top_k,
        max_evidence=8,
    )
    rows = []
    started = time.perf_counter()
    for case in CASES:
        case_started = time.perf_counter()
        result = adapter.execute(
            document_text,
            case["question"],
            document_name=document_name,
        )
        candidates = [
            str(item.get("answer_candidate") or "")
            for item in result.get("evidence", [])
        ]
        evidence_text = "\n".join(
            str(item.get("content") or "")
            for item in result.get("evidence", [])
        )
        rows.append(
            {
                **case,
                "status": result.get("status"),
                "candidate_exact": case["answer"] in candidates,
                "candidate_contains_answer": any(
                    case["answer"] in candidate for candidate in candidates
                ),
                "evidence_contains_answer": case["answer"] in evidence_text,
                "evidence": canonical_evidence(result),
                "workers": result.get("workers"),
                "retrieval": result.get("retrieval"),
                "elapsed_s": round(time.perf_counter() - case_started, 6),
            }
        )
    elapsed = time.perf_counter() - started
    canonical = json.dumps(
        [row["evidence"] for row in rows],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "concurrency": concurrency,
        "top_k": top_k,
        "elapsed_s": round(elapsed, 6),
        "candidate_exact": sum(row["candidate_exact"] for row in rows),
        "candidate_contains_answer": sum(
            row["candidate_contains_answer"] for row in rows
        ),
        "evidence_contains_answer": sum(
            row["evidence_contains_answer"] for row in rows
        ),
        "evidence_sha256": hashlib.sha256(canonical).hexdigest(),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", required=True)
    parser.add_argument(
        "--model-urls",
        default="http://127.0.0.1:8118,http://127.0.0.1:8119",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--modes",
        default="1,8",
        help="Comma-separated chunk-worker concurrency values.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/long_text_qa_smoke_v1.json",
    )
    args = parser.parse_args()
    document = Path(args.document).expanduser().resolve()
    if not document.is_file():
        raise SystemExit(f"missing document: {document}")
    document_bytes = document.read_bytes()
    document_text = document_bytes.decode("utf-8")
    client = ModelClient(
        [item.strip() for item in args.model_urls.split(",") if item.strip()]
    )
    modes = [
        run_mode(
            client=client,
            document_text=document_text,
            document_name=document.name,
            concurrency=int(item),
            top_k=args.top_k,
        )
        for item in args.modes.split(",")
        if item.strip()
    ]
    serial = next(
        (mode for mode in modes if mode["concurrency"] == 1),
        None,
    )
    fastest = min(modes, key=lambda mode: mode["elapsed_s"])
    summary = {
        "schema": "rwkv-agent-long-text-qa-smoke-v1",
        "model_mode": "greedy_argmax",
        "document": {
            "name": document.name,
            "bytes": len(document_bytes),
            "chars": len(document_text),
            "sha256": hashlib.sha256(document_bytes).hexdigest(),
            "runtime_source": "pasted_session_text",
        },
        "cases": len(CASES),
        "modes": modes,
        "fastest_concurrency": fastest["concurrency"],
        "speedup_vs_serial": (
            round(serial["elapsed_s"] / fastest["elapsed_s"], 6)
            if serial and fastest["elapsed_s"]
            else None
        ),
        "evidence_exact_across_modes": len(
            {mode["evidence_sha256"] for mode in modes}
        )
        == 1,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output.resolve())


if __name__ == "__main__":
    main()
