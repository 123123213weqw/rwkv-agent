#!/usr/bin/env python3
"""Compare strict three-tool prompt templates on the greedy G1I runtime."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

from rwkv_agent.controller import (
    ModelClient,
    TOOL_SCHEMAS,
    TOOLS,
    parse_tool_call,
    render_tool_prompt,
)


CASES = [
    {
        "id": "long_zh_1",
        "tool": "long_text_qa",
        "question": "红岸工程这次常规发射是第几次？",
        "message": "红岸工程这次常规发射是第几次？",
    },
    {
        "id": "long_zh_2",
        "tool": "long_text_qa",
        "question": "项目最终选择了哪种调度方案？",
        "message": "项目最终选择了哪种调度方案？",
    },
    {
        "id": "long_zh_3",
        "tool": "long_text_qa",
        "question": "Batch 8的吞吐是多少？",
        "message": "Batch 8的吞吐是多少？",
    },
    {
        "id": "long_en_1",
        "tool": "long_text_qa",
        "question": "Who sent the first warning?",
        "message": "Who sent the first warning?",
    },
    {
        "id": "long_en_2",
        "tool": "long_text_qa",
        "question": "What prefill quantum was selected?",
        "message": "What prefill quantum was selected?",
    },
    {
        "id": "long_en_3",
        "tool": "long_text_qa",
        "question": "Which candidate had the highest score?",
        "message": "Which candidate had the highest score?",
    },
    {
        "id": "long_zh_4",
        "tool": "long_text_qa",
        "question": "古筝行动的目标船叫什么？",
        "message": "古筝行动的目标船叫什么？",
    },
    {
        "id": "long_zh_5",
        "tool": "long_text_qa",
        "question": "最后一次实验为什么失败？",
        "message": "最后一次实验为什么失败？",
    },
    {
        "id": "long_zh_6",
        "tool": "long_text_qa",
        "question": "会议确定的下一步是什么？",
        "message": "会议确定的下一步是什么？",
    },
    {
        "id": "long_en_4",
        "tool": "long_text_qa",
        "question": "Which configuration won the benchmark?",
        "message": "Which configuration won the benchmark?",
    },
    {
        "id": "long_en_5",
        "tool": "long_text_qa",
        "question": "What action item was assigned to Alice?",
        "message": "What action item was assigned to Alice?",
    },
    {
        "id": "long_en_6",
        "tool": "long_text_qa",
        "question": "What caused the final timeout?",
        "message": "What caused the final timeout?",
    },
    {
        "id": "web_zh_1",
        "tool": "web_search",
        "message": "联网搜索今天RWKV的最新消息。",
    },
    {
        "id": "web_zh_2",
        "tool": "web_search",
        "message": "查一下Python当前稳定版本并给出官网来源。",
    },
    {
        "id": "web_en_1",
        "tool": "web_search",
        "message": "Search the web for today's latest CUDA release notes.",
    },
    {
        "id": "web_en_2",
        "tool": "web_search",
        "message": "Find the current RWKV organization website online.",
    },
    {
        "id": "web_zh_3",
        "tool": "web_search",
        "message": "搜索本周最新的PyTorch发布公告。",
    },
    {
        "id": "web_zh_4",
        "tool": "web_search",
        "message": "联网查明天北京天气。",
    },
    {
        "id": "web_en_3",
        "tool": "web_search",
        "message": "Find the latest official NVIDIA driver release online.",
    },
    {
        "id": "web_en_4",
        "tool": "web_search",
        "message": "Search current Hugging Face service status.",
    },
    {
        "id": "web_en_5",
        "tool": "web_search",
        "message": "Look up today's exchange rate and cite a public source.",
    },
    {
        "id": "knowledge_zh_1",
        "tool": "knowledge_search",
        "message": "搜索本地知识库中的RWKV Agent架构。",
    },
    {
        "id": "knowledge_zh_2",
        "tool": "knowledge_search",
        "message": "从内部知识索引查找FineWiki检索方案。",
    },
    {
        "id": "knowledge_en_1",
        "tool": "knowledge_search",
        "message": "Search the local knowledge base for the scheduler design.",
    },
    {
        "id": "knowledge_en_2",
        "tool": "knowledge_search",
        "message": "Look up the FineWiki benchmark in our internal documents.",
    },
    {
        "id": "knowledge_zh_3",
        "tool": "knowledge_search",
        "message": "在本地索引里查找长期知识检索实验结果。",
    },
    {
        "id": "knowledge_zh_4",
        "tool": "knowledge_search",
        "message": "查询内部文档中的Evidence设计，不指定单个文件。",
    },
    {
        "id": "knowledge_en_3",
        "tool": "knowledge_search",
        "message": "Search internal knowledge for the Agent memory decision.",
    },
    {
        "id": "knowledge_en_4",
        "tool": "knowledge_search",
        "message": "Find the V100 state benchmark in local indexed documents.",
    },
    {
        "id": "knowledge_en_5",
        "tool": "knowledge_search",
        "message": "Look up our retrieval architecture in the knowledge base.",
    },
]


def _examples() -> str:
    return (
        "\nUser: What is the current stable Python version?\n\nAssistant:"
        '<tool_call>{"name":"web_search","arguments":{"query":"Python current '
        'stable version official"}}</tool_call>\n'
        "\nUser: Search the local knowledge base for the RWKV Agent design."
        "\n\nAssistant:"
        '<tool_call>{"name":"knowledge_search","arguments":{"query":"RWKV Agent '
        'design"}}</tool_call>\n'
        "\nSystem: Active pasted long text: yes."
        "\nUser: Who founded the Red Coast base?\n\nAssistant:"
        '<tool_call>{"name":"long_text_qa","arguments":{"question":'
        '"Who founded the Red Coast base?"}}</tool_call>\n'
    )


def render_variant(style: str, case: dict[str, Any]) -> str:
    message = str(case["message"])
    has_pasted_text = case["tool"] == "long_text_qa"
    if style in {"production_examples", "schema_examples"}:
        return render_tool_prompt(
            message,
            has_pasted_text=has_pasted_text,
        )
    if style == "system_json_examples":
        system = {
            "task": "call_exactly_one_function",
            "output": (
                "one strict <tool_call> JSON block and no other text"
            ),
            "rules": [
                "web_search is only for live public Internet information",
                "knowledge_search is for local indexed knowledge without a path",
                "long_text_qa is only for active pasted session text",
                "long_text_qa takes only the current question",
            ],
            "active_pasted_long_text": has_pasted_text,
            "functions": [TOOL_SCHEMAS[name] for name in TOOLS],
        }
        prefix = "System: " + json.dumps(
            system,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return prefix + _examples() + f"\nUser: {message.strip()}\n\nAssistant:"
    if style == "compact_examples":
        prefix = (
            "System: Call exactly one function and output only "
            "<tool_call>{\"name\":...,\"arguments\":...}</tool_call>.\n"
            "Functions:\n"
            "- web_search(query): live public Internet\n"
            "- knowledge_search(query): local indexed knowledge, no file path\n"
            "- long_text_qa(question): active pasted session text only\n"
            f"Active pasted long text: {'yes' if has_pasted_text else 'no'}.\n"
            "Do not copy source text into arguments. Do not answer."
        )
        return prefix + _examples() + f"\nUser: {message.strip()}\n\nAssistant:"
    raise ValueError(f"unknown style: {style}")


def evaluate(case: dict[str, Any], parsed: dict[str, Any]) -> dict[str, bool]:
    strict = bool(parsed.get("strict"))
    tool_correct = strict and parsed.get("tool") == case["tool"]
    arguments = parsed.get("arguments") or {}
    question_present = True
    argument_valid = True
    if case["tool"] == "long_text_qa":
        question_present = (
            tool_correct
            and isinstance(arguments.get("question"), str)
            and bool(arguments["question"].strip())
        )
        argument_valid = question_present and set(arguments) == {"question"}
    return {
        "strict": strict,
        "tool_correct": tool_correct,
        "argument_valid": argument_valid,
        "question_present": question_present,
        "passed": strict and tool_correct and argument_valid,
    }


def run_one(
    client: ModelClient,
    *,
    style: str,
    case: dict[str, Any],
    repeat: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    completion = client.complete(
        render_variant(style, case),
        max_tokens=128,
    )
    parsed = parse_tool_call(completion["raw"])
    return {
        "style": style,
        "case_id": case["id"],
        "repeat": repeat,
        "expected_tool": case["tool"],
        "message": case["message"],
        "raw": completion["raw"],
        "parsed": parsed,
        "evaluation": evaluate(case, parsed),
        "output_tokens": completion["output_tokens"],
        "model_elapsed_ms": completion["model_elapsed_ms"],
        "request_elapsed_ms": completion["request_elapsed_ms"],
        "wall_elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "model": completion["model"],
        "url": completion["url"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    styles: dict[str, Any] = {}
    for style in sorted({row["style"] for row in rows}):
        selected = [row for row in rows if row["style"] == style]
        total = len(selected)
        by_tool = {}
        for tool in TOOLS:
            tool_rows = [
                row for row in selected if row["expected_tool"] == tool
            ]
            by_tool[tool] = {
                "total": len(tool_rows),
                "passed": sum(
                    row["evaluation"]["passed"] for row in tool_rows
                ),
            }
        styles[style] = {
            "total": total,
            "strict": sum(row["evaluation"]["strict"] for row in selected),
            "tool_correct": sum(
                row["evaluation"]["tool_correct"] for row in selected
            ),
            "passed": sum(row["evaluation"]["passed"] for row in selected),
            "argument_valid": sum(
                row["evaluation"]["argument_valid"] for row in selected
            ),
            "mean_output_tokens": statistics.fmean(
                row["output_tokens"] for row in selected
            ),
            "mean_model_elapsed_ms": statistics.fmean(
                row["model_elapsed_ms"] for row in selected
            ),
            "p95_model_elapsed_ms": sorted(
                row["model_elapsed_ms"] for row in selected
            )[max(0, math_ceil(0.95 * total) - 1)],
            "by_tool": by_tool,
        }
    ranking = sorted(
        styles,
        key=lambda style: (
            styles[style]["passed"],
            styles[style]["strict"],
            -styles[style]["mean_output_tokens"],
            -styles[style]["mean_model_elapsed_ms"],
        ),
        reverse=True,
    )
    repeated_outputs: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        repeated_outputs.setdefault(
            (row["style"], row["case_id"]),
            [],
        ).append(row["raw"])
    repeated_groups = [
        outputs
        for outputs in repeated_outputs.values()
        if len(outputs) > 1
    ]
    return {
        "model_mode": "greedy_argmax",
        "cases": len(CASES),
        "rows": len(rows),
        "styles": styles,
        "ranking": ranking,
        "winner": ranking[0],
        "repeat_groups": len(repeated_groups),
        "repeat_raw_exact": (
            all(len(set(outputs)) == 1 for outputs in repeated_groups)
            if repeated_groups
            else None
        ),
    }


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if value == integer else integer + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-urls",
        default="http://127.0.0.1:8118,http://127.0.0.1:8119",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--styles",
        default="production_examples,system_json_examples,compact_examples",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/greedy_tool_call_template_matrix_v1.json",
    )
    args = parser.parse_args()
    styles = [item.strip() for item in args.styles.split(",") if item.strip()]
    client = ModelClient(
        [item.strip() for item in args.model_urls.split(",") if item.strip()]
    )
    jobs = [
        (style, case, repeat)
        for repeat in range(max(1, args.repeats))
        for style in styles
        for case in CASES
    ]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=max(1, args.concurrency),
        thread_name_prefix="greedy-tool-bench",
    ) as executor:
        futures = {
            executor.submit(
                run_one,
                client,
                style=style,
                case=case,
                repeat=repeat,
            ): (style, case["id"], repeat)
            for style, case, repeat in jobs
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row["style"], row["case_id"], row["repeat"]))
    summary = summarize(rows)
    payload = {
        "schema": "rwkv-agent-greedy-tool-call-bench-v1",
        "created_unix": time.time(),
        "elapsed_s": round(time.perf_counter() - started, 6),
        "summary": summary,
        "rows": rows,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload["sha256_without_sha_field"] = hashlib.sha256(canonical).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output.resolve())


if __name__ == "__main__":
    main()
