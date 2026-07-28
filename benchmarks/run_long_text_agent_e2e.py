#!/usr/bin/env python3
"""Exercise pasted-text capture through greedy Tool Call, Evidence and answer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

from rwkv_agent.controller import AgentController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", required=True)
    parser.add_argument(
        "--question",
        default="红岸工程这次常规发射是第几次？",
    )
    parser.add_argument(
        "--model-urls",
        default="http://127.0.0.1:8118,http://127.0.0.1:8119",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/long_text_agent_e2e_v1.json",
    )
    args = parser.parse_args()

    document = Path(args.document).expanduser().resolve()
    document_bytes = document.read_bytes()
    document_text = document_bytes.decode("utf-8")
    session_id = "bench-pasted-text-e2e"
    started = time.perf_counter()

    with tempfile.TemporaryDirectory() as directory:
        controller = AgentController(
            model_urls=[
                item.strip()
                for item in args.model_urls.split(",")
                if item.strip()
            ],
            memory_path=str(Path(directory) / "sessions.sqlite3"),
        )
        try:
            capture = controller.run(
                document_text,
                session_id=session_id,
            )
            answer = controller.run(
                args.question,
                session_id=session_id,
            )
        finally:
            controller.close()

    capture_document = dict(
        (capture.get("tool_result") or {}).get("document") or {}
    )
    capture_document.pop("sha256", None)
    summary = {
        "schema": "rwkv-agent-pasted-text-e2e-v1",
        "model_mode": "greedy_argmax",
        "input": {
            "name": document.name,
            "bytes": len(document_bytes),
            "chars": len(document_text),
            "sha256": hashlib.sha256(document_bytes).hexdigest(),
            "transport": "chat_message_then_question",
        },
        "question": args.question,
        "capture": {
            "status": capture.get("status"),
            "route": capture.get("route"),
            "answer": capture.get("answer"),
            "document": capture_document,
            "model_called": (capture.get("trace") or {}).get("model_called"),
        },
        "qa": answer,
        "checks": {
            "capture_without_model": (
                (capture.get("route") or {}).get("mode")
                == "document_capture"
                and (capture.get("trace") or {}).get("model_called") is False
            ),
            "strict_question_only_tool_call": (
                answer.get("status") == "ok"
                and (answer.get("route") or {}).get("tool") == "long_text_qa"
                and set(
                    ((answer.get("route") or {}).get("arguments") or {})
                )
                == {"question"}
            ),
            "has_grounded_evidence": bool(
                (answer.get("tool_result") or {}).get("evidence")
            ),
        },
        "elapsed_s": round(time.perf_counter() - started, 6),
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
