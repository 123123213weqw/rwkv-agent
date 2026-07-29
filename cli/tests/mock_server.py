#!/usr/bin/env python3
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value: dict) -> None:
        payload = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self.send_json(
            {
                "status": "ready",
                "tools": [
                    "web_search",
                    "knowledge_search",
                    "long_text_qa",
                ],
                "model": [
                    {
                        "status": "ready",
                        "model": "rwkv7-g1i-preview4922-13.3b",
                        "context": 12288,
                    }
                ],
                "state_parallel_search": {"enabled": True},
            }
        )

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        if self.path == "/v1/agent/run":
            self.send_json(
                {
                    "status": "ok",
                    "session_id": body["session_id"],
                    "route": {"tool": "knowledge_search"},
                    "answer": f"mock answer for {body['message']}",
                    "tool_result": {"status": "ok", "evidence": []},
                }
            )
            return
        if self.path == "/v1/agent/run_stateful":
            branches = body.get("branch_width", 4)
            rounds = body.get("max_rounds", 2)
            self.send_json(
                {
                    "status": "ok",
                    "session_id": body["session_id"],
                    "route": {
                        "mode": "state_parallel_search",
                        "tool": "web_search",
                        "branch_width": branches,
                        "rounds": rounds,
                    },
                    "answer": f"mock research for {body['message']}",
                    "tool_result": {
                        "status": "ok",
                        "tool": "web_search",
                        "evidence": [
                            {
                                "id": "W1",
                                "title": "Mock research evidence",
                                "content": "Mock stateful content",
                                "uri": "https://example.invalid/research",
                            }
                        ],
                    },
                }
            )
            return
        name = body["name"]
        if name == "memory" and body["arguments"]["action"] == "write":
            self.send_json(
                {
                    "status": "accepted",
                    "memory_id": "MEM-MOCK",
                    "message": "Memory saved.",
                }
            )
        else:
            memory_read = (
                name == "memory" and body["arguments"]["action"] == "read"
            )
            self.send_json(
                {
                    "status": "ok",
                    "source": "agent_memory" if memory_read else "mock",
                    "evidence": [
                        {
                            "id": (
                                "M1"
                                if memory_read
                                else "W1"
                                if name == "web_search"
                                else "K1"
                            ),
                            "title": "Mock evidence",
                            "content": "Mock content",
                            "uri": "https://example.invalid/mock",
                        }
                    ],
                }
            )

    def log_message(self, format: str, *args: object) -> None:
        return


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18121
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
