from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from typing import Any
from urllib.parse import urlsplit

from .controller import AgentController


class AgentHTTPServer(ThreadingHTTPServer):
    controller: AgentController


class Handler(BaseHTTPRequestHandler):
    server: AgentHTTPServer

    def _send(self, status: int, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 4 * 1024 * 1024:
            raise ValueError("invalid request body length")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/health":
            self._send(404, {"status": "not_found"})
            return
        try:
            self._send(200, self.server.controller.health())
        except Exception as exc:
            self._send(
                503,
                {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"},
            )

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            body = self._body()
            session_id = str(body.get("session_id") or "").strip()
            if path in {
                "/v1/agent/run",
                "/v1/agent/run_stateful",
                "/v1/tools/call",
            } and not session_id:
                raise ValueError("session_id must not be empty")
            if path == "/v1/agent/run":
                result = self.server.controller.run(
                    str(body.get("message") or ""),
                    session_id=session_id,
                )
            elif path == "/v1/agent/run_stateful":
                result = self.server.controller.run_stateful_search(
                    str(body.get("message") or ""),
                    session_id=session_id,
                    branch_width=int(body.get("branch_width", 4)),
                    max_rounds=int(body.get("max_rounds", 2)),
                )
            elif path == "/v1/agent/gate":
                context = body.get("context", "")
                if not isinstance(context, str) or len(context) > 4000:
                    raise ValueError("context must be a string of at most 4000 chars")
                has_pasted_text = body.get("has_pasted_text", False)
                if not isinstance(has_pasted_text, bool):
                    raise ValueError("has_pasted_text must be boolean")
                threshold = body.get("threshold")
                result = self.server.controller.decide_tool(
                    str(body.get("message") or ""),
                    threshold=None if threshold is None else float(threshold),
                    context=context,
                    has_pasted_text=has_pasted_text,
                )
            elif path == "/v1/tools/call":
                arguments = body.get("arguments") or {}
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                result = self.server.controller.execute_tool(
                    str(body.get("name") or ""),
                    arguments,
                    session_id=session_id,
                )
            else:
                self._send(404, {"status": "not_found"})
                return
            self._send(200, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"status": "invalid", "message": str(exc)})
        except Exception as exc:
            self._send(
                500,
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
            )

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8120)
    parser.add_argument(
        "--model-urls",
        default="http://127.0.0.1:8118,http://127.0.0.1:8119",
    )
    parser.add_argument(
        "--knowledge-endpoint",
        default="http://127.0.0.1:19220",
    )
    parser.add_argument(
        "--memory-path",
        default="var/sessions.sqlite3",
    )
    parser.add_argument("--web-config", default="configs/default.json")
    parser.add_argument(
        "--tool-gate-threshold",
        type=float,
        default=float(os.getenv("RWKV_AGENT_TOOL_GATE_THRESHOLD", "0.7")),
    )
    args = parser.parse_args()
    controller = AgentController(
        model_urls=[
            item.strip() for item in args.model_urls.split(",") if item.strip()
        ],
        knowledge_endpoint=args.knowledge_endpoint,
        memory_path=args.memory_path,
        web_config=args.web_config,
        tool_gate_threshold=args.tool_gate_threshold,
    )
    server = AgentHTTPServer((args.host, args.port), Handler)
    server.controller = controller
    try:
        server.serve_forever()
    finally:
        server.server_close()
        controller.close()


if __name__ == "__main__":
    main()
