"""HTTP boundary for the Python retrieval/Evidence data plane."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any
from urllib.parse import urlsplit

from .controller import build_semantic_scorer_from_env
from .data_plane import AgentDataPlane
from .model_client import ModelClient
from .session_text import SessionTextBuffer
from .tools import KnowledgeSearchAdapter, LongTextQAAdapter, WebSearchAdapter


class DataPlaneHTTPServer(ThreadingHTTPServer):
    data_plane: AgentDataPlane
    model: ModelClient


class Handler(BaseHTTPRequestHandler):
    server: DataPlaneHTTPServer

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

    @staticmethod
    def _session_id(body: dict[str, Any]) -> str:
        session_id = str(body.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        return session_id

    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/health":
            self._send(404, {"status": "not_found"})
            return
        try:
            self._send(
                200,
                {
                    **self.server.data_plane.health(),
                    "model": self.server.model.health(),
                },
            )
        except Exception as exc:
            self._send(
                503,
                {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"},
            )

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            body = self._body()
            data_plane = self.server.data_plane
            if path == "/v1/tools/call":
                session_id = self._session_id(body)
                arguments = body.get("arguments")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                result = data_plane.execute(
                    str(body.get("name") or ""),
                    arguments,
                    session_id=session_id,
                    original_query=str(body.get("original_query") or "") or None,
                )
            elif path == "/v1/session/text":
                result = data_plane.capture_text(
                    self._session_id(body),
                    str(body.get("text") or ""),
                )
            elif path == "/v1/session/text/status":
                result = data_plane.text_status(self._session_id(body))
            elif path == "/v1/answers/validate":
                evidence = body.get("evidence")
                if not isinstance(evidence, list):
                    raise ValueError("evidence must be an array")
                result = data_plane.validate_answer(
                    question=str(body.get("question") or ""),
                    answer=str(body.get("answer") or ""),
                    evidence=[dict(item) for item in evidence if isinstance(item, dict)],
                )
            elif path == "/v1/evidence/reduce":
                tool_results = body.get("tool_results")
                if not isinstance(tool_results, list):
                    raise ValueError("tool_results must be an array")
                result = {
                    "status": "ok",
                    "evidence": data_plane.reduce_evidence(
                        question=str(body.get("question") or ""),
                        tool_results=[
                            dict(item) for item in tool_results if isinstance(item, dict)
                        ],
                        limit=int(body.get("limit", 8)),
                    ),
                }
            elif path == "/v1/queries/coordinate":
                observation = body.get("observation")
                if observation is not None and not isinstance(observation, dict):
                    raise ValueError("observation must be an object or null")
                used_queries = body.get("used_queries") or []
                if not isinstance(used_queries, list) or not all(
                    isinstance(item, str) for item in used_queries
                ):
                    raise ValueError("used_queries must be a string array")
                result = data_plane.coordinate_query(
                    question=str(body.get("question") or ""),
                    generated_query=str(body.get("generated_query") or ""),
                    branch_index=int(body.get("branch_index", 0)),
                    round_index=int(body.get("round_index", 1)),
                    observation=observation,
                    used_queries=used_queries,
                )
            else:
                self._send(404, {"status": "not_found"})
                return
            self._send(200, result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
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
    parser.add_argument("--port", type=int, default=8121)
    parser.add_argument("--model-urls", default="http://127.0.0.1:8417")
    parser.add_argument("--knowledge-endpoint", default="http://127.0.0.1:19220")
    parser.add_argument("--web-config", default="configs/default.json")
    args = parser.parse_args()
    model = ModelClient(
        [item.strip() for item in args.model_urls.split(",") if item.strip()]
    )
    semantic_scorer = build_semantic_scorer_from_env()
    web = WebSearchAdapter(args.web_config, semantic_scorer=semantic_scorer)
    knowledge = KnowledgeSearchAdapter(args.knowledge_endpoint)
    long_text = LongTextQAAdapter(model.complete)
    session_text = SessionTextBuffer(
        max_chars=int(getattr(long_text, "max_document_chars", 1_000_000)),
    )
    data_plane = AgentDataPlane(
        web=web,
        knowledge=knowledge,
        long_text=long_text,
        session_text=session_text,
        semantic_scorer=semantic_scorer,
    )
    server = DataPlaneHTTPServer((args.host, args.port), Handler)
    server.data_plane = data_plane
    server.model = model
    try:
        server.serve_forever()
    finally:
        server.server_close()
        data_plane.close()


if __name__ == "__main__":
    main()
