from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlsplit

from .config import (
    CrawlConfig,
    RealtimeSearchConfig,
    SearchConfig,
    ShadowSearchConfig,
)
from .crawler import FocusedCrawler
from .db import SearchDatabase
from .debug_trace import DebugTrace, DebugTraceStore
from .protocol import (
    SCHEMA_VERSION,
    ChatRequest,
    EventFactory,
    ProtocolError,
    RequestRegistry,
    chunk_text,
    normalize_history,
)
from .service import SearchService


WEB_ROOT = Path(__file__).resolve().parent / "web"
_CANCEL_PATH = re.compile(r"^/api/v1/requests/([A-Za-z0-9][A-Za-z0-9._:-]{0,127})/cancel$")


class SearchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        database: SearchDatabase,
        search_config: Optional[SearchConfig] = None,
        crawl_config: Optional[CrawlConfig] = None,
        answerer: Optional[Any] = None,
        model_status: Optional[Dict[str, Any]] = None,
        realtime_config: Optional[RealtimeSearchConfig] = None,
        shadow_config: Optional[ShadowSearchConfig] = None,
    ) -> None:
        super().__init__(address, SearchRequestHandler)
        self.database = database
        self.service = SearchService(
            database,
            search_config=search_config,
            answerer=answerer,
            realtime_config=realtime_config,
            shadow_config=shadow_config,
        )
        self.crawl_config = crawl_config or CrawlConfig()
        self.model_status = model_status or self._answerer_status(answerer)
        self.request_registry = RequestRegistry()
        self.debug_traces = DebugTraceStore()

    def server_close(self) -> None:
        self.service.close()
        super().server_close()

    @staticmethod
    def _answerer_status(answerer: Optional[Any]) -> Dict[str, Any]:
        if answerer is None:
            return {
                "enabled": False,
                "ready": False,
                "label": "RWKV",
                "model": None,
                "error": None,
            }
        status = getattr(answerer, "status", None)
        if callable(status):
            return dict(status())
        return {
            "enabled": True,
            "ready": True,
            "label": "RWKV",
            "model": type(answerer).__name__,
            "error": None,
        }


class SearchRequestHandler(BaseHTTPRequestHandler):
    server: SearchHTTPServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/api/health":
            live_model_status = self.server._answerer_status(
                self.server.service.answerer
            )
            if not live_model_status.get("enabled") and self.server.model_status.get(
                "error"
            ):
                live_model_status = self.server.model_status
            self._json(
                {
                    "ok": True,
                    "stats": self.server.database.stats(),
                    "model": live_model_status,
                    "protocol": {"version": SCHEMA_VERSION, "stream_endpoint": "/api/v1/chat/stream"},
                    "active_requests": self.server.request_registry.active_count(),
                    "backend_debug": self.server.debug_traces.status(),
                    "realtime_search": (
                        self.server.service.realtime_engine.status()
                        if self.server.service.realtime_engine
                        else {"enabled": False, "ready": False, "error": None}
                    ),
                    "shadow_search": self.server.service.shadow_status(),
                }
            )
            return
        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = (params.get("q") or [""])[0]
            freshness = (params.get("freshness") or ["stable"])[0]
            try:
                limit = min(50, max(1, int((params.get("limit") or ["10"])[0])))
            except ValueError:
                limit = 10
            self._json({"query": query, "results": self.server.service.search(query, freshness=freshness, limit=limit)})
            return
        self._static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        cancel_match = _CANCEL_PATH.fullmatch(parsed.path)
        if cancel_match:
            request_id = cancel_match.group(1)
            accepted = self.server.request_registry.cancel(request_id)
            self._json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "request_id": request_id,
                    "cancelled": accepted,
                    "error": None
                    if accepted
                    else {"code": "REQUEST_NOT_FOUND", "message": "request is not active"},
                },
                status=HTTPStatus.ACCEPTED if accepted else HTTPStatus.NOT_FOUND,
            )
            return
        payload = self._read_json()
        if payload is None:
            return
        if parsed.path == "/api/v1/chat/stream":
            self._v1_chat_stream(payload)
            return
        if parsed.path == "/api/ask":
            query = str(payload.get("query") or "")
            timezone = str(payload.get("timezone") or "Asia/Shanghai")
            mode = str(payload.get("mode") or "fast")
            history = self._normalize_history(payload.get("history"))
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                for event in self.server.service.ask_events(
                    query,
                    user_timezone=timezone,
                    mode=mode,
                    history=history,
                ):
                    event_name = str(event.get("type") or "message")
                    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                    self.wfile.write(f"event: {event_name}\ndata: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True
            return
        if parsed.path == "/api/crawl":
            raw_urls = payload.get("urls") or []
            if isinstance(raw_urls, str):
                raw_urls = [raw_urls]
            urls = [str(item) for item in raw_urls[:100]]
            max_pages = min(1000, max(1, int(payload.get("max_pages") or 50)))

            def worker() -> None:
                crawler = FocusedCrawler(self.server.database, self.server.crawl_config)
                crawler.seed(urls)
                asyncio.run(crawler.run(max_pages))

            threading.Thread(target=worker, daemon=True, name="rwkv-search-crawler").start()
            self._json({"accepted": len(urls), "max_pages": max_pages}, status=HTTPStatus.ACCEPTED)
            return
        self._json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def _v1_chat_stream(self, payload: Dict[str, Any]) -> None:
        try:
            request = ChatRequest.from_payload(payload)
            cancel_event = self.server.request_registry.register(request.request_id)
        except ProtocolError as exc:
            status = (
                HTTPStatus.CONFLICT
                if exc.code == "DUPLICATE_REQUEST_ID"
                else HTTPStatus.BAD_REQUEST
            )
            self._json(
                {"schema_version": SCHEMA_VERSION, "error": exc.to_dict()}, status=status
            )
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-RWKV-Request-ID", request.request_id)
        self.end_headers()
        trace = self.server.debug_traces.open(request)
        trace_status = "completed"
        trace_error: Optional[str] = None
        try:
            for event in self._v1_events(request, cancel_event, trace=trace):
                if trace is not None and event.get("type") not in {
                    "answer_delta",
                    "discovery_progress",
                    "fetch_progress",
                }:
                    trace.write("protocol_event", {"event": event})
                self._write_sse(event)
        except (BrokenPipeError, ConnectionResetError):
            cancel_event.set()
            trace_status = "client_disconnected"
        except Exception as exc:
            trace_status = "failed"
            trace_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            factory = EventFactory(request)
            try:
                self._write_sse(
                    factory.make(
                        "error",
                        state="failed",
                        error={
                            "code": "INTERNAL_ERROR",
                            "message": f"{type(exc).__name__}: {str(exc)[:300]}",
                            "retryable": True,
                        },
                    )
                )
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            self.server.request_registry.finish(request.request_id)
            if trace is not None:
                trace.close(trace_status, trace_error)
            self.close_connection = True

    def _v1_events(
        self,
        request: ChatRequest,
        cancel_event: threading.Event,
        *,
        trace: Optional[DebugTrace] = None,
    ) -> Any:
        factory = EventFactory(request)
        started = time.perf_counter()
        search_finished: Optional[float] = None
        generation_started: Optional[float] = None
        final_sources: list[Dict[str, Any]] = []
        candidate_sources: list[Dict[str, Any]] = []
        model_meta: Optional[Dict[str, Any]] = None
        yielded_done = False
        fetched_pages = 0
        saw_answer_delta = False

        yield factory.make(
            "request_started",
            state="queued",
            accepted={
                "search_mode": request.search_mode,
                "research_depth": request.research_depth,
                "source_scope": request.source_scope,
                "use_finewiki": request.use_finewiki,
                "timezone": request.timezone,
                "locale": request.locale,
            },
        )
        service_events = self.server.service.ask_events(
            request.query,
            user_timezone=request.timezone,
            mode=request.research_depth,
            history=request.history,
            search_mode=request.search_mode,
            source_scope=request.source_scope,
            use_finewiki=request.use_finewiki,
            cancel_event=cancel_event,
            conversation_id=request.conversation_id,
            debug=trace is not None,
        )
        for service_event in service_events:
            kind = str(service_event.get("type") or "")
            if kind == "route":
                route = dict(service_event.get("route") or {})
                yield factory.make("route", state="routing", route=route)
                if route.get("queries") and not route.get("needs_clarification"):
                    yield factory.make(
                        "search_plan",
                        state="discovering",
                        plan={
                            "queries": route.get("queries") or [],
                            "tools": route.get("tools") or [],
                            "depth": route.get("depth") or "single",
                            "freshness": route.get("freshness") or "stable",
                            "source_scope": request.source_scope,
                            "use_finewiki": request.use_finewiki,
                        },
                    )
                elif route.get("intent") != "time" and not route.get(
                    "needs_clarification"
                ):
                    generation_started = time.perf_counter()
                    yield factory.make(
                        "generation_started",
                        state="generating",
                        model={
                            "label": self.server.model_status.get("label"),
                            "model": self.server.model_status.get("model"),
                        },
                    )
            elif kind == "sources":
                candidate_sources = [
                    self._source_record(item, fallback_id=f"D{index}")
                    for index, item in enumerate(service_event.get("sources") or [], start=1)
                ]
                yield factory.make(
                    "discovery_progress",
                    state="discovering",
                    progress={
                        "candidate_count": len(candidate_sources),
                        "query_count": 0,
                        "message": f"检索得到 {len(candidate_sources)} 个候选结果",
                    },
                )
                fetched_pages = int(
                    (service_event.get("stats") or {}).get("fetched") or fetched_pages
                )
            elif kind in {"discovery_progress", "fetch_progress"}:
                progress = dict(service_event.get("progress") or {})
                if kind == "fetch_progress":
                    fetched_pages = max(
                        fetched_pages, int(progress.get("succeeded") or 0)
                    )
                yield factory.make(
                    kind,
                    state="fetching" if kind == "fetch_progress" else "discovering",
                    progress=progress,
                )
            elif kind == "search_warning":
                yield factory.make(
                    "warning",
                    state="discovering",
                    warning={
                        "code": str(service_event.get("code") or "SEARCH_WARNING"),
                        "message": str(service_event.get("message") or "搜索链路部分降级"),
                    },
                )
            elif kind == "evidence":
                evidence = service_event.get("evidence") or []
                final_sources = [
                    self._source_record(item, fallback_id=f"S{index}")
                    for index, item in enumerate(evidence, start=1)
                ]
                if not final_sources:
                    final_sources = candidate_sources
                search_finished = time.perf_counter()
                yield factory.make(
                    "evidence_ready",
                    state="ranking",
                    sources=final_sources,
                    evidence_count=len(evidence),
                )
                generation_started = time.perf_counter()
                yield factory.make(
                    "generation_started",
                    state="generating",
                    model={
                        "label": self.server.model_status.get("label"),
                        "model": self.server.model_status.get("model"),
                    },
                )
            elif kind == "answer_delta":
                if generation_started is None:
                    generation_started = time.perf_counter()
                    yield factory.make(
                        "generation_started",
                        state="generating",
                        model={
                            "label": self.server.model_status.get("label"),
                            "model": self.server.model_status.get("model"),
                        },
                    )
                if not cancel_event.is_set():
                    saw_answer_delta = True
                    yield factory.make(
                        "answer_delta",
                        state="generating",
                        delta=str(service_event.get("delta") or ""),
                    )
            elif kind == "debug":
                if trace is not None:
                    trace.write(
                        "model_debug",
                        {"debug": dict(service_event.get("debug") or {})},
                    )
            elif kind == "answer":
                if generation_started is None:
                    generation_started = time.perf_counter()
                    yield factory.make(
                        "generation_started", state="generating", model=None
                    )
                answer = dict(service_event.get("answer") or {})
                content = str(answer.get("answer") or "")
                if not saw_answer_delta:
                    for delta in chunk_text(content):
                        if cancel_event.is_set():
                            break
                        yield factory.make(
                            "answer_delta", state="generating", delta=delta
                        )
                if cancel_event.is_set():
                    yield factory.make(
                        "warning",
                        state="cancelled",
                        warning={
                            "code": "REQUEST_CANCELLED",
                            "message": "请求已取消",
                        },
                    )
                    yield factory.make("done", state="cancelled")
                    yielded_done = True
                    break
                model_meta = service_event.get("model")
                now = time.perf_counter()
                search_ms = (
                    max(0.0, (search_finished - started) * 1000.0)
                    if search_finished
                    else 0.0
                )
                generation_ms = (
                    max(0.0, (now - generation_started) * 1000.0)
                    if generation_started
                    else 0.0
                )
                usage = {
                    "search_ms": round(search_ms, 2),
                    "generation_ms": round(generation_ms, 2),
                    "total_ms": round((now - started) * 1000.0, 2),
                    "new_tokens": int((model_meta or {}).get("new_tokens") or 0),
                    "candidate_sources": len(candidate_sources),
                    "evidence_sources": len(final_sources),
                    "fetched_pages": fetched_pages,
                }
                yield factory.make(
                    "answer_final",
                    state="completed",
                    answer={
                        "content": content,
                        "citations": list(answer.get("citations") or []),
                        "data_time": str(answer.get("data_time") or ""),
                        "insufficient_evidence": bool(
                            answer.get("insufficient_evidence")
                        ),
                        "needs_clarification": bool(
                            answer.get("needs_clarification")
                        ),
                    },
                    sources=final_sources,
                    usage=usage,
                    model=model_meta,
                )
            elif kind == "cancelled":
                yield factory.make(
                    "warning",
                    state="cancelled",
                    warning={"code": "REQUEST_CANCELLED", "message": "请求已取消"},
                )
                yield factory.make("done", state="cancelled")
                yielded_done = True
                break
            elif kind == "error":
                yield factory.make(
                    "error",
                    state="failed",
                    error={
                        "code": "INVALID_REQUEST",
                        "message": str(service_event.get("message") or "request failed"),
                        "retryable": False,
                    },
                )
            elif kind == "done":
                yield factory.make("done", state="completed")
                yielded_done = True
        if not yielded_done:
            state = "cancelled" if cancel_event.is_set() else "completed"
            yield factory.make("done", state=state)

    @staticmethod
    def _source_record(item: Any, *, fallback_id: str) -> Dict[str, Any]:
        value = dict(item) if isinstance(item, dict) else {}
        return {
            "id": str(value.get("evidence_id") or value.get("id") or fallback_id),
            "title": str(value.get("title") or value.get("url") or "来源"),
            "url": str(value.get("url") or ""),
            "snippet": str(value.get("text") or value.get("snippet") or "")[:1600],
            "source_type": str(value.get("source_type") or "web"),
            "published_at": value.get("published_at"),
            "updated_at": value.get("updated_at"),
            "fetched_at": value.get("fetched_at"),
            "authority": value.get("authority_score", value.get("authority")),
            "score": value.get("retrieval_score", value.get("score")),
            "matched_channels": value.get("matched_channels") or [],
        }

    def _write_sse(self, event: Dict[str, Any]) -> None:
        event_name = str(event.get("type") or "message")
        data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        self.wfile.write(f"event: {event_name}\ndata: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _read_json(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 1024 * 1024:
            self._json({"error": "invalid content length"}, status=HTTPStatus.BAD_REQUEST)
            return None
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json({"error": "invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
            return None
        if not isinstance(value, dict):
            self._json({"error": "JSON object required"}, status=HTTPStatus.BAD_REQUEST)
            return None
        return value

    @staticmethod
    def _normalize_history(value: Any) -> list[Dict[str, str]]:
        return normalize_history(value)

    def _json(self, value: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")


def serve(
    database: SearchDatabase,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    search_config: Optional[SearchConfig] = None,
    crawl_config: Optional[CrawlConfig] = None,
    answerer: Optional[Any] = None,
    model_status: Optional[Dict[str, Any]] = None,
    realtime_config: Optional[RealtimeSearchConfig] = None,
    shadow_config: Optional[ShadowSearchConfig] = None,
) -> None:
    server = SearchHTTPServer(
        (host, port),
        database,
        search_config,
        crawl_config,
        answerer=answerer,
        model_status=model_status,
        realtime_config=realtime_config,
        shadow_config=shadow_config,
    )
    print(f"RWKV Local Search: http://{host}:{port}")
    model = server.model_status
    if model.get("ready"):
        print(
            "Model: "
            f"{model.get('model') or model.get('label')} "
            f"({model.get('device', 'unknown')}, {model.get('dtype', 'unknown')})"
        )
    elif model.get("enabled"):
        print(f"Model unavailable; extractive fallback active: {model.get('error')}")
    else:
        print("Model disabled; extractive fallback active")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
