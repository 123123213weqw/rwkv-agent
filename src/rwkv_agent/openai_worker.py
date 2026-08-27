"""StatePool Worker adapter for vLLM and other OpenAI-compatible servers.

The adapter is intentionally a thin, optional process. It registers one exact
model identity with StatePool, reports load and health, proxies OpenAI API
requests, and implements drain-safe admission. It never claims that an
upstream KV cache can be serialized or migrated: every request still carries
its transcript, while same-Worker affinity may let the upstream reuse its own
prefix cache.

No third-party Python dependency is required. The standard-library proxy also
makes it possible to put this adapter next to an independently managed vLLM,
SGLang, llama.cpp server, or compatible hosted endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from rwkv_agent.statepool_worker import StatePoolWorkerAgent, WorkerSettings


AFFINITY_ONLY_STATE_CAPABILITY = {
    "mode": "affinity_only",
    "affinity": True,
    "snapshot": False,
    "restore": False,
    "portable_across_workers": False,
}

PROXY_POST_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/embeddings",
    }
)
MAX_ERROR_BODY_BYTES = 64 * 1024


class OpenAIWorkerConfigurationError(ValueError):
    """Raised when the explicitly launched adapter is incomplete."""


@dataclass(frozen=True, slots=True)
class OpenAIWorkerSettings:
    worker: WorkerSettings
    upstream_url: str
    upstream_model: str
    host: str = "0.0.0.0"
    port: int = 8128
    upstream_api_key: str = ""
    ingress_api_key: str = ""
    health_interval_seconds: float = 5.0
    upstream_timeout_seconds: float = 300.0
    queue_timeout_seconds: float = 30.0
    max_request_bytes: int = 16 * 1024 * 1024

    @classmethod
    def from_environment(cls) -> OpenAIWorkerSettings:
        worker = WorkerSettings.from_environment()
        if worker is None:
            raise OpenAIWorkerConfigurationError(
                "RWKV_STATEPOOL_URL is required for rwkv-openai-worker"
            )
        upstream_url = (
            os.getenv("RWKV_OPENAI_UPSTREAM_URL")
            or os.getenv("VLLM_UPSTREAM_URL")
            or ""
        ).strip().rstrip("/")
        if not upstream_url:
            raise OpenAIWorkerConfigurationError(
                "RWKV_OPENAI_UPSTREAM_URL (or VLLM_UPSTREAM_URL) is required"
            )
        parsed = urlsplit(upstream_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise OpenAIWorkerConfigurationError(
                "OpenAI upstream URL must be an absolute HTTP(S) URL"
            )
        upstream_model = os.getenv(
            "RWKV_OPENAI_UPSTREAM_MODEL", worker.model_ref["model_id"]
        ).strip()
        if not upstream_model:
            raise OpenAIWorkerConfigurationError("upstream model must not be empty")
        port = _positive_int(os.getenv("RWKV_OPENAI_WORKER_PORT", "8128"), "port")
        worker = replace(
            worker,
            endpoint=os.getenv(
                "RWKV_WORKER_ENDPOINT",
                f"http://{os.getenv('POD_IP', '127.0.0.1').strip()}:{port}",
            ).strip().rstrip("/"),
            # An OpenAI-compatible API does not imply portable runtime State.
            state_capability=dict(AFFINITY_ONLY_STATE_CAPABILITY),
        )
        return cls(
            worker=worker,
            upstream_url=upstream_url,
            upstream_model=upstream_model,
            host=os.getenv("RWKV_OPENAI_WORKER_HOST", "0.0.0.0").strip(),
            port=port,
            upstream_api_key=os.getenv("RWKV_OPENAI_UPSTREAM_API_KEY", ""),
            ingress_api_key=os.getenv("RWKV_OPENAI_WORKER_API_KEY", ""),
            health_interval_seconds=_positive_float(
                os.getenv("RWKV_OPENAI_HEALTH_INTERVAL_SECONDS", "5"),
                "health interval",
            ),
            upstream_timeout_seconds=_positive_float(
                os.getenv("RWKV_OPENAI_UPSTREAM_TIMEOUT_SECONDS", "300"),
                "upstream timeout",
            ),
            queue_timeout_seconds=_positive_float(
                os.getenv("RWKV_OPENAI_QUEUE_TIMEOUT_SECONDS", "30"),
                "queue timeout",
            ),
            max_request_bytes=_positive_int(
                os.getenv("RWKV_OPENAI_MAX_REQUEST_BYTES", str(16 * 1024 * 1024)),
                "maximum request bytes",
            ),
        )


class OpenAIWorkerRuntime:
    """Own adapter load counters, upstream health and StatePool registration."""

    def __init__(
        self,
        settings: OpenAIWorkerSettings,
        *,
        statepool_transport: (
            Callable[[str, dict[str, Any], float], dict[str, Any]] | None
        ) = None,
    ) -> None:
        self.settings = settings
        self._slots = threading.BoundedSemaphore(settings.worker.max_batch)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._probe_wake = threading.Event()
        self._probe_thread: threading.Thread | None = None
        self._active = 0
        self._waiting = 0
        self._upstream_ready = False
        self._last_probe_ms = 0
        self._last_probe_error = "not probed"
        self.worker_agent = StatePoolWorkerAgent(
            settings.worker,
            self._worker_health,
            transport=statepool_transport,
        )

    def start(self) -> None:
        self.worker_agent.start(ready=False)
        self._probe_thread = threading.Thread(
            target=self._probe_loop,
            name="openai-worker-health",
            daemon=True,
        )
        self._probe_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._probe_wake.set()
        thread = self._probe_thread
        if thread is not None:
            thread.join(timeout=max(1.0, self.settings.health_interval_seconds + 1.0))
        self.worker_agent.stop()

    def acquire(self) -> bool:
        with self._lock:
            self._waiting += 1
        acquired = self._slots.acquire(timeout=self.settings.queue_timeout_seconds)
        with self._lock:
            self._waiting -= 1
        if not acquired:
            return False
        # Check lifecycle after capacity admission. A request queued before a
        # drain cannot slip through after the Worker changes to draining.
        if not self.worker_agent.enter_request():
            self._slots.release()
            return False
        with self._lock:
            self._active += 1
        return True

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
        self._slots.release()
        self.worker_agent.exit_request()

    def status(self) -> dict[str, Any]:
        with self._lock:
            upstream_ready = self._upstream_ready
            active = self._active
            waiting = self._waiting
            last_probe_ms = self._last_probe_ms
            last_probe_error = self._last_probe_error
        return {
            "adapter": "openai-compatible",
            "upstream_ready": upstream_ready,
            "upstream_model": self.settings.upstream_model,
            "active_requests": active,
            "waiting_requests": waiting,
            "last_probe_ms": last_probe_ms,
            "last_probe_error": last_probe_error,
            "state_capability": dict(AFFINITY_ONLY_STATE_CAPABILITY),
            "statepool": self.worker_agent.status(),
        }

    def probe_once(self) -> bool:
        ready = False
        error = "upstream has no usable health endpoint"
        # vLLM exposes /health. Some otherwise compatible servers expose only
        # /v1/models, so use that as a bounded fallback rather than declaring
        # them ready without any live request.
        for path in ("/health", "/v1/models"):
            request = Request(
                self.settings.upstream_url + path,
                headers=self.upstream_headers(),
                method="GET",
            )
            try:
                with urlopen(
                    request,
                    timeout=min(10.0, self.settings.upstream_timeout_seconds),
                ) as response:
                    ready = 200 <= response.status < 300
                    error = "" if ready else f"{path}: HTTP {response.status}"
            except HTTPError as exc:
                error = f"{path}: HTTP {exc.code}"
                if exc.code == HTTPStatus.NOT_FOUND:
                    continue
            except (URLError, TimeoutError, OSError) as exc:
                error = f"{path}: {exc}"[:500]
            break
        with self._lock:
            self._upstream_ready = ready
            self._last_probe_ms = int(time.time() * 1000)
            self._last_probe_error = error
        self.worker_agent.set_available(ready)
        return ready

    def upstream_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.upstream_api_key:
            headers["Authorization"] = (
                f"Bearer {self.settings.upstream_api_key}"
            )
        return headers

    def authorized(self, authorization: str) -> bool:
        expected = self.settings.ingress_api_key
        if not expected:
            return True
        prefix = "Bearer "
        supplied = authorization[len(prefix) :] if authorization.startswith(prefix) else ""
        return hmac.compare_digest(supplied, expected)

    def normalize_payload(self, raw: bytes) -> bytes:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        requested = payload.get("model")
        advertised = self.settings.worker.model_ref["model_id"]
        if requested not in {None, advertised, self.settings.upstream_model}:
            raise ValueError(
                f"model {requested!r} is not served by this Worker ({advertised!r})"
            )
        payload["model"] = self.settings.upstream_model
        return json.dumps(payload, separators=(",", ":")).encode()

    def _worker_health(self) -> dict[str, Any]:
        with self._lock:
            active = self._active
            waiting = self._waiting
            ready = self._upstream_ready
        capacity = self.settings.worker.max_batch
        return {
            "inference": {
                "waiting": waiting,
                "prefilling": active,
                "decoding": 0,
            },
            "persistent_states": {
                # The adapter owns no serializable runtime State. Active work
                # is reported separately and therefore still blocks drain.
                "capacity": capacity,
                "allocated": 0,
                "free": max(0, capacity - active - waiting) if ready else 0,
                "busy": 0,
                "reconstructible": 0,
                "batching": {
                    "waiting": waiting,
                    "active_rows": active,
                    "max_batch_size": capacity,
                },
            },
        }

    def _probe_loop(self) -> None:
        while not self._stop.is_set():
            self.probe_once()
            self._probe_wake.wait(self.settings.health_interval_seconds)
            self._probe_wake.clear()


class OpenAIWorkerHandler(BaseHTTPRequestHandler):
    """Minimal streaming HTTP proxy bound to an `OpenAIWorkerRuntime`."""

    runtime: OpenAIWorkerRuntime
    server_version = "rwkv-openai-worker/0.1"
    protocol_version = "HTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path == "/live":
            self._json(HTTPStatus.OK, {"status": "alive"})
            return
        if path in {"/health", "/ready"}:
            status = self.runtime.status()
            ready = bool(status["upstream_ready"] and status["statepool"]["ready"])
            code = HTTPStatus.OK if path == "/health" or ready else HTTPStatus.SERVICE_UNAVAILABLE
            self._json(code, {"status": "ready" if ready else "starting", **status})
            return
        if path == "/v1/statepool/drain":
            self._json(HTTPStatus.OK, self.runtime.worker_agent.drain_status())
            return
        if path == "/v1/models":
            if not self._authorized():
                return
            self._proxy("GET", path, None)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path == "/v1/statepool/drain":
            try:
                payload = self._read_json()
                timeout = float(payload.get("timeout_seconds", 120))
                result = self.runtime.worker_agent.begin_draining(
                    timeout_seconds=timeout
                )
            except (ValueError, TypeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
                return
            self._json(HTTPStatus.OK, result)
            return
        if path not in PROXY_POST_PATHS:
            self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return
        if not self._authorized():
            return
        try:
            raw = self._read_body()
            normalized = self.runtime.normalize_payload(raw)
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            return
        self._proxy("POST", path, normalized)

    def _authorized(self) -> bool:
        if self.runtime.authorized(self.headers.get("Authorization", "")):
            return True
        self._json(
            HTTPStatus.UNAUTHORIZED,
            {"error": {"message": "invalid bearer token", "type": "authentication_error"}},
            extra_headers={"WWW-Authenticate": "Bearer"},
        )
        return False

    def _proxy(self, method: str, path: str, body: bytes | None) -> None:
        if not self.runtime.acquire():
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": {"message": "Worker is draining or unavailable"}},
            )
            return
        headers = self.runtime.upstream_headers()
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(
            self.runtime.settings.upstream_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            try:
                response = urlopen(
                    request,
                    timeout=self.runtime.settings.upstream_timeout_seconds,
                )
            except HTTPError as exc:
                response = exc
            with response:
                self.send_response(response.status)
                for name in ("Content-Type", "Cache-Control", "X-Request-Id"):
                    value = response.headers.get(name)
                    if value:
                        self.send_header(name, value)
                self.send_header(
                    "X-StatePool-Worker-Id",
                    self.runtime.settings.worker.worker_id,
                )
                # urllib de-chunks upstream responses. Closing the HTTP/1.0
                # response delimits streamed SSE without buffering it all.
                self.send_header("Connection", "close")
                self.end_headers()
                while chunk := response.read(64 * 1024):
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (URLError, TimeoutError, OSError) as exc:
            if not self.wfile.closed:
                self._json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": {"message": f"upstream unavailable: {exc}"}},
                )
        finally:
            self.runtime.release()
            self.close_connection = True

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 1:
            raise ValueError("request body is required")
        if length > self.runtime.settings.max_request_bytes:
            raise ValueError("request body exceeds configured limit")
        return self.rfile.read(length)

    def _read_json(self) -> dict[str, Any]:
        raw = self._read_body()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the standard server log, but include Worker identity for
        # multi-Worker nodes.
        print(f"[{self.runtime.settings.worker.worker_id}] {format % args}")


def make_server(
    settings: OpenAIWorkerSettings,
    *,
    statepool_transport: (
        Callable[[str, dict[str, Any], float], dict[str, Any]] | None
    ) = None,
) -> tuple[ThreadingHTTPServer, OpenAIWorkerRuntime]:
    runtime = OpenAIWorkerRuntime(
        settings,
        statepool_transport=statepool_transport,
    )
    handler = type(
        "BoundOpenAIWorkerHandler",
        (OpenAIWorkerHandler,),
        {"runtime": runtime},
    )
    server = ThreadingHTTPServer((settings.host, settings.port), handler)
    server.daemon_threads = True
    return server, runtime


def main() -> None:
    settings = OpenAIWorkerSettings.from_environment()
    server, runtime = make_server(settings)
    runtime.start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        runtime.stop()


def _positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise OpenAIWorkerConfigurationError(f"{name} must be positive")
    return parsed


def _positive_float(value: str, name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise OpenAIWorkerConfigurationError(f"{name} must be positive")
    return parsed


if __name__ == "__main__":
    main()
