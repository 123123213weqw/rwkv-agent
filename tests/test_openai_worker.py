from __future__ import annotations

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from rwkv_agent.openai_worker import (
    AFFINITY_ONLY_STATE_CAPABILITY,
    OpenAIWorkerConfigurationError,
    OpenAIWorkerRuntime,
    OpenAIWorkerSettings,
    make_server,
)
from rwkv_agent.statepool_worker import WorkerSettings


MODEL_REF = {
    "model_id": "qwen-certified",
    "revision": "sha256:model-revision",
    "tokenizer": "qwen-tokenizer-revision",
    "state_abi": "context-replay.v1",
}


def settings(**changes) -> OpenAIWorkerSettings:
    worker = WorkerSettings(
        statepool_url="http://statepool.test",
        worker_id="vllm-worker-a",
        endpoint="http://worker.test:8128",
        zone="cloud",
        model_ref=MODEL_REF,
        state_slots=4,
        max_batch=2,
        state_capability=dict(AFFINITY_ONLY_STATE_CAPABILITY),
    )
    base = OpenAIWorkerSettings(
        worker=worker,
        upstream_url="http://upstream.test:8000",
        upstream_model="Qwen/Qwen3.5-9B",
    )
    return replace(base, **changes)


def test_environment_builds_affinity_only_worker(monkeypatch) -> None:
    values = {
        "RWKV_STATEPOOL_URL": "http://statepool:8130",
        "RWKV_WORKER_ID": "vllm-a",
        "RWKV_WORKER_MODEL_ID": MODEL_REF["model_id"],
        "RWKV_WORKER_MODEL_REVISION": MODEL_REF["revision"],
        "RWKV_WORKER_TOKENIZER": MODEL_REF["tokenizer"],
        "RWKV_WORKER_STATE_ABI": MODEL_REF["state_abi"],
        "RWKV_OPENAI_UPSTREAM_URL": "http://vllm:8000",
        "RWKV_OPENAI_UPSTREAM_MODEL": "Qwen/Qwen3.5-9B",
        "RWKV_OPENAI_WORKER_PORT": "8128",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    parsed = OpenAIWorkerSettings.from_environment()

    assert parsed.worker.state_capability == AFFINITY_ONLY_STATE_CAPABILITY
    assert parsed.worker.endpoint == "http://127.0.0.1:8128"
    assert parsed.upstream_model == "Qwen/Qwen3.5-9B"


def test_environment_rejects_non_http_upstream(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_STATEPOOL_URL", "http://statepool:8130")
    monkeypatch.setenv("RWKV_WORKER_MODEL_ID", MODEL_REF["model_id"])
    monkeypatch.setenv("RWKV_WORKER_MODEL_REVISION", MODEL_REF["revision"])
    monkeypatch.setenv("RWKV_WORKER_TOKENIZER", MODEL_REF["tokenizer"])
    monkeypatch.setenv("RWKV_WORKER_STATE_ABI", MODEL_REF["state_abi"])
    monkeypatch.setenv("RWKV_OPENAI_UPSTREAM_URL", "file:///tmp/socket")
    with pytest.raises(OpenAIWorkerConfigurationError, match="HTTP"):
        OpenAIWorkerSettings.from_environment()


def test_payload_maps_logical_model_to_upstream_and_rejects_others() -> None:
    runtime = OpenAIWorkerRuntime(settings())
    value = json.loads(
        runtime.normalize_payload(
            json.dumps({"model": "qwen-certified", "messages": []}).encode()
        )
    )
    assert value["model"] == "Qwen/Qwen3.5-9B"
    with pytest.raises(ValueError, match="not served"):
        runtime.normalize_payload(
            json.dumps({"model": "unregistered-model", "messages": []}).encode()
        )


def test_affinity_worker_has_no_serializable_state_and_drains_on_active_work() -> None:
    runtime = OpenAIWorkerRuntime(settings())
    runtime.worker_agent.set_available(True)
    capability = runtime.worker_agent.capability()
    assert capability["state_capability"] == AFFINITY_ONLY_STATE_CAPABILITY
    assert capability["capacity"]["unpersisted_state_slots"] == 0

    assert runtime.acquire()
    assert runtime.worker_agent.drain_status(deadline_ms=10**15) == {
        "contract_version": "statepool-drain-status.v1",
        "worker_id": "vllm-worker-a",
        "status": "draining",
        "active_requests": 1,
        "unpersisted_states": 0,
    }
    runtime.release()
    assert runtime.worker_agent.drain_status(deadline_ms=10**15)["status"] == (
        "safe_to_stop"
    )


class _HealthHandler(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        value = json.loads(self.rfile.read(length))
        self.received.append(value)
        body = json.dumps({"id": "result", "model": value["model"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args) -> None:
        pass


def test_probe_controls_worker_availability() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runtime = OpenAIWorkerRuntime(
        settings(upstream_url=f"http://127.0.0.1:{server.server_port}")
    )
    try:
        assert runtime.probe_once()
        assert runtime.status()["upstream_ready"] is True
        assert runtime.worker_agent.capability()["lifecycle"] == "ready"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_adapter_proxies_logical_model_and_drain_closes_admission() -> None:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    statepool_calls: list[str] = []

    def statepool_transport(url: str, payload: dict, _timeout: float) -> dict:
        statepool_calls.append(url)
        return {"status": "ok", "lifecycle": payload.get("lifecycle", "ready")}

    adapter_settings = settings(
        upstream_url=f"http://127.0.0.1:{upstream.server_port}",
        host="127.0.0.1",
        port=0,
        health_interval_seconds=0.05,
    )
    adapter, runtime = make_server(
        adapter_settings,
        statepool_transport=statepool_transport,
    )
    adapter_thread = threading.Thread(target=adapter.serve_forever, daemon=True)
    runtime.start()
    adapter_thread.start()
    base = f"http://127.0.0.1:{adapter.server_port}"
    try:
        deadline = time.time() + 2
        while not runtime.worker_agent.ready() and time.time() < deadline:
            time.sleep(0.01)
        assert runtime.worker_agent.ready()

        body = json.dumps(
            {"model": "qwen-certified", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        request = Request(
            base + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            assert response.headers["X-StatePool-Worker-Id"] == "vllm-worker-a"
            result = json.loads(response.read())
        assert result["model"] == "Qwen/Qwen3.5-9B"
        assert _HealthHandler.received[-1]["model"] == "Qwen/Qwen3.5-9B"

        drain = Request(
            base + "/v1/statepool/drain",
            data=b'{"timeout_seconds":10}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(drain, timeout=2) as response:
            assert json.loads(response.read())["status"] == "safe_to_stop"
        with pytest.raises(HTTPError) as rejected:
            urlopen(request, timeout=2)
        assert rejected.value.code == 503
        assert any(url.endswith("/workers/register") for url in statepool_calls)
        assert any(url.endswith("/vllm-worker-a/drain") for url in statepool_calls)
    finally:
        adapter.shutdown()
        adapter.server_close()
        runtime.stop()
        upstream.shutdown()
        upstream.server_close()
        adapter_thread.join(timeout=2)
        upstream_thread.join(timeout=2)
