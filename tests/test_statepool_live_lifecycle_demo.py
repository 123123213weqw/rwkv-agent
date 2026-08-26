from __future__ import annotations

import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from scripts import statepool_live_lifecycle_demo as lifecycle


run = lifecycle.run


MODEL_REF = {
    "model_id": "rwkv-test",
    "revision": "revision-test",
    "tokenizer": "tokenizer-test",
    "state_abi": "state-abi-test",
}
PAYLOAD = b"safe-rwkv-state"
CHECKSUM = "sha256:" + hashlib.sha256(PAYLOAD).hexdigest()


class Handler(BaseHTTPRequestHandler):
    fencing = 0
    events: list[str] = []

    def log_message(self, _format, *_args):
        return

    def send_json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ready", "model_ref": MODEL_REF})
            return
        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        self.events.append(self.path)
        if self.path == "/plugin/v1/leases/acquire":
            type(self).fencing += 1
            self.send_json(
                {
                    "contract_version": "statepool-lease.v1",
                    "lease_id": f"lease-{self.fencing}",
                    "session_id": request["session_id"],
                    "owner_id": request["owner_id"],
                    "holder_id": request["holder_id"],
                    "fencing_token": self.fencing,
                    "expected_state_version": request["expected_state_version"],
                    "expires_at_ms": 9_999_999_999_999,
                }
            )
        elif self.path == "/plugin/v1/leases/release":
            self.send_response(204)
            self.end_headers()
        elif self.path == "/v1/states/prefill":
            self.send_json({"state": {"state_id": "source-state", "seen_tokens": 5}})
        elif self.path == "/v1/states/batch_continue":
            state_id = request["items"][0]["state_id"]
            self.send_json(
                {
                    "results": [
                        {
                            "state_id": state_id,
                            "text": "continued",
                            "token_ids": [1, 2],
                            "seen_tokens": 7,
                        }
                    ]
                }
            )
        elif self.path == "/v1/states/source-state/snapshot":
            self.send_json(
                {
                    "checkpoint": {"checksum": CHECKSUM},
                    "payload_base64": base64.b64encode(PAYLOAD).decode(),
                }
            )
        elif self.path == "/plugin/v1/states/snapshot":
            self.send_json(
                {
                    "contract_version": "statepool-state-reference.v1",
                    "state_id": "state-v1",
                    "session_id": request["lease"]["session_id"],
                    "owner_id": request["lease"]["owner_id"],
                    "version": 1,
                    "fencing_token": 1,
                    "provider_mode": "rwkv_recurrent",
                    "model_ref": MODEL_REF,
                    "placement": "cold",
                    "worker_id": None,
                    "object_uri": "file:///state-v1",
                    "checksum": CHECKSUM,
                    "size_bytes": len(PAYLOAD),
                    "atomic": True,
                    "created_at_ms": 1,
                    "last_active_at_ms": 1,
                    "encryption": None,
                }
            )
        elif self.path == "/plugin/v1/states/restore":
            self.send_json(
                {
                    "contract_version": "statepool-restore-response.v1",
                    "state_ref": request["state_ref"],
                    "payload_base64": base64.b64encode(PAYLOAD).decode(),
                }
            )
        elif self.path == "/v1/states/restore":
            assert request["checksum"] == CHECKSUM
            self.send_json({"state": {"state_id": "target-state", "seen_tokens": 7}})
        elif self.path == "/v1/states/release":
            self.send_json({"released": len(request["state_ids"])})
        else:
            self.send_json({"error": "not found"}, 404)


def test_live_lifecycle_driver_composes_worker_and_plugin_contracts():
    Handler.fencing = 0
    Handler.events = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        result = run(
            argparse.Namespace(
                plugin_url=url,
                source_worker_url=url,
                target_worker_url=url,
                source_worker_id="source",
                target_worker_id="target",
                session_id="session-1",
                owner_id="owner-1",
                prompt="prompt",
                before_input="before",
                after_input="after",
                max_tokens=4,
                lease_ttl_ms=30_000,
                target_tier="cold",
                source_stop_command=None,
            )
        )
    finally:
        server.shutdown()
        thread.join()
    assert result["status"] == "passed"
    assert result["source_transition"] == "released"
    assert result["source_state_id"] == "source-state"
    assert result["target_state_id"] == "target-state"
    assert Handler.fencing == 2
    assert Handler.events.index("/plugin/v1/states/snapshot") < Handler.events.index(
        "/v1/states/release"
    )


def test_live_lifecycle_can_replace_worker_on_one_reused_endpoint(monkeypatch):
    Handler.fencing = 0
    Handler.events = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    observed: dict[str, object] = {}

    class FakeProcess:
        pid = 4242

        @staticmethod
        def poll():
            return None

    def fake_popen(command, *, start_new_session):
        observed["command"] = command
        observed["start_new_session"] = start_new_session
        return FakeProcess()

    def fake_wait_for_down(worker_url, *, timeout_seconds):
        observed["down"] = (worker_url, timeout_seconds)

    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        lifecycle.subprocess,
        "run",
        lambda command, *, check: observed.update(
            {"stop_command": command, "stop_check": check}
        ),
    )
    monkeypatch.setattr(lifecycle, "wait_for_worker_down", fake_wait_for_down)
    try:
        result = run(
            argparse.Namespace(
                plugin_url=url,
                source_worker_url=url,
                target_worker_url=url,
                source_worker_id="source",
                target_worker_id="target",
                session_id="session-restart",
                owner_id="owner-restart",
                prompt="prompt",
                before_input="before",
                after_input="after",
                max_tokens=4,
                lease_ttl_ms=30_000,
                target_tier="cold",
                source_stop_command="/usr/bin/true",
                target_start_command="/opt/demo/start-worker target",
                target_ready_timeout_seconds=5,
                source_down_timeout_seconds=3,
            )
        )
    finally:
        server.shutdown()
        thread.join()

    assert result["status"] == "passed"
    assert result["source_transition"] == "forcibly_stopped"
    assert result["target_worker_started_by_driver"] is True
    assert result["target_worker_process_pid"] == 4242
    assert observed == {
        "down": (url, 3.0),
        "stop_command": ["/usr/bin/true"],
        "stop_check": True,
        "command": ["/opt/demo/start-worker", "target"],
        "start_new_session": True,
    }
