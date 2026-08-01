from __future__ import annotations

import json
import os
import threading
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rwkv_search.config import AppConfig


ROOT = Path(__file__).resolve().parents[1]


def test_public_release_audit_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_public_release.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Public release audit passed" in result.stdout


def test_production_example_loads_and_uses_safe_web_defaults() -> None:
    path = ROOT / "configs/production.example.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = AppConfig.load(path)

    assert raw["realtime_search"]["enabled"] is True
    assert raw["realtime_search"]["allow_private_networks"] is False
    assert config.realtime_search.enabled is True
    assert config.realtime_search.allow_private_networks is False
    assert config.realtime_search.api_discovery_providers


def test_unconfigured_service_doctor_reports_all_required_paths(tmp_path: Path) -> None:
    script = ROOT / "cli/scripts/rwkv-agent-service"
    env = os.environ | {
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "RWKV_AGENT_PROJECT_ROOT": str(ROOT),
    }
    init_result = subprocess.run(
        [script, "init"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert init_result.returncode == 0, init_result.stderr

    doctor = subprocess.run(
        [script, "doctor"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    output = doctor.stdout + doctor.stderr
    assert doctor.returncode != 0
    assert "python is not executable" in output
    assert "invalid RWKV_AGENT_PROJECT_ROOT" in output
    assert "set G1I_MODEL_PATH" in output
    assert "set G1I_RUNTIME_DIR" in output
    assert "missing web config" in output


def test_service_refuses_to_signal_pid_with_wrong_command(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "controller-8120.pid").write_text(str(os.getpid()), encoding="utf-8")
    env = os.environ | {
        "RWKV_AGENT_ENV_FILE": str(tmp_path / "missing.env"),
        "RWKV_AGENT_PROJECT_ROOT": str(ROOT),
        "RWKV_AGENT_STATE_DIR": str(state),
    }

    result = subprocess.run(
        [ROOT / "cli/scripts/rwkv-agent-service", "stop"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing to signal it" in result.stderr
    os.kill(os.getpid(), 0)


def test_rwkv_launcher_tolerates_remote_tunnel_health_latency(tmp_path: Path) -> None:
    class SlowHealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            time.sleep(2.2)
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    service = tmp_path / "must-not-start"
    service.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    service.chmod(0o755)
    endpoint = f"http://127.0.0.1:{server.server_port}"
    env = os.environ | {
        "RWKV_AGENT_ENDPOINT": endpoint,
        "RWKV_AGENT_SERVICE_COMMAND": str(service),
        "RWKV_AGENT_CLIENT_COMMAND": "/bin/echo",
    }
    try:
        result = subprocess.run(
            [ROOT / "cli/scripts/rwkv", "health"],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"--endpoint {endpoint} health"
    assert "backend is offline" not in result.stderr
