from __future__ import annotations

import json
import subprocess
import sys
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
