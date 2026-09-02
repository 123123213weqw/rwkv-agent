#!/usr/bin/env python3
"""Fail fast when the public release surface contains unsafe or broken data."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 development environments
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "0.3.0b2"
RUST_VERSION = "0.3.0-beta.2"
FROZEN_PUBLIC_RESULTS_RELEASE = "0.3.0-beta.1"

REQUIRED_FILES = (
    ".env.example",
    "CHANGELOG.md",
    "DEVELOPING.md",
    "README.md",
    "SECURITY.md",
    "bench/baselines/agent-unified-regression-v1/manifest.json",
    "bench/baselines/agent-unified-regression-v1/public-results-manifest.json",
    "configs/production.example.json",
    "Cargo.toml",
    "Cargo.lock",
    "crates/agent-cli/Cargo.toml",
    "crates/agent-cli/README.md",
    "crates/agent-core/Cargo.toml",
    "crates/agent-core/README.md",
    "crates/agent-runtime/Cargo.toml",
    "crates/agent-runtime/README.md",
    "crates/agent-server/Cargo.toml",
    "crates/agent-server/README.md",
    "docs/CONFIGURATION.md",
    "docs/CODEMAP.md",
    "docs/KNOWN_ISSUES.md",
    "docs/QUICKSTART.md",
    "docs/REPOSITORY_SURFACE.md",
    "docs/RELEASE.md",
    "scripts/dev",
)

EXECUTABLE_FILES = (
    "scripts/dev",
)

FORBIDDEN_LEGACY_FILES = (
    "src/rwkv_agent/controller.py",
    "src/rwkv_agent/server.py",
    "src/rwkv_agent/state_agent.py",
    "src/rwkv_search/api.py",
    "src/rwkv_search/cli.py",
    "src/rwkv_search/web/index.html",
    "cli/scripts/rwkv",
    "cli/scripts/rwkv-agent-service",
    "scripts/train_fitgen_lora.py",
    "benchmarks/run_fitgen_benchmark.py",
)

SCAN_FILES = (
    ".env.example",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "DEVELOPING.md",
    "README.md",
    "SECURITY.md",
)
SCAN_DIRS = (
    ".github",
    "bench/baselines",
    "benchmarks",
    "crates",
    "cli/tests",
    "cli/scripts",
    "configs",
    "deploy",
    "docs",
    "scripts",
    "src",
    "tests",
)
SCAN_EXCLUDES = {"docs/TODO.md"}
# Frozen benchmark evidence intentionally preserves the exact experiment
# workspace, including machine-local paths. It is integrity-checked by its own
# SHA manifests and must not be rewritten by the public-source hygiene audit.
SCAN_PREFIX_EXCLUDES = ("bench/baselines/long_horizon/",)
TEXT_SUFFIXES = {
    "",
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SECRET_PATTERNS = {
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "Tavily secret": re.compile(r"\btvly-[A-Za-z0-9_-]{12,}\b"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{16,}\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}

PRIVATE_HOME_PATTERN = re.compile(r"/(Users|home)/([^/\s`'\"]+)(?:/|$)")
IPV4_PATTERN = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HOME_PLACEHOLDERS = {"data", "user", "username", "your-user", "your_user", "$USER", "<user>"}


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _iter_public_text_files() -> list[Path]:
    paths: set[Path] = set()
    for relative in SCAN_FILES:
        path = ROOT / relative
        if path.is_file():
            paths.add(path)
    for relative in SCAN_DIRS:
        base = ROOT / relative
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            rel = _relative(path)
            if (
                rel in SCAN_EXCLUDES
                or rel.startswith(SCAN_PREFIX_EXCLUDES)
                or any(part.startswith(".") for part in path.relative_to(base).parts)
            ):
                continue
            if path.suffix.lower() in TEXT_SUFFIXES:
                paths.add(path)
    return sorted(paths)


def _check_versions(errors: list[str]) -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    actual_python = pyproject.get("project", {}).get("version")
    if actual_python != PYTHON_VERSION:
        _error(errors, f"pyproject version is {actual_python!r}; expected {PYTHON_VERSION!r}")

    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    actual_rust = cargo.get("workspace", {}).get("package", {}).get("version")
    if actual_rust != RUST_VERSION:
        _error(errors, f"Cargo workspace version is {actual_rust!r}; expected {RUST_VERSION!r}")

    lock = tomllib.loads((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    cli_packages = [p for p in lock.get("package", []) if p.get("name") == "rwkv-agent-cli"]
    if len(cli_packages) != 1 or cli_packages[0].get("version") != RUST_VERSION:
        _error(errors, "Cargo.lock does not contain exactly one rwkv-agent-cli at the release version")
    core_packages = [
        p for p in lock.get("package", []) if p.get("name") == "rwkv-agent-core"
    ]
    if len(core_packages) != 1 or core_packages[0].get("version") != RUST_VERSION:
        _error(
            errors,
            "Cargo.lock does not contain exactly one rwkv-agent-core at the release version",
        )
    for package_name, manifest_path in (
        ("rwkv-agent-cli", "crates/agent-cli/Cargo.toml"),
        ("rwkv-agent-core", "crates/agent-core/Cargo.toml"),
        ("rwkv-agent-runtime", "crates/agent-runtime/Cargo.toml"),
        ("rwkv-agent-server", "crates/agent-server/Cargo.toml"),
    ):
        manifest = tomllib.loads((ROOT / manifest_path).read_text(encoding="utf-8"))
        if manifest.get("package", {}).get("version", {}).get("workspace") is not True:
            _error(errors, f"{package_name} must inherit the Cargo workspace version")
        packages = [
            package
            for package in lock.get("package", [])
            if package.get("name") == package_name
        ]
        if len(packages) != 1 or packages[0].get("version") != RUST_VERSION:
            _error(
                errors,
                f"Cargo.lock does not contain exactly one {package_name} at the release version",
            )


def _check_files(errors: list[str]) -> None:
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            _error(errors, f"required release file is missing: {relative}")
    for relative in EXECUTABLE_FILES:
        path = ROOT / relative
        if path.exists() and not os.access(path, os.X_OK):
            _error(errors, f"release script is not executable: {relative}")
    for relative in FORBIDDEN_LEGACY_FILES:
        if (ROOT / relative).exists():
            _error(errors, f"removed legacy release path returned: {relative}")


def _check_config(errors: list[str]) -> None:
    config = json.loads((ROOT / "configs/production.example.json").read_text(encoding="utf-8"))
    realtime = config.get("realtime_search")
    if not isinstance(realtime, dict):
        _error(errors, "production example has no realtime_search object")
        return
    if realtime.get("enabled") is not True:
        _error(errors, "production example must enable realtime_search")
    if realtime.get("allow_private_networks") is not False:
        _error(errors, "production example must disable private-network fetching")
    if not realtime.get("api_discovery_providers") and not realtime.get("searxng_url"):
        _error(errors, "production example has no configured discovery source")


def _check_environment_template(errors: list[str]) -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    required = {
        "RWKV_AGENT_PROJECT_ROOT",
        "RWKV_AGENT_PYTHON",
        "G1I_MODEL_PATH",
        "G1I_RUNTIME_DIR",
        "RWKV_AGENT_HOST",
        "RWKV_AGENT_PORT",
        "RWKV_AGENT_MODEL_URLS",
        "RWKV_AGENT_DATA_PLANE_URL",
        "RWKV_AGENT_SESSION_DIR",
        "RWKV_AGENT_WEB_CONFIG",
    }
    for key in sorted(required - values.keys()):
        _error(errors, f"environment template is missing {key}")
    for key in (
        "RWKV_AGENT_PROJECT_ROOT",
        "RWKV_AGENT_PYTHON",
        "G1I_MODEL_PATH",
        "G1I_RUNTIME_DIR",
        "RWKV_AGENT_SESSION_DIR",
    ):
        if key in values and not values[key].startswith("/absolute/path/"):
            _error(errors, f"{key} must use the /absolute/path placeholder")
    if values.get("TAVILY_API_KEY") or values.get("GITHUB_TOKEN"):
        _error(errors, "optional API keys must be empty in .env.example")


def _check_benchmark_publication(errors: list[str]) -> None:
    base = ROOT / "bench/baselines/agent-unified-regression-v1"
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    publication = manifest.get("publication")
    if not isinstance(publication, dict):
        _error(errors, "unified regression manifest has no publication policy")
        return
    if publication.get("raw_cases_public") is not False:
        _error(errors, "unified regression raw cases must remain private")
    if publication.get("aggregate_results_only") is not True:
        _error(errors, "unified regression publication must be aggregate-results-only")
    safe = set(publication.get("safe_to_publish") or [])
    actual = {path.name for path in base.iterdir() if path.is_file() and not path.name.startswith(".")}
    if safe != actual:
        missing = sorted(actual - safe)
        absent = sorted(safe - actual)
        _error(
            errors,
            "unified regression publication list does not match directory "
            f"(unlisted={missing}, missing={absent})",
        )
    if (base / "cases.jsonl").exists():
        _error(errors, "license-restricted unified regression cases.jsonl must not be public")

    index_path = base / "index.jsonl"
    index_record = (manifest.get("files") or {}).get("index.jsonl") or {}
    index_data = index_path.read_bytes()
    index_rows = sum(1 for line in index_data.splitlines() if line.strip())
    if index_record.get("bytes") != len(index_data):
        _error(errors, "unified regression index byte count does not match its manifest")
    if index_record.get("rows") != index_rows:
        _error(errors, "unified regression index row count does not match its manifest")
    if index_record.get("sha256") != hashlib.sha256(index_data).hexdigest():
        _error(errors, "unified regression index SHA-256 does not match its manifest")

    result_manifest_path = base / "public-results-manifest.json"
    result_manifest = json.loads(result_manifest_path.read_text(encoding="utf-8"))
    if result_manifest.get("release") != FROZEN_PUBLIC_RESULTS_RELEASE:
        _error(
            errors,
            "public result manifest does not match frozen release "
            f"{FROZEN_PUBLIC_RESULTS_RELEASE}",
        )
    for name, record in (result_manifest.get("artifacts") or {}).items():
        path = base / name
        if not path.is_file():
            _error(errors, f"public result artifact is missing: {name}")
            continue
        data = path.read_bytes()
        if record.get("bytes") != len(data):
            _error(errors, f"public result byte count mismatch: {name}")
        if record.get("sha256") != hashlib.sha256(data).hexdigest():
            _error(errors, f"public result SHA-256 mismatch: {name}")


def _check_provider_entrypoints(errors: list[str]) -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject.get("project", {}).get("scripts", {})
    expected = {
        "rwkv-agent-data-plane": "rwkv_agent.data_server:main",
        "rwkv-g1i-sidecar": "rwkv_agent.sidecar:main",
        "rwkv-statepool-drain": "rwkv_agent.statepool_drain:main",
    }
    if scripts != expected:
        _error(
            errors,
            "Python release entrypoints must contain only the narrow Sidecar, "
            f"Data Plane and native StatePool drain helpers; got {scripts!r}",
        )


def _check_markdown_links(errors: list[str]) -> None:
    for path in _iter_public_text_files():
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK_PATTERN.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (path.parent / target).resolve().exists():
                _error(errors, f"{_relative(path)}: broken local Markdown link: {raw_target}")


def _check_public_text(errors: list[str]) -> int:
    scanned = 0
    for path in _iter_public_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        relative = _relative(path)
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                _error(errors, f"{relative}: contains a {label}")
        for match in PRIVATE_HOME_PATTERN.finditer(text):
            username = match.group(2)
            if username.lower() not in HOME_PLACEHOLDERS and not username.startswith("${"):
                _error(errors, f"{relative}: contains a user-specific home path: {match.group(0)}")
        for candidate in IPV4_PATTERN.findall(text):
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                continue
            if address.version == 4 and address in ipaddress.ip_network("100.64.0.0/10"):
                if relative == "scripts/check_public_release.py" and candidate == "100.64.0.0":
                    continue
                _error(errors, f"{relative}: contains a Tailscale CGNAT address: {address}")
    return scanned


def main() -> int:
    errors: list[str] = []
    _check_files(errors)
    if not errors:
        _check_versions(errors)
        _check_config(errors)
        _check_environment_template(errors)
        _check_benchmark_publication(errors)
        _check_provider_entrypoints(errors)
        _check_markdown_links(errors)
    scanned = _check_public_text(errors)

    if errors:
        print("Public release audit failed:", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
        return 1
    print(
        f"Public release audit passed: Python {PYTHON_VERSION}, Rust {RUST_VERSION}, "
        f"{scanned} public text files scanned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
