#!/usr/bin/env python3
"""Static checks for StatePool deployment assets (no containers are started)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy" / "statepool"
CHART = DEPLOY / "helm" / "statepool"


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path} must contain a YAML object")
    return value


def main() -> int:
    compose = load_yaml(DEPLOY / "compose.yaml")
    services = compose.get("services", {})
    required = {
        "statepool",
        "statepool-cloud-lite",
        "agent",
        "prometheus",
        "grafana",
        "postgres",
        "minio",
        "minio-init",
    }
    if missing := required.difference(services):
        raise AssertionError(f"Compose services missing: {sorted(missing)}")
    for name, service in services.items():
        image = service.get("image", "")
        if image.endswith(":latest") or image == "latest":
            raise AssertionError(f"{name} uses a mutable latest image")
    for name in ("statepool-cloud-lite", "postgres", "minio", "minio-init"):
        if "cloud-lite" not in services[name].get("profiles", []):
            raise AssertionError(f"{name} must remain behind the cloud-lite profile")

    values = load_yaml(CHART / "values.yaml")
    if values["plugin"]["replicaCount"] != 1:
        raise AssertionError("In-memory Lease profile must remain single replica")
    if values["plugin"]["durable"]["enabled"]:
        raise AssertionError("PostgreSQL/S3 must remain opt-in")
    if values["worker"]["enabled"] or values["autoscaling"]["enabled"]:
        raise AssertionError("Worker and KEDA gates must default to disabled")
    if values["autoscaling"]["minReplicaCount"] != 0:
        raise AssertionError("KEDA profile must preserve scale-to-zero")

    keda = (CHART / "templates" / "keda-scaledobject.yaml").read_text(encoding="utf-8")
    for metric in ("statepool_pending_requests", "statepool_estimated_decode_seconds"):
        if metric not in keda:
            raise AssertionError(f"KEDA template missing metric {metric}")
    worker = (CHART / "templates" / "worker-deployment.yaml").read_text(encoding="utf-8")
    for marker in (
        "preStop",
        "rwkv-statepool-drain",
        "terminationGracePeriodSeconds",
        "RWKV_WORKER_HEARTBEAT_SECONDS",
        "POD_UID",
    ):
        if marker not in worker:
            raise AssertionError(f"Worker lifecycle template missing {marker}")

    dashboard = json.loads(
        (DEPLOY / "grafana" / "dashboards" / "statepool-overview.json").read_text(
            encoding="utf-8"
        )
    )
    chart_dashboard = json.loads(
        (CHART / "files" / "statepool-overview.json").read_text(encoding="utf-8")
    )
    if dashboard != chart_dashboard:
        raise AssertionError("Compose and Helm dashboard copies differ")
    if len(dashboard.get("panels", [])) < 8:
        raise AssertionError("StatePool dashboard is missing required panels")

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    if not dockerignore or dockerignore[0] != "**":
        raise AssertionError("Docker context must default-deny repository files")
    for required_path in ("!Cargo.toml", "!Cargo.lock", "!crates/**", "!web/**"):
        if required_path not in dockerignore:
            raise AssertionError(f"Docker build allowlist missing {required_path}")

    sbom = json.loads((ROOT / "sbom" / "statepool.cdx.json").read_text(encoding="utf-8"))
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") != "1.5":
        raise AssertionError("StatePool SBOM must remain CycloneDX 1.5")
    if len(sbom.get("components", [])) < 100:
        raise AssertionError("StatePool SBOM is unexpectedly incomplete")
    if any(component.get("version") == "latest" for component in sbom["components"]):
        raise AssertionError("StatePool SBOM contains a mutable latest version")

    print(
        f"StatePool deploy assets valid: {len(services)} Compose services, "
        f"{len(dashboard['panels'])} dashboard panels, "
        f"{len(sbom['components'])} SBOM components"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
