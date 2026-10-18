#!/usr/bin/env python3
"""Generate a deterministic CycloneDX inventory without invoking Cargo locally.

Produce Cargo metadata on the approved remote build host, then pass the JSON to
this script. The script also inventories exact packages from ``uv.lock`` and
versioned Compose images. It is an inventory, not a vulnerability scan.
"""

from __future__ import annotations

import argparse
import json
import tomllib
import urllib.parse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def purl(kind: str, name: str, version: str) -> str:
    return f"pkg:{kind}/{urllib.parse.quote(name, safe='/')}@{urllib.parse.quote(version, safe='')}"


def cargo_components(metadata: dict) -> tuple[list[dict], list[str]]:
    components = []
    workspace = set(metadata["workspace_members"])
    workspace_refs = []
    for package in metadata["packages"]:
        ref = purl("cargo", package["name"], package["version"])
        component = {
            "type": "application" if package["id"] in workspace else "library",
            "bom-ref": ref,
            "name": package["name"],
            "version": package["version"],
            "purl": ref,
            "properties": [{"name": "statepool:ecosystem", "value": "cargo"}],
        }
        if license_expression := package.get("license"):
            component["licenses"] = [{"expression": license_expression}]
        if source := package.get("source"):
            component["properties"].append({"name": "cargo:source", "value": source})
        if package["id"] in workspace:
            workspace_refs.append(ref)
        components.append(component)
    return components, workspace_refs


def python_components() -> list[dict]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    components = []
    for package in lock.get("package", []):
        version = package.get("version")
        if not version:
            continue
        ref = purl("pypi", package["name"], version)
        component = {
            "type": "library",
            "bom-ref": ref,
            "name": package["name"],
            "version": version,
            "purl": ref,
            "properties": [{"name": "statepool:ecosystem", "value": "python"}],
        }
        if source := package.get("source"):
            component["properties"].append(
                {"name": "uv:source", "value": json.dumps(source, sort_keys=True)}
            )
        components.append(component)
    return components


def container_components() -> list[dict]:
    compose = yaml.safe_load(
        (ROOT / "deploy" / "statepool" / "compose.yaml").read_text(encoding="utf-8")
    )
    components = []
    for service_name, service in compose["services"].items():
        image = service.get("image")
        if not image or ":" not in image:
            continue
        repository, version = image.rsplit(":", 1)
        ref = purl("oci", repository, version)
        components.append(
            {
                "type": "container",
                "bom-ref": ref,
                "name": repository,
                "version": version,
                "purl": ref,
                "properties": [
                    {"name": "statepool:compose-service", "value": service_name},
                    {"name": "statepool:ecosystem", "value": "container"},
                    {
                        "name": "statepool:verification",
                        "value": "version-pinned; see COMPATIBILITY.md for runtime status",
                    },
                ],
            }
        )
    return components


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cargo_metadata", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--timestamp", default="2026-08-26T00:00:00Z")
    args = parser.parse_args()

    metadata = json.loads(args.cargo_metadata.read_text(encoding="utf-8"))
    cargo, workspace_refs = cargo_components(metadata)
    components = cargo + python_components() + container_components()
    unique = {component["bom-ref"]: component for component in components}
    components = [unique[key] for key in sorted(unique)]
    root_ref = purl("github", "123123213weqw/rwkv-agent", args.revision)
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": args.timestamp,
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": "rwkv-agent",
                "version": args.revision,
                "purl": root_ref,
                "licenses": [{"license": {"id": "MIT"}}],
            },
            "properties": [
                {
                    "name": "statepool:generator-note",
                    "value": "Cargo metadata generated on WZU_Server; Python from uv.lock; containers from Compose",
                }
            ],
        },
        "components": components,
        "dependencies": [{"ref": root_ref, "dependsOn": sorted(workspace_refs)}],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {len(components)} components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
