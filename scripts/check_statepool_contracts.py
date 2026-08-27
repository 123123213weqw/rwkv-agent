#!/usr/bin/env python3
"""Validate StatePool JSON Schemas and their checked-in examples.

Run without changing the project environment:

    uv run --with jsonschema python scripts/check_statepool_contracts.py

The registry is explicit because the repository uses stable, non-routable
``https://rwkv-agent.local/contracts/...`` schema identifiers. Validation must
never try to resolve those identifiers over the network.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"

SCHEMA_FILES = (
    "stateful-inference-session-v1.schema.json",
    "plugin-v1.schema.json",
    "worker-capability-v1.schema.json",
    "state-reference-v1.schema.json",
    "execution-plan-v1.schema.json",
    "usage-record-v1.schema.json",
    "state-lifecycle-v1.schema.json",
)

EXAMPLES = {
    "handshake-request.json": "plugin-v1.schema.json",
    "handshake-response.json": "plugin-v1.schema.json",
    "worker.json": "worker-capability-v1.schema.json",
    "state-reference.json": "state-reference-v1.schema.json",
    "plan-request.json": "execution-plan-v1.schema.json",
    "execution-plan.json": "execution-plan-v1.schema.json",
    "usage-record.json": "usage-record-v1.schema.json",
    "acquire-lease-request.json": "state-lifecycle-v1.schema.json",
    "lease.json": "state-lifecycle-v1.schema.json",
    "renew-lease-request.json": "state-lifecycle-v1.schema.json",
    "release-lease-request.json": "state-lifecycle-v1.schema.json",
    "snapshot-request.json": "state-lifecycle-v1.schema.json",
    "restore-request.json": "state-lifecycle-v1.schema.json",
    "restore-response.json": "state-lifecycle-v1.schema.json",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    schemas = {name: load_json(CONTRACTS / name) for name in SCHEMA_FILES}
    registry = Registry().with_resources(
        (
            schema["$id"],
            Resource.from_contents(schema),
        )
        for schema in schemas.values()
    )
    checker = FormatChecker()

    for name, schema in schemas.items():
        Draft202012Validator.check_schema(schema)
        if "$id" not in schema:
            raise AssertionError(f"{name} has no stable $id")

    example_root = CONTRACTS / "examples" / "statepool-plugin-v1"
    for example_name, schema_name in EXAMPLES.items():
        instance = load_json(example_root / example_name)
        validator = Draft202012Validator(
            schemas[schema_name],
            registry=registry,
            format_checker=checker,
        )
        validator.validate(instance)

    openapi = load_json(CONTRACTS / "statepool-plugin-v1.openapi.json")
    if openapi.get("openapi") != "3.1.0":
        raise AssertionError("StatePool OpenAPI document must remain on OpenAPI 3.1.0")
    required_paths = {
        "/plugin/v1/handshake",
        "/plugin/v1/health",
        "/plugin/v1/plan",
        "/plugin/v1/usage",
        "/plugin/v1/workers/register",
        "/plugin/v1/states/snapshot",
        "/plugin/v1/states/restore",
        "/plugin/v1/leases/acquire",
        "/plugin/v1/leases/renew",
        "/plugin/v1/leases/release",
    }
    missing = required_paths.difference(openapi.get("paths", {}))
    if missing:
        raise AssertionError(f"OpenAPI document is missing paths: {sorted(missing)}")

    print(
        f"StatePool contracts valid: {len(schemas)} schemas, "
        f"{len(EXAMPLES)} examples, {len(openapi['paths'])} API paths"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
