# Repository surface

This document records the current executable ownership after the
repository-surface reduction. Project status remains authoritative in the
owner-local `docs/TODO.md`.

## Active

- Rust control plane: `crates/agent-{core,runtime,server,cli}/`;
- recurrent scheduling and lifecycle harness: `crates/state-runtime/`;
- narrow RWKV Provider and native StatePool Worker hooks:
  `src/rwkv_agent/sidecar.py`, `src/rwkv_agent/statepool_worker.py`,
  `src/rwkv_agent/statepool_drain.py`, `src/rwkv7_scheduler/` and
  `src/rwkv_runtime/`;
- narrow Retrieval/Evidence Provider: `src/rwkv_agent/data_server.py`,
  `src/rwkv_agent/data_plane.py`, its adapters and the reachable
  `src/rwkv_search/` library;
- current generic benchmark runners and immutable reviewed evidence.

## Removed from the executable release surface

- Python Agent Controller, HTTP Server, chat-State cache, memory store, Tool
  router and parallel-State Agent loop;
- old `rwkv-search` application CLI/API, SSE protocol, packaged Web preview,
  answerer and shadow-search product layer;
- automatic `rwkv` / `rwkv-agent-service` wrappers and their deployment
  shims;
- FitGen training/merge/serve tools, policy-curriculum generators and
  model-specific one-off benchmark/replay runners;
- contracts and documentation used only by those removed entrypoints.

The Rust Server retains compatibility HTTP aliases, but every alias invokes the
same Rust handler and lifecycle.

## Evidence retained

`bench/baselines/` and `bench/artifacts/` were not rewritten. A content
audit found duplicate files inside historical checksummed bundles, but those
duplicates are part of their frozen evidence layout. Qwen comparison source
records, result artifacts, README figures, failure records and SHA manifests
remain immutable.

Runtime caches are already ignored: `.venv/`, `target/`, `var/`,
`data/`, `bench/runs/`, Python bytecode, logs and PID files. Local
`.venv/` size is therefore not counted as repository reduction.

## Guardrails

`scripts/check_public_release.py` rejects reintroduction of representative
legacy entrypoints and requires the Python wheel to expose only:

- `rwkv-g1i-sidecar`;
- `rwkv-agent-data-plane`;
- `rwkv-statepool-drain`.

Any further removal from the active Provider graph requires a separate Rust
parity gate. Historical evidence is retrieved from Git when needed rather than
kept executable in the current tree.
