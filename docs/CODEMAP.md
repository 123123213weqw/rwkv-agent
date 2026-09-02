# RWKV Agent code map

This file is the authoritative map from a product capability to its active
implementation. Project status remains authoritative in `docs/TODO.md`; this
map answers only where code lives.

## Start here

| Capability | Status | Canonical location |
|---|---|---|
| Terminal client | Active | `crates/agent-cli/` |
| Strict Tool/Answer protocol | Active | `crates/agent-core/src/protocol.rs` |
| Service v1 HTTP/frontend contract | Active | `docs/HTTP_API.md`, `contracts/agent-service-v1.openapi.json`, `contracts/agent-service-v1.schema.json` and `crates/agent-core/src/service_contract.rs` |
| Tool definitions and argument validation | Active | `crates/agent-core/src/registry.rs` |
| Bounded same-State Agent loop | Active | `crates/agent-core/src/run_loop.rs` |
| Agent service and routing | Active | `crates/agent-runtime/src/service.rs` |
| RWKV Sidecar State client | Active | `crates/agent-runtime/src/sidecar.rs` |
| Session transcript and State cache | Active | `crates/agent-runtime/src/session.rs` and `service.rs` |
| Parallel-State research | Active | `crates/agent-runtime/src/research.rs` |
| Command sandbox policy | Active, opt-in | `crates/agent-runtime/src/command.rs` |
| Rust HTTP control plane | Active, isolated default | `crates/agent-server/` |
| Durable Task Ledger v2 | Active | `crates/agent-runtime/src/task_ledger.rs` |
| Local Rust Debug Trace v1 | Active, opt-in/off by default | `crates/agent-runtime/src/debug_trace.rs`, `contracts/debug-trace-v1.schema.json`, and `docs/DEBUG_TRACE.md` |
| Retrieval/Evidence data plane | Active | `src/rwkv_agent/data_plane.py` |
| Python data-plane HTTP process | Active | `src/rwkv_agent/data_server.py` |
| CUDA Sidecar | Active | `src/rwkv_agent/sidecar.py` |
| Native StatePool Worker registration/drain | Active, opt-in | `src/rwkv_agent/statepool_worker.py`, `src/rwkv_agent/statepool_drain.py` and `docs/STATEPOOL_CLOUD_PLUGIN.md` |
| Unified recurrent scheduler | Active | `src/rwkv7_scheduler/` |
| Shared inference contracts | Active | `src/rwkv_runtime/` |
| Web and knowledge retrieval | Active dependency | `src/rwkv_search/` |
| Client installation and release packaging | Active | `cli/` |

## Request flow

```text
crates/agent-cli
    -> crates/agent-server (/v1/tasks)
        -> crates/agent-runtime
            -> crates/agent-core
            -> RWKV CUDA Sidecar
            -> src/rwkv_agent data plane
                -> src/rwkv_search
                -> Evidence and claim validation
```

## Ownership boundary

### Rust control plane

Rust owns request budgets, strict protocol parsing, exact argument validation,
Service request identity, TaskSpec conversion, Task Ledger, Session
serialization, recurrent State identity and release, tool sequencing,
research branches, cancellation and the sandbox boundary. It does not import
Torch, CUDA, crawlers or indexes.

### Python data plane

Python owns model/CUDA integration, Web discovery and extraction, local
knowledge, pasted long text, Evidence admission/reduction and claim validation.
It does not own the Rust Agent State machine.

## Removed compatibility surface

The Python Controller/Server, Python Agent state loop, old `rwkv-search` HTTP
product, packaged Web preview, automatic service wrapper and training-only
FitGen utilities are not release entrypoints. Frozen benchmark results remain
under `bench/baselines/`; their historical source revisions remain available
from Git rather than being kept executable in the current tree.

## Repository data

- `benchmarks/`: reproducible benchmark runners and public fixtures.
- `bench/baselines/`: reviewed aggregate results safe for publication.
- ignored `bench/runs/`, `var/` and `data/`: raw or runtime-only artifacts.
- `docs/TODO.md`: local project status and execution record; intentionally not
  part of the public release surface.
- `docs/SERVICE_PIPELINE.md`: canonical endpoints, startup order and failure
  diagnosis for the isolated Rust service.
- `docs/DEBUG_TRACE.md`: local full/redacted diagnostic capture, owner APIs,
  privacy boundary, checksum verification and cleanup.
