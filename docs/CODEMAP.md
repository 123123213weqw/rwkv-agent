# RWKV Agent code map

This file is the authoritative map from a product capability to its active
implementation. Project status remains authoritative in `docs/TODO.md`; this
map answers only where code lives.

## Start here

| Capability | Status | Canonical location |
|---|---|---|
| Terminal client | Active | `crates/agent-cli/` |
| Strict Tool/Answer protocol | Active | `crates/agent-core/src/protocol.rs` |
| Tool definitions and argument validation | Active | `crates/agent-core/src/registry.rs` |
| Bounded same-State Agent loop | Active | `crates/agent-core/src/run_loop.rs` |
| Agent service and routing | Active | `crates/agent-runtime/src/service.rs` |
| RWKV Sidecar State client | Active | `crates/agent-runtime/src/sidecar.rs` |
| Session transcript and State cache | Active | `crates/agent-runtime/src/session.rs` and `service.rs` |
| Parallel-State research | Active | `crates/agent-runtime/src/research.rs` |
| Command sandbox policy | Active, opt-in | `crates/agent-runtime/src/command.rs` |
| Rust HTTP control plane | Active, isolated default | `crates/agent-server/` |
| Retrieval/Evidence data plane | Active | `src/rwkv_agent/data_plane.py` |
| Python data-plane HTTP process | Active | `src/rwkv_agent/data_server.py` |
| Python compatibility Controller | Compatibility | `src/rwkv_agent/controller.py` |
| CUDA Sidecar | Active | `src/rwkv_agent/sidecar.py` |
| Unified recurrent scheduler | Active | `src/rwkv7_scheduler/` |
| Shared inference contracts | Active | `src/rwkv_runtime/` |
| Web and knowledge retrieval | Active dependency | `src/rwkv_search/` |
| Client installation and service lifecycle | Compatibility packaging | `cli/` |
| Legacy Web preview | Legacy/compatibility | `src/rwkv_search/web/` |

## Request flow

```text
crates/agent-cli
    -> crates/agent-server
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
Session serialization, recurrent State identity and release, tool sequencing,
research branches, cancellation and the sandbox boundary. It does not import
Torch, CUDA, crawlers or indexes.

### Python data plane

Python owns model/CUDA integration, Web discovery and extraction, local
knowledge, pasted long text, Evidence admission/reduction and claim validation.
It does not own the Rust Agent State machine.

## Repository data

- `benchmarks/`: reproducible benchmark runners and public fixtures.
- `bench/baselines/`: reviewed aggregate results safe for publication.
- ignored `bench/runs/`, `var/` and `data/`: raw or runtime-only artifacts.
- `docs/TODO.md`: local project status and execution record; intentionally not
  part of the public release surface.
