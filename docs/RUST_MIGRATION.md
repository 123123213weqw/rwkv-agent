# Rust-only migration

The target repository is Rust-only for executable runtime, harness, benchmark,
server, CLI, provider adapters, migrations, tests and the future WASM UI.
Current Python/JavaScript/Shell files are frozen legacy evidence, not the target
architecture. No new implementation may be added in those languages.

## Why migration is incremental

Deleting the existing Python inference, retrieval and benchmark paths before a
Rust replacement exists would destroy the frozen baseline and make performance
claims unverifiable. Each lane is therefore replaced behind the same contract,
run against the same cases, and removed only after parity or an explicitly
accepted behavior change.

## Ordered lanes

1. `rwkv-state-runtime`: state contract, bounded Tokio scheduler, traces,
   conformance and performance harness.
2. Stateful HTTP provider adapters and same-hardware Qwen FP16/RWKV probes.
3. GPU/CPU checkpoint movement and durable lifecycle/event router.
4. Monitoring and coding farms.
5. Search, extraction, indexing and evaluation utilities.
6. Rust/WASM UI, packaging and final removal of Python/JS release dependencies.

## Mandatory gates per lane

- same input and immutable artifact hashes;
- correctness before throughput;
- protocol, owner, cancellation and lifecycle parity;
- P50/P95/P99 latency, throughput, CPU/RAM and overload/backpressure;
- GPU memory, TTFT, transfer bytes and PCIe bandwidth where applicable;
- Dev, public regression, one-run Sealed Blind and OOD Transfer;
- no rules derived from Gold, Case ID or failure traces.

## Inventory

The Rust binary below inventories remaining executable legacy files and records
their SHA-256 without importing or executing them:

```bash
rwkv-rust-migration-audit --root . --output rust-migration-audit.json
```

`migration_complete=false` is expected until all ordered lanes pass. It must
never be changed by hiding legacy files from the audit; only verified removal
can close it.
