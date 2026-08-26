# StatePool compatibility policy

`rwkv-agent` does not promise compatibility with every upstream main branch.
Each release supports a bounded, evidence-backed compatibility window.

## Current verified core

| Component | Version/revision | Status | Evidence |
|---|---|---|---|
| rwkv-agent | `0.3.0-beta.2` plus current StatePool changes | verified remotely | `cargo test --workspace --all-targets` on `WZU_Server` |
| StatePool plugin API | `statepool-plugin.v1` | verified | schema examples and Rust contract tests |
| Stateful inference session | `stateful-inference-session.v1` | verified for conformance provider | `crates/state-runtime/tests/contract.rs` |
| Live RWKV snapshot/restore | not implemented | unsupported | `RwkvHttpProvider` returns `unsupported` |
| StatePool container images | local remote build after `c0aa5ed` | build/smoke verified on 2026-08-26 | `evidence/statepool/remote-container-smoke-2026-08-26.md` |

## Optional infrastructure

The following integrations remain **not runtime verified**. Exact versions are
pinned where manifests now exist; a pinned version alone is not a verified
deployment claim.

| Component | Intended interface | Status |
|---|---|---|
| Kubernetes | Deployment/Service/lifecycle hooks | Helm template authored; cluster test pending |
| RWKV Sidecar Worker adapter | `statepool-worker-capability.v1`, TTL heartbeat, readiness and drain | implementation/unit tests verified; real GPU Pod test pending |
| KEDA `2.20.1` | `ScaledObject` using StatePool Prometheus metrics | template authored; 0→1→N→0 test pending |
| PostgreSQL `17.6-bookworm` | lease, fencing and State version CAS | adapter verified with two clients against exact 17.6 container on WZU_Server |
| MinIO `RELEASE.2025-04-22T22-12-26Z` | immutable Cold State objects | adapter and real-container integration verified on WZU_Server |
| Prometheus `3.11.3` / Grafana `13.1.0` | scrape and dashboard | configuration authored; runtime test pending |
| AIBrix | optional gateway/route adapter | planned |
| HAMi | optional resource annotations/profile | planned |

An optional integration becomes `verified` only when the table contains:

1. an exact version or immutable image digest;
2. the deployment profile used;
3. a reproducible command;
4. archived raw output;
5. the date and hardware environment.

Upgrading an upstream dependency creates a new compatibility row; it does not
silently change an existing release claim.
