# StatePool compatibility policy

`rwkv-agent` does not promise compatibility with every upstream main branch.
Each release supports a bounded, evidence-backed compatibility window.

## Current verified core

| Component | Version/revision | Status | Evidence |
|---|---|---|---|
| rwkv-agent | `0.3.0-beta.2` plus current StatePool changes | verified remotely | `cargo test --workspace --all-targets` on `WZU_Server` |
| StatePool plugin API | `statepool-plugin.v1` | verified | schema examples and Rust contract tests |
| Stateful inference session | `stateful-inference-session.v1` | verified for conformance provider | `crates/state-runtime/tests/contract.rs` |
| Live RWKV snapshot/restore | Albatross `rwkv7_fast_v3a`, `fp32io16`, exact State ABI | RTX 4080 process-loss recovery verified | `evidence/statepool/real-gpu-worker-kill-2026-08-27.md` |
| StatePool container images | plugin image ID `sha256:396fbb6b…`; mock KEDA Worker `sha256:7c6d2d51…` | remote build/smoke and kind runtime verified | `evidence/statepool/remote-container-smoke-2026-08-26.md`; `bench/artifacts/statepool-keda-kind-20260827/` |

## Optional infrastructure

Each row distinguishes runtime evidence from a merely authored integration.
An exact pin alone is not a verified deployment claim.

| Component | Intended interface | Status |
|---|---|---|
| Kubernetes `1.34.0` on kind `0.30.0` | Deployment/Service/lifecycle hooks | live control-plane cycle verified with a simulated Worker; no GPU Pod claim |
| RWKV Sidecar Worker adapter | `statepool-worker-capability.v1`, TTL heartbeat, readiness and drain | real RTX 4080 process lifecycle verified; Kubernetes used a protocol-faithful simulator, not the GPU image |
| KEDA `2.20.1` | `ScaledObject` using StatePool Prometheus metrics | kind 0→1→3→0 verified; all three simulated Worker preStop results safe |
| PostgreSQL `17.6-bookworm` | lease, fencing and State version CAS | adapter verified with two clients against exact 17.6 container on WZU_Server |
| MinIO `RELEASE.2025-04-22T22-12-26Z` | immutable Cold State objects | adapter and real-container integration verified on WZU_Server |
| Prometheus `3.11.3` | scrape and KEDA external metrics | live kind scrape and scaling input verified |
| Grafana `13.1.0` | provisioned FinOps dashboard | Compose configuration and dashboard schema authored; screenshot/runtime review pending |
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
