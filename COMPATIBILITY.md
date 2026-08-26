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

## Optional infrastructure

The following integrations are planned and are **not yet verified**. Versions
and image digests will be pinned only when their deployment tests are added.

| Component | Intended interface | Status |
|---|---|---|
| Kubernetes | Deployment/Service/lifecycle hooks | planned |
| KEDA | `ScaledObject` using StatePool Prometheus metrics | planned |
| PostgreSQL | lease, fencing and State version CAS | planned |
| S3/MinIO | immutable Cold State objects | planned |
| Prometheus/Grafana | scrape and dashboard | planned |
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
