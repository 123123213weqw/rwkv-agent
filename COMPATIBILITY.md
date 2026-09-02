# StatePool compatibility policy

`rwkv-agent` does not promise compatibility with every upstream main branch.
Each release supports a bounded, evidence-backed compatibility window.

## Current verified core

| Component | Version/revision | Status | Evidence |
|---|---|---|---|
| rwkv-agent | `0.3.0-beta.2` plus current StatePool changes | verified remotely | `cargo test --workspace --all-targets` on `WZU_Server` |
| StatePool plugin API | `statepool-plugin.v1` | verified | schema examples and Rust contract tests |
| Worker State modes | `replay_only`, `affinity_only`, `native_export` | contract and placement tests implemented | legacy Worker omission remains `native_export`; non-native modes never receive restore Leases |
| Stateful inference session | `stateful-inference-session.v1` | verified for conformance provider | `crates/state-runtime/tests/contract.rs` |
| Live RWKV snapshot/restore | Albatross `rwkv7_fast_v3a`, `fp32io16`, exact State ABI | native client/runtime implemented; product evidence external | [`statepool-cloud/evidence`](https://github.com/123123213weqw/statepool-cloud/tree/main/evidence) |
| StatePool Cloud service | independent repository | client contract only in this repository | [`statepool-cloud`](https://github.com/123123213weqw/statepool-cloud) |

## Optional infrastructure

Cloud deployment, OpenAI/vLLM, KEDA, PostgreSQL/S3, Prometheus/Grafana and
FinOps compatibility are versioned and evidenced in the independent
[`statepool-cloud`](https://github.com/123123213weqw/statepool-cloud) repository.

An optional integration becomes `verified` only when the table contains:

1. an exact version or immutable image digest;
2. the deployment profile used;
3. a reproducible command;
4. archived raw output;
5. the date and hardware environment.

Upgrading an upstream dependency creates a new compatibility row; it does not
silently change an existing release claim.
