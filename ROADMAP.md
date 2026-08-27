# Roadmap

Roadmap items are evidence gates, not date promises.

## Gate 0 — Local product compatibility (complete)

- local RWKV Agent, UI/CLI, tools and State reuse remain the default;
- cloud plugin disabled by default and creates no HTTP client;
- remote Rust workspace regression passes.

## Gate 1 — StatePool client and native RWKV adapter

- Sidecar Snapshot/Restore/Batch Continue/Release contract (complete);
- opt-in Worker registration, heartbeat, readiness and conservative preStop
  drain adapter (complete);
- immutable model/tokenizer/State ABI from the Worker, not operator guesswork
  (complete);
- exact-compatible fresh-Worker-process export/import continuation test
  (validated externally by StatePool Cloud evidence);
- Controller Lease path for any plan with `lease_required=true` (complete);
- no fallback after ambiguous remote start (complete and fault-injection tested).

## External cloud roadmap

Cloud durability, Kubernetes/KEDA elasticity, OpenAI/vLLM adapters, FinOps and
multi-cloud work now live in the independent
[`statepool-cloud`](https://github.com/123123213weqw/statepool-cloud) roadmap.
This repository only advances the client and native RWKV State interface.
