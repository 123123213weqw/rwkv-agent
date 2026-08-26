# Roadmap

Roadmap items are evidence gates, not date promises.

## Gate 0 — Local product compatibility (complete)

- local RWKV Agent, UI/CLI, tools and State reuse remain the default;
- cloud plugin disabled by default and creates no HTTP client;
- remote Rust workspace regression passes.

## Gate 1 — StatePool development profile (complete)

- plugin-v1 handshake, health and capability negotiation;
- Worker TTL registry, privacy/exact-model filtering and explainable placement;
- single-process Lease/fencing/version CAS;
- atomic checksum-verified LocalFS State round trip;
- conservative Drain admission and State-aware Prometheus metrics;
- Compose/Helm/KEDA/ServiceMonitor/Grafana assets with safe defaults.

## Gate 2 — Live RWKV State adapter

- Sidecar Snapshot/Restore/Batch Continue/Release contract (complete);
- opt-in Worker registration, heartbeat, readiness and conservative preStop
  drain adapter (complete; Kubernetes runtime evidence pending);
- immutable model/tokenizer/State ABI from the Worker, not operator guesswork
  (complete);
- local two-Worker export/import deterministic continuation test;
- Controller Lease path for any plan with `lease_required=true`;
- no fallback after ambiguous remote start.

## Gate 3 — Cloud Lite durability

- PostgreSQL MetadataStore with transactionally monotonic fencing and CAS;
- S3-compatible StateStore with checksum, immutable key, multipart limits and
  orphan garbage collection;
- restart/reconnect/concurrent-replica fault injection;
- Compose closes create→continue→snapshot→kill→restore→continue→release.

## Gate 4 — Kubernetes elasticity

- published image digests and SBOM;
- publish a real GPU Worker image containing the implemented heartbeat and
  preStop drain adapter;
- KEDA measured 0→1→N→0 without double execution or dirty State loss;
- Dashboard screenshot and raw Prometheus series archived;
- A/B/C Sticky/Re-prefill/StatePool benchmark.

## Gate 5 — Optional ecosystem adapters

- AIBrix/KServe routing integration where it reduces duplicate gateway work;
- HAMi annotations/values for supported devices, without making HAMi required;
- edge Worker registration through authenticated mTLS/VPN boundary;
- multi-cloud price catalog adapter with timestamped currency/source.

## Gate 6 — Stable plugin release

- authentication/authorization and tenant isolation threat model;
- high-availability metadata and disaster recovery runbook;
- compatibility and migration test matrix;
- public release checklist, signed artifacts and reproducible benchmark bundle.
