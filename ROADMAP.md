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
  drain adapter (complete; real RTX 4080 process evidence and simulated-Worker
  Kubernetes lifecycle evidence archived separately);
- immutable model/tokenizer/State ABI from the Worker, not operator guesswork
  (complete);
- exact-compatible fresh-Worker-process export/import continuation test
  (complete on one physical RTX 4080, sequential processes);
- Controller Lease path for any plan with `lease_required=true` (complete);
- no fallback after ambiguous remote start (complete and fault-injection tested).

## Gate 3 — Cloud Lite durability

- PostgreSQL MetadataStore with transactionally monotonic fencing and CAS
  (complete; two-client PostgreSQL 17.6 integration);
- S3-compatible StateStore with checksum and immutable keys (complete against
  MinIO; multipart transfer and orphan garbage collection remain future work);
- restart/reconnect/concurrent-client fault injection (complete for the
  published adapter claims);
- Compose closes a generic cross-plugin-restart byte lifecycle, and the same
  services backed an exact-model RTX 4080 process-loss lifecycle.

## Gate 4 — Kubernetes elasticity

- image identities and development SBOM archived (published immutable registry
  digests remain pending);
- publish a real GPU Worker image containing the implemented heartbeat and
  preStop drain adapter (pending; the current real GPU run used a process);
- KEDA control plane measured 0→1→3→0 with a protocol-faithful simulator and
  three safe drains (complete); real GPU Pod lifecycle remains pending;
- raw Prometheus/Kubernetes series archived; Grafana runtime screenshot pending;
- evidence-backed 100-Session A/B/C capacity replay complete; a same-topology
  live GPU benchmark remains pending.

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
