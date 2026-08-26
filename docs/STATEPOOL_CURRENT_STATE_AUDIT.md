# StatePool Cloud Plugin — current-state audit

Original audit baseline: `main` at
`0630910835827ed9591062be5b37d174e0bdbe6c` (`0.3.0-beta.2`). The status below
is updated through the 2026-08-27 Cloud Plugin development line; remaining
gaps are retained explicitly rather than mixed with the original baseline.

## Existing control and data planes

The canonical control plane is the Rust `AgentService` in
`crates/agent-runtime/src/service.rs`, exposed by `crates/agent-server`. The
Python model Sidecar and retrieval service remain external data-plane
providers. The default server configuration is local and requires no cloud
service.

The current request path is:

```text
client -> agent-server -> AgentService
  -> SessionStore / TaskLedger
  -> SidecarClient -> RWKV model Sidecar
  -> DataPlaneClient -> tools and retrieval
```

Important existing invariants:

- `RuntimeConfig::default()` selects local endpoints and local filesystem
  session storage.
- `AgentService::new` constructs one `SidecarClient`, `SessionStore`,
  `TaskLedger`, command sandbox and debug store.
- `/live` is provider-independent. `/ready` verifies every configured model
  Sidecar, the data plane, the sandbox when enabled, and state capacity.
- shutdown cancels active tasks, waits for the bounded provider boundary and
  releases cached chat states.
- the canonical service API and compatibility aliases share one Agent loop.

These invariants are the backwards-compatibility baseline for the optional
Cloud Plugin.

## Existing recurrent-state lifecycle

`SidecarClient` in `crates/agent-runtime/src/sidecar.rs` already exposes:

- `prefill`;
- `fork`;
- `batch_continue` and streamed continuation;
- `release_many`;
- owner-scoped state access.

New states are assigned a `home_url`. Every later continuation, fork and
release uses that `home_url`, so state affinity already exists inside one
Sidecar lifetime. Selection of a new Sidecar is round-robin and is not aware of
state location, queue time, price, zone or SLO.

The direct-chat cache in `AgentService` retains a bounded number of
`CachedChatState` values in process memory. In default local mode these contain
the original Hot Sidecar handles. With the independently opt-in lifecycle,
safe turns are committed through a fenced Lease and the cache retains a
Warm/Cold `StateReference` instead, so the Worker slot is released between
turns. A transcript-length mismatch, eviction or explicit invalidation drops
the cached location; the durable transcript remains the compatibility fallback
only when no committed remote State is being resumed.

## Existing state contract

`contracts/stateful-inference-session-v1.schema.json` and
`crates/state-runtime` already define:

- a complete `ModelRef` containing `model_id`, `revision`, `tokenizer` and
  `state_abi`;
- owner-isolated `SessionHandle` values;
- `create`, `continue`, `snapshot`, `restore`, `describe` and `release`;
- atomic checkpoints with SHA-256, byte size and placement;
- explicit provider and model mismatch failures.

The in-memory conformance provider implements the complete lifecycle. The live
Albatross Sidecar and `RwkvHttpProvider` now implement an exact CPU transport:
the scheduler exports recurrent tensors plus logits into a bounded
`safetensors` envelope, the HTTP boundary checks SHA-256 and exact `ModelRef`,
and restore installs the tensors into a fresh slab slot. Unit and HTTP-provider
tests cover snapshot, mandatory source release, restore and continuation. The
opt-in Controller path now commits that payload through the plugin to the
configured LocalFS or S3 store and caches the returned reference. The
deterministic full-path test is complemented by a real RTX 4080
forced-process-loss run: PostgreSQL/S3 commit, source PID termination, fresh
exact-compatible Worker process, restore to a new GPU State ID, continuation
and release all passed. Raw evidence is linked from
[`evidence/statepool/real-gpu-worker-kill-2026-08-27.md`](../evidence/statepool/real-gpu-worker-kill-2026-08-27.md).
The HF recurrent backend explicitly rejects exact snapshots until its cache
constructor has an equivalent checked adapter.

The v1 stateful inference contract remains unchanged. Cloud-specific location,
version, lease and cost data are defined by separate StatePool plugin-v1
contracts so existing consumers are not broken.

## Safe plugin seams

The minimum-intrusion seams are:

1. Add an optional `CloudPluginConfig` to `RuntimeConfig`, with a disabled
   default.
2. Construct a `CloudPluginClient` in `AgentService::new`. Disabled mode uses
   an in-process no-op implementation and performs no network request.
3. Ask the plugin for an `ExecutionPlan` before creating a new Sidecar state.
   A cached state remains pinned to its `home_url` unless an explicit,
   ABI-compatible restore succeeds.
4. Emit usage after an atomic provider boundary. Usage failure must not change
   a successful model result.
5. Keep distributed persistence, Lease ownership and transfer orchestration in
   the out-of-process plugin; the Sidecar implements only the Worker-local
   tensor export/import boundary.

No existing public Agent endpoint needs to change. New request placement hints
must be optional and old clients must continue to deserialize successfully.

## Failure and consistency gaps

The remaining gaps must be closed rather than hidden:

- the local `SessionStore` lock is process-local and cannot prevent two
  Controllers from entering before Lease acquisition; lifecycle mutations are
  fenced, but multi-Controller request admission still needs a distributed
  Session boundary;
- PostgreSQL/S3 retain the current State, but Controller restart does not yet
  reconstruct its in-memory Session→`StateReference` index;
- `SidecarClient::health` currently fails readiness if any configured endpoint
  fails, whereas a dynamic Worker pool needs per-Worker readiness;
- default local selection remains round-robin; lifecycle restore placement is
  state-aware through the plugin;
- Controller usage records now include observed restore/snapshot wall time,
  exact State bytes, output/token estimates, a documented model-wall-time GPU
  proxy, selected-plan queue/cost estimates and actual Worker zone; production
  accelerator utilization and billing still require provider telemetry;
- Worker drain is unit-, cross-process- and real-GPU preStop-tested, including
  a rebuildable system root that does not mask dirty user State, but a live
  Kubernetes/KEDA 0→1→N→0 cycle has not been exercised.

The product must retain the documented transcript-reprefill fallback for
incompatible or absent durable State. The measured migration claim is limited
to an exact-compatible fresh Worker process on one physical RTX 4080; it is not
a cross-model, simultaneous cross-node, or production-latency claim.

## Existing tests that protect compatibility

The primary Rust regression boundaries are:

- `crates/agent-runtime/tests/mock_full_path.rs` — direct chat reuse, error and
  cancellation release, tool loop and debug behavior;
- `crates/agent-server/tests/service_pipeline.rs` — canonical HTTP identity,
  idempotency, streaming, cancellation and provider-unavailable behavior;
- `crates/state-runtime/tests/contract.rs` — lifecycle contract conformance;
- `crates/state-runtime/tests/rwkv_http.rs` — live-provider affinity and release;
- `crates/state-runtime/tests/trace_and_concurrency.rs` — trace and concurrency
  invariants.

Rust compilation and tests must run remotely under the repository
`AGENTS.md` policy. Schema and documentation checks may run locally because
they do not compile Rust.

## Integration decision

The Cloud Plugin will be an optional, out-of-process service using a versioned
HTTP/JSON contract. Kubernetes, KEDA, S3-compatible storage, PostgreSQL,
Prometheus, Grafana, AIBrix and HAMi remain independently deployed systems.
Only adapters, deployment profiles, dashboards and a tested compatibility
matrix enter this repository.

The complete decision and failure semantics are recorded in
`docs/adr/0002-statepool-cloud-plugin-boundary.md`.
