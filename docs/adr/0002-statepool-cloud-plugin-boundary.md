# ADR 0002: Optional out-of-process StatePool Cloud Plugin

- Status: Accepted
- Date: 2026-08-26
- Scope: plugin boundary, transport, state consistency, failure semantics,
  privacy and third-party integration

## Context

`rwkv-agent` is a local-first personal assistant with a Rust control plane and
external model/data Sidecars. It already reuses owner-isolated RWKV recurrent
state, persists durable transcripts and defines a provider-neutral stateful
inference session contract. The default product must remain usable without a
cluster, database, object store or cloud account.

The Cloud Plugin needs to add state-aware placement, tiered persistence,
elastic GPU Workers, safe scale-to-zero and FinOps without turning the existing
application into a mandatory Kubernetes platform or maintaining forks of
upstream infrastructure projects.

## Decision

### Process boundary

The Cloud Plugin is an optional out-of-process service. The host communicates
over versioned HTTP/JSON on TCP or a Unix domain socket. Rust dynamic libraries
are rejected because Rust does not provide a stable cross-release dynamic ABI
and an in-process plugin would weaken failure isolation.

The plugin is disabled by default. Disabled mode uses a no-op/local decision
path, opens no plugin socket and preserves the existing local Sidecar,
filesystem and API behavior.

### Ownership boundary

The host remains authoritative for:

- public Agent APIs, identity and tool policy;
- durable transcripts, task records and user-visible results;
- cancellation, deadlines and local fallback;
- the existing local mode.

The plugin owns only:

- dynamic Worker registration and health;
- placement decisions and their explanations;
- State location, tier, checkpoint version and lease metadata;
- snapshot/restore orchestration after a provider supports export/import;
- drain coordination and FinOps events.

Kubernetes owns Pod/node placement. KEDA owns replica count. HAMi, when
installed, owns GPU slicing and isolation. S3 owns checkpoint objects.
PostgreSQL owns distributed metadata. Prometheus/Grafana own telemetry storage
and visualization. AIBrix, when enabled, owns gateway concerns. None of these
systems becomes a source copy inside `rwkv-agent`.

### Contract and compatibility

All plugin messages carry a `contract_version`. Host and plugin perform a
handshake before the plugin becomes ready. Unknown required capabilities or a
major contract mismatch keep the plugin unavailable without making local mode
unavailable.

The existing `stateful-inference-session.v1` contract is not modified. Plugin
location, version, lease and cost information uses separate `statepool-*.v1`
schemas. Additive optional host configuration is the only change to the normal
startup surface.

### State compatibility

Exact recurrent-State restore is permitted only when all of the following are
equal:

```text
provider_mode
model_id
revision
tokenizer
state_abi
```

The checkpoint checksum and byte size must also verify before import. Model
family similarity, parameter count or a shared tokenizer is not sufficient.

When exact compatibility is unavailable, the plan must select
`context_capsule` or `transcript_reprefill`. It must never label this fallback
as raw-State migration.

### Single-writer consistency

Every mutating continuation requires a lease containing `session_id`,
`owner_id`, `fencing_token`, `expected_state_version`, holder and expiry. A
successful continuation commits a new immutable State version with compare-and-
swap. A stale holder cannot commit after another holder acquires a newer
fencing token.

Checkpoint publication is atomic:

1. upload a temporary object;
2. verify byte size and SHA-256;
3. publish the immutable object;
4. compare-and-swap metadata from version N to N+1;
5. release the lease;
6. garbage-collect the temporary object asynchronously.

### Failure semantics

- Plugin disabled: use the original local path.
- Handshake, health or planning failure before any remote lease or provider
  operation: local fallback is allowed when policy permits it.
- Remote lease acquired but no provider request sent: release/expire the lease,
  then re-plan.
- Provider request sent and completion unknown: do not execute locally in
  parallel. Reconcile the committed version or wait for lease expiry before
  retrying.
- Usage/telemetry failure after a committed model result: record a bounded local
  retry item but do not change the user result.
- Snapshot verification or restore compatibility failure: fail closed for raw
  State and use an explicitly reported fallback only if policy permits it.
- Plugin crash must not crash the host process.

### Privacy

Planning receives metadata, not transcript or raw recurrent-State bytes. Raw
State transfer occurs only for policies that allow the selected zone. Three
portable policy classes are defined:

- `local_only`: neither prompts nor State leave the trusted local boundary;
- `hybrid`: the host may send an explicitly constructed Context Capsule or an
  allowed encrypted checkpoint;
- `cloud_allowed`: compatible cloud Workers may receive the required prompt and
  State.

Credentials are passed by environment/Secret references and never serialized
into plugin contracts, traces, compatibility files or repository manifests.

### Third-party repositories

Upstream source is not vendored by default. Integrations enter this repository
as one of:

1. a stable protocol adapter;
2. a package-manager dependency pinned by lockfile;
3. a separately deployed OCI image/Helm dependency pinned by version or digest;
4. Kubernetes manifests, dashboards and compatibility tests.

Git submodules, source copies and maintained forks require a separate ADR and
license review. Optional services with reciprocal licenses remain separate
processes and are documented in `third_party/COMPONENTS.yaml`.

## Consequences

The local product remains small and stable, plugin failures are isolated, and
cloud components can evolve behind standard contracts. The trade-off is an
extra network boundary and the need for explicit reconciliation after
ambiguous failures. The Albatross Sidecar passes CPU snapshot/restore
conformance, distributed PostgreSQL/S3 fencing, and one real RTX 4080
forced-Worker-process-loss restore. That evidence permits an exact-compatible
fresh-process recovery claim only; cross-model, live multi-node rollout and
production-latency claims remain out of scope until separately measured.

## Verification

The decision is satisfied only when:

- plugin-disabled regression tests prove unchanged behavior and no plugin
  network call;
- contract fixtures pass schema validation;
- an ambiguous remote completion cannot cause two committed State versions;
- a compatible Worker kill/restore test proves continuation;
- an incompatible model test selects an explicitly labeled fallback;
- external integrations can be omitted from the default installation.
