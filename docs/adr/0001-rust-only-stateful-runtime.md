# ADR-0001: Rust-only Stateful Inference Runtime

- Status: Accepted for Phase 0
- Date: 2026-08-22
- Scope: new runtime, harness, benchmark and migration target

## Decision

The repository end state is a Rust Workspace. New executable runtime, harness,
benchmark, service, provider adapter, migration utility and test code must be
Rust. Existing Python and JavaScript are frozen legacy paths and may be removed
only after a Rust replacement passes the same correctness and artifact gates.
The browser UI will move to Rust/WASM in a later isolated migration.

The core project is a **Stateful Inference Scheduler**, not another multi-Agent
framework. Agent code owns durable task state. Providers own disposable
inference state. Agents receive opaque handles and never tensors.

Model weights may be served by a separately versioned inference engine behind a
narrow protocol. The public repository must not require a Python adapter to run
its scheduler, control plane or benchmarks. A provider result always declares
one of:

- `rwkv_recurrent`;
- `qwen_native_kv`;
- `qwen_transcript_reprefill`;
- `contract_test` (never a model-quality result).

Transcript replay must never be described as native KV snapshot/restore.

## Concurrency architecture

The Rust workload runner uses one Tokio multi-thread runtime, a bounded queue per
worker and deterministic owner-affine routing. All events for one Agent reach
the same worker in order; different Agents advance concurrently. Full queues
apply backpressure and are counted. There is no OS thread per Agent, unbounded
queue, busy poll or global Agent lock.

The benchmark reports Dormant, Resident, Ready and Active separately. `10,000`
durable metadata records do not mean `10,000` simultaneous GPU decode rows.

## StatefulInferenceSession v1

```text
create(owner_id, durable_session_ref, model_ref) -> SessionHandle
continue(owner_id, session_handle, input, token_budget) -> ContinueResult
snapshot(owner_id, session_handle, target_tier) -> CheckpointRef
restore(owner_id, checkpoint_ref, expected_model_ref) -> SessionHandle
describe(owner_id, session_handle) -> SessionDescription
release(owner_id, session_handle) -> ReleaseOutcome
```

`ModelRef` includes model ID, immutable revision, tokenizer and State ABI.
Snapshot metadata includes owner, provider mode, placement, bytes, timestamps,
an atomic flag and SHA-256. Restore fails closed on owner, provider, model, ABI,
size or checksum mismatch. Release is owner-isolated and idempotent.

## Existing RWKV HTTP adaptation table

| Contract | Existing endpoint | Phase 0 status |
|---|---|---|
| `create` | `POST /v1/states/prefill` | adaptable after Durable State resolver |
| `continue` | `POST /v1/states/batch_continue` | available |
| `describe` | `/health` plus local handle metadata | partial |
| `release` | `POST /v1/states/release` | available; exact idempotency must be verified |
| `snapshot` | none | unsupported |
| `restore` | none | unsupported |
| fork extension | `POST /v1/states/{id}/fork` | existing extension, not part of v1 minimum |

The existing GPU State Pool and scheduler are reused. Phase 0 does not create a
parallel GPU scheduler or claim CPU tensor migration.

## Performance and correctness gates

Every performance result must pass sentinel correctness, protocol validity,
owner isolation, cancellation cleanup and final State closure. Core levels are
`N=1/4/16/32/64/100/1,000`; metadata-only stress adds `10,000`. Report
throughput, useful throughput, P50/P95/P99, backpressure, CPU/RAM and, for live
providers, GPU memory, TTFT, transfer bytes and PCIe bandwidth.

The in-memory conformance provider only validates the Rust contract and control
plane. Its events/s numbers cannot be presented as RWKV or Qwen inference
throughput.

## Anti-overfitting

Development, public regression, sealed blind and OOD transfer are separate.
Case IDs, repository names, gold data and failure traces are unavailable to the
scheduler. Only generic runtime signals may influence placement. Runtime and
contract revisions are frozen before the one formal sealed run. Results include
macro average, worst category and feature-off/feature-on ablations.

## Repository migration sequence

1. Rust contract, trace, schema and control-plane stress benchmark.
2. Rust HTTP providers and live same-hardware Qwen FP16/RWKV measurement.
3. Rust GPU State lifecycle and CPU checkpoint path.
4. Rust durable lifecycle, router, monitoring and coding farms.
5. Rust replacements for legacy retrieval/index/benchmark utilities.
6. Rust/WASM UI and deletion of the final Python/JavaScript release dependency.

Deletion happens only after parity evidence is frozen; no big-bang rewrite is
allowed to erase the baseline.
