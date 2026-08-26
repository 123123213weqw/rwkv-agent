# StatePool Cloud Plugin

Status: opt-in development slice. The default local personal assistant remains
unchanged.

The plugin makes cloud placement an optional extension instead of a hard
dependency:

```text
rwkv-agent-server-rs
  -> versioned plugin-v1 HTTP boundary
    -> rwkv-statepool-cloud-plugin :8130
      -> Worker directory
      <- Sidecar Worker registration / TTL heartbeat
      -> privacy/model/state-aware placement
      -> Lease + fencing + State version CAS
      -> atomic LocalFS or S3 State store
      -> in-memory or PostgreSQL metadata
      -> drain admission
      -> FinOps metrics
```

The host sends placement metadata only. It does not send transcript text,
prompts, credentials or raw recurrent-State bytes to `/plugin/v1/plan`.

## Compatibility guarantee

Without `--cloud-plugin` or `RWKV_AGENT_CLOUD_PLUGIN=true`:

- no plugin HTTP client is constructed;
- no request is sent to port 8130;
- existing Sidecar round-robin selection and `home_url` state affinity remain
  in force;
- local transcript, task and UI behavior remain the source-compatible default.

When the plugin is enabled with `fallback=local`, a failed handshake or plan
before any remote operation degrades to the original local path. With
`fallback=fail_closed`, plugin readiness becomes mandatory.

## Start the current development service

Rust commands must be run on the remote build host under the repository
`AGENTS.md` policy.

```text
rwkv-statepool-cloud-plugin \
  --host 127.0.0.1 \
  --port 8130 \
  --worker-ttl-seconds 30 \
  --lease-max-ttl-seconds 120 \
  --max-state-bytes 536870912 \
  --state-dir var/statepool/states
```

Register at least one compatible Worker:

```http
POST /plugin/v1/workers/register
Content-Type: application/json
```

The request schema and a complete example are:

- `contracts/worker-capability-v1.schema.json`;
- `contracts/examples/statepool-plugin-v1/worker.json`.

The Python Sidecar performs this automatically only when
`RWKV_STATEPOOL_URL` is non-empty. It reports its exact model identity, device,
State capacity, queue/running load and resident-State count. With the variable
unset, no registration thread or drain admission middleware is activated.

During drain, new inference/prefill/restore requests receive HTTP 503 while
snapshot and release remain available. Every resident State is conservatively
reported as `unpersisted_state_slots`; therefore the preStop client cannot
claim `safe_to_stop` until the Controller has committed each State under a
fenced Lease and released it from the Worker. A deadline with remaining work
returns `deadline_exceeded` instead of silently allowing dirty termination.

Then start the Agent server with a complete model identity:

```text
rwkv-agent-server-rs \
  --cloud-plugin \
  --cloud-plugin-url http://127.0.0.1:8130 \
  --cloud-plugin-fallback local \
  --cloud-plugin-privacy cloud_allowed \
  --cloud-model-id rwkv7-g1i-7b \
  --cloud-model-revision <immutable-model-revision> \
  --cloud-tokenizer rwkv-world-v20230424 \
  --cloud-state-abi rwkv7-g1i-fp16-v1
```

The separately opt-in lifecycle transport adds:

```text
  --cloud-state-lifecycle \
  --cloud-state-target-tier cold \
  --cloud-lease-ttl-seconds 120 \
  --cloud-lifecycle-timeout-seconds 180
```

Enabling it upgrades handshake requirements to include `leases` and
`state_lifecycle`. The longer lifecycle timeout applies only to State payload
transfers; placement keeps its short fail-fast timeout. A committed
`StateReference` disables local fallback so a plugin outage cannot trigger a
second execution from transcript.

Equivalent `RWKV_AGENT_*` variables are documented in
`docs/CONFIGURATION.md`.

## Current placement policy

The current deterministic policy:

1. removes stale, non-ready and capacity-exhausted Workers;
2. requires exact `model_id`, `revision`, `tokenizer` and `state_abi` equality;
3. enforces `local_only`, `hybrid` or `cloud_allowed` zone policy;
4. prefers the Worker already holding a compatible Hot State;
5. otherwise scores preferred zone, queue depth, active requests and configured
   GPU-hour price;
6. returns an explainable `reason_code` and estimated queue/restore/cost values.

The host invokes placement when direct chat opens a new recurrent State and
again whenever a durable Warm/Cold `StateReference` must be restored. With
lifecycle disabled, cached chat State retains the original `home_url` pinning.
Research forks and tool-loop roots still use the original local Sidecar
selection; those paths will be enabled only after their owner/release tests are
extended to cover a remote Worker.

`hybrid` currently permits local and edge Workers. Explicit cloud transfer for
hybrid policy remains disabled until Context Capsule/encrypted checkpoint
policy is implemented.

## Current metrics

`GET /metrics` exposes:

```text
statepool_ready_workers
statepool_plan_requests_total
statepool_local_plans_total
statepool_remote_plans_total
statepool_rejected_plans_total
statepool_worker_registrations_total
statepool_usage_records_total
statepool_usage_requests_total{zone="local|edge|cloud"}
statepool_gpu_seconds_total
statepool_prefill_tokens_avoided_total
statepool_state_bytes_read_total
statepool_state_bytes_written_total
statepool_leases_acquired_total
statepool_lease_conflicts_total
statepool_snapshots_committed_total
statepool_restores_completed_total
statepool_pending_requests
statepool_estimated_decode_seconds
statepool_hot_state_hits_total
statepool_warm_state_hits_total
statepool_cold_state_hits_total
statepool_transcript_reprefills_total
statepool_estimated_cost_total{currency="CNY"}
```

For every successful direct-chat inference, a Controller connected to a plugin
that advertises `finops` submits a validated `UsageRecord`. Reporting is
best-effort and happens after the answer and State safety decisions: a metrics
timeout or HTTP failure is exposed as `trace.finops.status=report_failed` but
never converts successful inference into a failed response. Plugins without
the capability remain compatible and receive no usage request.

Current measurement semantics are intentionally explicit:

- `elapsed_ms`, `restore_ms` and `snapshot_ms` are Controller-observed wall
  times. Restore includes durable read plus Sidecar import; snapshot includes
  Sidecar export plus CAS commit.
- `input_tokens` and `prefill_tokens_avoided` are character/4 estimates until
  the provider exports tokenizer counters. Avoided prefill is reported only
  when an existing recurrent Session State is reused.
- `gpu_seconds` is Sidecar-reported model wall time treated as a one-active-GPU
  proxy. It is **not** SM utilization, energy use or an nvidia-smi sample.
- `queue_ms` and `estimated_cost` are the selected ExecutionPlan estimates,
  not observed billing values.
- `state_bytes_read_total` and `state_bytes_written_total` are counted by the
  authoritative restore/snapshot endpoints. Usage records repeat the byte
  values for per-turn attribution but do not increment those global counters
  again.
- `statepool_usage_requests_total` counts actual completed inference by Worker
  zone; `local_plans_total` and `remote_plans_total` remain placement decisions
  and therefore are not used as request-volume claims.

The plugin stores at most 10,000 recent usage records in memory. Prometheus is
the intended metrics persistence boundary. Session/Lease metadata can now use
PostgreSQL while the local default remains in memory.

## Lease and LocalFS State lifecycle

The development profile now implements the checked-in
`state-lifecycle-v1.schema.json` contract:

1. acquire exactly one writer Lease for `(session_id, owner_id,
   expected_state_version)`;
2. issue a monotonically increasing fencing token, including after expiry;
3. upload a base64 snapshot to Warm/Cold storage;
4. recompute SHA-256, publish the file with a temporary-file + `fsync` + atomic
   rename sequence, then CAS the immutable State metadata;
5. release the first Lease, acquire a Lease for the committed version and
   restore only the current exact-model State;
6. reject expired holders and stale fencing tokens.

`POST /plugin/v1/leases/acquire`, `/renew`, `/release`,
`/plugin/v1/states/snapshot` and `/restore` are implemented and covered by a
round-trip test. Payload upload is intentionally a simple base64 development
transport; production Workers will use an S3 presigned transfer adapter rather
than proxying large State blobs through the controller.

The default profile's metadata is in memory and its immutable objects are on LocalFS.
A plugin restart loses Lease/current-version metadata, so it is not a
multi-replica or production durability claim.

The opt-in Cloud Lite backend implements the same `MetadataStore` and
`StateStore` protocols using PostgreSQL JSONB rows locked by transactions and
S3 conditional-create objects. Two independent PostgreSQL clients are covered
by a one-writer/fencing/CAS integration test; the S3 adapter is covered against
a real MinIO bucket for immutable put, idempotent retry, conflicting write,
read and delete. These adapters preserve the plugin process boundary and do
not embed either upstream project.

## Live RWKV Worker adapter

The Albatross Sidecar exposes two exact-model Worker-local operations:

- `POST /v1/states/{state_id}/snapshot` exports a decode-ready recurrent State
  and logits as a bounded, digest-protected `safetensors` envelope;
- `POST /v1/states/restore` verifies SHA-256, owner, `model_id`, `revision`,
  tokenizer and `state_abi`, then allocates a fresh State identity.

The source State must be released before restore, preventing an accidental
second writer. `RwkvHttpProvider` implements the stateful-inference contract
over these endpoints. The Rust Controller has typed, integrity-checking
Sidecar snapshot/restore and fenced plugin acquire/renew/commit/read/release
clients. Both transports validate model identity, owner, State version,
checksum and atomicity at their respective trust boundaries.

When `--cloud-state-lifecycle` is enabled, each safe direct-chat turn now runs
`acquire → continue → renew → snapshot → CAS commit → Worker release → Lease
release`, caches only the durable reference, and restores it on the next turn.
The checked-in mock full-path test proves versions 0→1→2, Hot release after
each commit and Cold restore without another prefill. An injected uncertain
commit moves the Session to `blocked_hot` and proves that a following request
does not execute again automatically.

The remaining recovery gap is Controller restart: durable bytes and metadata
survive in PostgreSQL/S3, but the Controller-side current `StateReference`
index is not yet reconstructed at startup. Exact Albatross GPU bytes have
passed one S3 forced-Worker-process-loss experiment on an RTX 4080; see
[`evidence/statepool/real-gpu-worker-kill-2026-08-27.md`](../evidence/statepool/real-gpu-worker-kill-2026-08-27.md).
The HF recurrent backend still fails closed for exact snapshot/restore.

## Deliberate non-claims

The current plugin advertises `placement`, `worker_registry`, `leases`,
`state_lifecycle`, `drain` and `finops`. `leases` and `state_lifecycle` are
available in local or PostgreSQL/S3 profiles. It does not advertise the
stronger reserved `remote_state` capability because Controller restart
reconstruction and a live multi-node rollout have not passed. Mock/CPU
lifecycles remain conformance evidence; the separate RTX 4080 run is measured
process-loss recovery, not a production latency or cross-node claim.

The exact-compatible process-loss claim is backed by:

1. Sidecar export/import remains conformance-tested on the target GPU backend;
2. checksum and exact model identity remain validated end to end;
3. PostgreSQL lease/CAS and fencing tests continue to pass;
4. a real Worker-kill restore benchmark with raw PID, GPU, PostgreSQL, MinIO
   and preStop-drain evidence.

See the [current-state audit](STATEPOOL_CURRENT_STATE_AUDIT.md) and
[ADR 0002](adr/0002-statepool-cloud-plugin-boundary.md).

Deployment profiles, KEDA gating and the provisioned Grafana dashboard are in
[`deploy/statepool/README.md`](../deploy/statepool/README.md).
