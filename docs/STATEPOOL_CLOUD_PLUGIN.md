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
      -> privacy/model/state-aware placement
      -> Lease + fencing + State version CAS
      -> atomic LocalFS State store (development profile)
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

The host currently invokes placement only when direct chat opens a new
recurrent State. Cached chat State remains pinned to `home_url`; research forks
and tool-loop roots still use the original local Sidecar selection. Those paths
will be enabled only after their owner/release tests are extended to cover a
remote Worker.

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

The plugin stores at most 10,000 recent usage records in memory in this slice.
Prometheus persistence and PostgreSQL metadata are later deployment gates.

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

This profile's metadata is in memory and its immutable objects are on LocalFS.
A plugin restart loses Lease/current-version metadata, so it is not a
multi-replica or production durability claim.

## Deliberate non-claims

The current plugin advertises `placement`, `worker_registry`, `leases`,
`state_lifecycle`, `drain` and `finops`. `leases` and `state_lifecycle` are
limited to the single-process development profile above. It does not advertise
`remote_state`, because the live RWKV HTTP provider still lacks verified
snapshot/restore, S3 persistence and distributed fencing. A remote plan may
select a compatible Worker for a newly opened State; it does not prove
migration of an already-open State.

Exact cross-Worker continuation may be claimed only after:

1. Sidecar export/import is implemented;
2. checksum and exact model identity are validated;
3. PostgreSQL lease/CAS and fencing tests pass;
4. a real Worker-kill restore benchmark passes.

See the [current-state audit](STATEPOOL_CURRENT_STATE_AUDIT.md) and
[ADR 0002](adr/0002-statepool-cloud-plugin-boundary.md).

Deployment profiles, KEDA gating and the provisioned Grafana dashboard are in
[`deploy/statepool/README.md`](../deploy/statepool/README.md).
