# StatePool deployment profiles

These assets integrate upstream infrastructure without copying or forking its
source. They do not change the default local `rwkv-agent` startup path.

## Profile 1: Local control-plane demo

Requirements: Docker Engine with Compose v2. The repository `AGENTS.md` still
applies: do not build Rust on the workstation. Build the image on an approved
remote build machine, or use a published image when one is available.

The Dockerfile's two targets were built and smoke-tested on `WZU_Server`; see
[`evidence/statepool/remote-container-smoke-2026-08-26.md`](../../evidence/statepool/remote-container-smoke-2026-08-26.md).

Validate the Compose model without starting or building anything:

```bash
docker compose -f deploy/statepool/compose.yaml config --quiet
```

The default profile contains only:

- the single-replica StatePool plugin with an immutable LocalFS State volume;
- Prometheus `v3.11.3`;
- Grafana `13.1.0` with a provisioned StatePool/FinOps dashboard.

After an image has been built on the approved remote host:

```bash
docker compose -f deploy/statepool/compose.yaml up -d statepool prometheus grafana
curl --fail http://127.0.0.1:8130/live
curl --fail http://127.0.0.1:8130/metrics
```

Open Prometheus at <http://127.0.0.1:9090> and Grafana at
<http://127.0.0.1:3000>. The dashboard shows local/cloud placement, pending
demand, estimated decode backlog, State tier events, State I/O, Lease
conflicts, snapshots/restores, GPU seconds, avoided Prefill tokens and
currency-scoped estimated cost.

The optional `agent` profile connects the containerized Controller to an
already-running model Sidecar and data plane on the host:

```bash
docker compose -f deploy/statepool/compose.yaml --profile agent up -d
```

Replace the example immutable model revision before enabling it.

The opt-in `cloud-lite` profile starts a second plugin on port `8131` backed by
PostgreSQL Lease/CAS metadata and a MinIO S3 bucket:

```bash
docker compose -f deploy/statepool/compose.yaml \
  --profile cloud-lite up -d statepool-cloud-lite
curl --fail http://127.0.0.1:8131/plugin/v1/health
```

The checked-in credentials are local-demo values only. PostgreSQL and S3
adapters were exercised against real containers on `WZU_Server`; this does not
yet constitute the live GPU Worker-kill result.

## Profile 2: Kubernetes development control plane

The Helm chart defaults to one LocalFS/in-memory plugin replica and no Worker:

```bash
helm template demo deploy/statepool/helm/statepool
```

For observability with Prometheus Operator/Grafana sidecar discovery:

```bash
helm upgrade --install statepool deploy/statepool/helm/statepool \
  --namespace rwkv-statepool --create-namespace \
  --set serviceMonitor.enabled=true \
  --set grafanaDashboard.enabled=true
```

For Cloud Lite, create one Kubernetes Secret containing `postgres-url`,
`access-key-id` and `secret-access-key`, then set
`plugin.durable.enabled=true`, the Secret names, and the external S3 endpoint,
bucket and region. The chart never renders credentials into values or a
ConfigMap.

`worker.enabled=false` and `autoscaling.enabled=false` are safety gates. Enable
them only with a Worker image that implements:

1. `statepool-worker-capability.v1` registration and TTL heartbeat;
2. exact model/tokenizer/State ABI reporting;
3. `/ready` and `/live`;
4. `/opt/rwkv/bin/statepool-drain`, which stops admission, waits for in-flight
   requests, snapshots every dirty State under a fenced Lease, reports
   `unpersisted_state_slots=0`, and exits successfully only then.

The plugin treats a missing `unpersisted_state_slots` heartbeat as unknown and
will not report `safe_to_stop`. The Pod has a 180-second termination grace
period and a preStop hook; scale-down is stabilized and limited to one Pod per
minute.

## Live two-Worker lifecycle client

Once two exact-model Albatross Sidecars and the plugin are running, execute the
real Worker and plugin protocols with:

```bash
python scripts/statepool_live_lifecycle_demo.py \
  --plugin-url http://127.0.0.1:8130 \
  --source-worker-url http://127.0.0.1:8118 \
  --target-worker-url http://127.0.0.1:8218 \
  --source-worker-id worker-v100-a \
  --target-worker-id worker-v100-b \
  --output bench/artifacts/statepool-live-lifecycle.json
```

The sequence is prefill → continue → acquire Lease → Sidecar snapshot →
StatePool immutable commit/CAS → source release → new fenced Lease → StatePool
restore → target Sidecar restore → continue → release. To make it a forced-loss
experiment, pass `--source-stop-command 'docker kill rwkv-worker-a'`. It runs
only after the StatePool commit succeeds. A run without that option is a
lifecycle test, not Worker-kill evidence.

## KEDA integration

The chart emits a KEDA `ScaledObject` only when both Worker and autoscaling are
enabled. It targets the upstream Prometheus scaler using:

- `statepool_pending_requests` for 0→1 demand;
- `statepool_estimated_decode_seconds` for backlog-aware 1→N scaling;
- `minReplicaCount: 0` and a five-minute cooldown for N→0.

KEDA itself is installed separately; the manifests were authored against
[KEDA v2.20.1](https://github.com/kedacore/keda/releases/tag/v2.20.1).
No KEDA source or Chart is vendored here.

The current pending metric is a bounded demand signal: a cloud-allowed plan
miss increments it, and a compatible Worker registration clears it. A request
that safely falls back locally is not replayed remotely. Thus scale-from-zero
benefits subsequent requests and does not create ambiguous double execution.

## Claim boundary

These assets prove configuration and contract wiring, not a live GPU
scale-to-zero benchmark. A competition result can be marked measured only after
the exact images, cluster, KEDA version, Sidecar adapter, commands and raw logs
are archived under `bench/`.
