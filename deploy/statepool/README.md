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

Replace the example immutable model revision before enabling it. This profile
enables the automatic Cold lifecycle, so each safe direct-chat turn commits
through StatePool and releases its Worker-local State.

The independent `openai-worker` profile builds only this repository's thin
adapter and connects to an already-running vLLM/OpenAI-compatible server. It
does not clone, fork or package vLLM:

```bash
export STATEPOOL_OPENAI_UPSTREAM_URL=http://host.docker.internal:8000
export STATEPOOL_OPENAI_MODEL_REVISION=sha256:REPLACE_ME
export STATEPOOL_OPENAI_TOKENIZER=REPLACE_WITH_IMMUTABLE_TOKENIZER_REVISION
docker compose -f deploy/statepool/compose.yaml \
  --profile openai-worker up -d statepool openai-worker
```

The example advertises `qwen3.5-9b-vllm` as a logical model and rewrites it to
`Qwen/Qwen3.5-9B` upstream. Those names are a candidate profile, not live-model
evidence. Replace the immutable revision/tokenizer placeholders and run the
model/GPU evidence gate before calling it certified. This adapter reports
`affinity_only`: transcript replay is mandatory and Transformer KV is never
sent to StatePool.

The opt-in `cloud-lite` profile starts a second plugin on port `8131` backed by
PostgreSQL Lease/CAS metadata and a MinIO S3 bucket:

```bash
docker compose -f deploy/statepool/compose.yaml \
  --profile cloud-lite up -d statepool-cloud-lite
curl --fail http://127.0.0.1:8131/plugin/v1/health
```

The checked-in credentials are local-demo values only. PostgreSQL and S3
adapters were exercised against real containers on `WZU_Server`; an exact-model
RTX 4080 Worker-process-loss recovery has also passed through those adapters.

## Profile 2: Kubernetes development control plane

The Helm chart defaults to one LocalFS/in-memory plugin replica and no Worker:

```bash
helm template demo deploy/statepool/helm/statepool
```

Render the optional external-vLLM adapter overlay with:

```bash
helm template demo deploy/statepool/helm/statepool \
  -f deploy/statepool/helm/statepool/values-openai-worker.yaml
```

The overlay is a CPU proxy/registration Pod; the upstream vLLM deployment
owns its GPU lifecycle. Do not enable the chart's KEDA Worker scaler for this
topology because scaling the proxy does not scale the external engine.

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

The Controller is another explicit safety gate. Enable it only after replacing
the model/data-plane endpoints and immutable model identity:

```bash
helm upgrade --install statepool deploy/statepool/helm/statepool \
  --namespace rwkv-statepool --create-namespace \
  --set controller.enabled=true \
  --set controller.modelUrls=http://routing-sidecar.example:8118 \
  --set controller.dataPlaneUrl=http://data-plane.example:8121 \
  --set controller.model.revision=sha256:REPLACE_ME
```

It exposes `/live` and `/ready`, persists the transcript volume, enables the
fenced Cold lifecycle, and points its plugin URL at the in-chart StatePool
Service. The chart deliberately enforces one Controller replica because the
local transcript lock is not yet a distributed request-admission boundary.

`worker.enabled=false` and `autoscaling.enabled=false` are safety gates. The
repository now contains the opt-in Sidecar adapter, but the example Worker
image remains a deliberate placeholder. Enable it only after publishing an
image that packages the model runtime and these implemented interfaces:

1. `statepool-worker-capability.v1` registration and TTL heartbeat;
2. exact model/tokenizer/State ABI reporting;
3. a Pod-IP `RWKV_WORKER_ENDPOINT` reachable from the Controller;
4. `/ready` and `/live`;
5. `/usr/local/bin/rwkv-statepool-drain`, which stops admission and polls until
   in-flight work and dirty States are zero.

The plugin treats a missing `unpersisted_state_slots` heartbeat as unknown and
will not report `safe_to_stop`. The Pod has a 180-second termination grace
period and a preStop hook; scale-down is stabilized and limited to one Pod per
minute. The drain client does not invent durability: the Controller must
snapshot each resident State through StatePool Lease/CAS and release it while
snapshot/release maintenance endpoints remain admitted. If that does not
happen before the deadline, preStop exits non-zero.

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

The Worker Deployment uses `replicas: 0` only while KEDA owns it; with
autoscaling disabled it uses `worker.replicaCount`. `idleReplicaCount: 0`, an
initial cooldown, a one-Pod-per-minute scale-down policy and the blocking
preStop drain are rendered explicitly.

KEDA itself is installed separately; the manifests were authored against
[KEDA v2.20.1](https://github.com/kedacore/keda/releases/tag/v2.20.1).
No KEDA source or Chart is vendored here.

The chart and metrics path were runtime-verified on kind 0.30.0/Kubernetes
1.34.0 with KEDA 2.20.1 and Prometheus 3.11.3. Three placement misses produced
the measured desired-replica sequence 0→1→3→0, all three Pods became Ready,
and all three preStop calls returned `safe_to_stop`. Commands, applied values,
live objects, 230 samples and checksums are in
[`bench/artifacts/statepool-keda-kind-20260827/`](../../bench/artifacts/statepool-keda-kind-20260827/README.md).

The current pending metric is a bounded demand signal: a cloud-allowed plan
miss increments it, and a compatible Worker registration clears it. A request
that safely falls back locally is not replayed remotely. Thus scale-from-zero
benefits subsequent requests and does not create ambiguous double execution.

The current Agent still performs its semantic Tool Gate through
`controller.modelUrls` before placement. An end-to-end 0→1 demo therefore
needs an always-reachable routing Sidecar (which can be local/edge) or must
drive the versioned placement endpoint directly. This limitation is explicit:
the checked-in chart does not yet prove that the GPU Worker pool can be the
only model endpoint at replica zero.

## Claim boundary

These assets and the linked kind artifact prove live control-plane scaling with
a protocol-faithful simulated Worker, not a live GPU scale-to-zero benchmark.
The RTX 4080 artifact separately proves real State bytes, forced process loss,
restore/continue/release and safe drain. A GPU Kubernetes result remains a
future evidence gate and must not be inferred by combining the two.
