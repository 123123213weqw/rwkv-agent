# KEDA 0→1→N→0 control-plane evidence — 2026-08-27

## Verdict

A real Kubernetes/KEDA control plane completed the following cycle against the
StatePool plugin:

```text
0 Worker Pods
  -> three exact-model cloud plan misses
  -> statepool_pending_requests = 3
  -> KEDA activation: 0 -> 1
  -> HPA external metric: 1 -> 3
  -> three Worker Pods Ready
  -> compatible Worker registration clears pending demand
  -> HPA/KEDA scale-down: 3 -> 0
  -> every Pod preStop receives control-plane safe_to_stop
```

`verified-summary.json` records `success=true`, a desired replica sequence of
`[0, 1, 3, 0]`, three Ready replicas, a return to zero, and three distinct
`safe_to_stop` results. `transitions.jsonl` is the 230-sample raw time series;
`transitions-compact.csv` keeps only observable state changes.

## Measured timings

Times are relative to the first placement request. They describe this local
single-node kind environment and are not production GPU SLOs.

| Event | Seconds |
|---|---:|
| KEDA Active / first desired replica | 3.234 |
| desired replicas reached 3 | 7.017 |
| Ready replicas reached 3 | 14.380 |
| plugin pending demand cleared | 21.457 |
| desired replicas returned to 0 | 31.248 |
| last terminating Worker Pod disappeared | 62.260 |

The 30-second gap between desired zero and Pod disappearance is the configured
termination grace window. It is not State snapshot time.

## Environment and executed images

| Component | Executed version/image |
|---|---|
| kind | `v0.30.0` |
| Kubernetes server/client | `v1.34.0` |
| KEDA | `2.20.1` operator, metrics server and admission webhook |
| Helm | `v3.18.6` |
| Prometheus | `3.11.3` |
| StatePool plugin | `rwkv-statepool-cloud-plugin:0.3.0-beta.2-dev`, image ID `sha256:396fbb6b4e7474255341b476d4aca9847503d1b692be7503ff88764228669040` |
| Worker simulator | `rwkv-statepool-worker-keda-sim:20260827`, image ID `sha256:7c6d2d5148ee11e60c697391e1d73345dd9c4e5e872f8b417b5390198981941c` |

The cluster was a one-node kind cluster on `WZU_Server`. Exact version and
image inspection output is retained in `versions.txt`,
`keda-deployments.json`, `kind-node-images.json`, and
`local-image-inspect.json`.

## Workload and claim boundary

The Worker image is a standard-library HTTP simulator, retained in
`mock_worker.py`, `drain.py`, and `mock-worker.Dockerfile`. It implements the
real versioned Worker registration/heartbeat/drain contract, delays
registration for 15 seconds so HPA can observe pending demand, and reports
zero active requests and zero dirty States. It performs **no model inference**
and requests zero GPU resources in this kind run.

Therefore this artifact proves:

- the checked-in Helm Deployment and KEDA `ScaledObject` run on Kubernetes;
- Prometheus supplies the StatePool external metrics used by KEDA;
- KEDA performs 0→1 activation and HPA performs 1→3 scaling;
- compatible registration resolves the bounded scale-from-zero signal;
- scale-down invokes the configured preStop command on every Worker;
- each simulated Worker and the StatePool control plane agree on
  `safe_to_stop`, `active_requests=0`, and `unpersisted_states=0`.

It does **not** prove GPU scheduling, model throughput, snapshot latency, or
dirty-State persistence inside Kubernetes. The separate
[`statepool-4080-worker-kill-20260827`](../statepool-4080-worker-kill-20260827/README.md)
artifact proves exact RWKV State snapshot/restore after real GPU Worker-process
loss and the real Sidecar preStop gate. Together they validate the data-plane
and control-plane slices without presenting their measurements as one
end-to-end GPU Kubernetes run.

## Trigger

The authoritative run started from a fresh plugin registry and zero Worker
replicas. It submitted three `statepool-plan-request.v1` requests with:

```text
model_id  = rwkv7-keda-sim
revision  = keda-sim-20260827
tokenizer = rwkv_vocab_v20230424
state_abi = rwkv7-keda-sim-state-v1
privacy   = cloud_allowed
```

All three returned the expected `no_compatible_worker`/local fallback plan and
incremented the bounded pending metric. The exact requests and responses are
in `trigger-responses.json`.

The executed driver is retained as `run_experiment.py`. Its environment paths
identify the archived host layout; the portable steps are:

```bash
kind create cluster --name statepool-keda
kubectl apply -f <official-keda-2.20.1-release-manifest>
kubectl wait -n keda --for=condition=available deploy/keda-operator --timeout=180s
kubectl apply -f prometheus.yaml
helm upgrade --install statepool <repo>/deploy/statepool/helm/statepool \
  --namespace statepool-keda --create-namespace -f values.yaml
python run_experiment.py
```

The two locally built images must first be loaded with `kind load docker-image`.
The KEDA release manifest itself is intentionally not copied into this
repository.

## Safe-drain proof

`prestop-results.json` contains one unique event for each of the three Pod and
Worker identities. Every result contains:

```json
{
  "status": "safe_to_stop",
  "active_requests": 0,
  "unpersisted_states": 0,
  "control_plane": { "status": "safe_to_stop" }
}
```

The reconnecting log collector read the same terminated-container buffer more
than once. To avoid publishing 72 identical copies per Pod,
`worker-lifecycle-unique.jsonl` contains the parsed, byte-for-byte unique JSON
events and `prestop-results.json` contains the three authoritative drain
outcomes. The original sampling series and Kubernetes events remain retained.

## Artifact index

- `verified-summary.json`: derived pass/fail result and event timings;
- `transitions.jsonl`, `transitions.csv`: raw 250 ms observations;
- `transitions-compact.csv`: state-transition-only view;
- `trigger-responses.json`: exact demand requests and placement replies;
- `prestop-results.json`, `worker-lifecycle-unique.jsonl`: unique Worker
  lifecycle and drain outcomes;
- `scaledobject.yaml`, `hpa-final.yaml`, `worker-deployment.yaml`: live objects;
- `helm-manifest.yaml`, `helm-values-all.yaml`, `values.yaml`: rendered and
  applied configuration;
- `prometheus-targets.json`, `plugin-metrics-final.prom`: scrape and metric
  evidence;
- `kubernetes-events.*`, `keda-operator.txt`: controller/event evidence;
- `versions.txt`, `nodes.*`, `*-images.json`: environment identity;
- `mock_worker.py`, `drain.py`, `mock-worker.Dockerfile`: simulator source.

`SHA256SUMS` covers every retained file other than itself.
