# Live KEDA 0→1→N→0 control-plane cycle — 2026-08-27

The StatePool Helm/KEDA path has passed a live Kubernetes control-plane test
on kind/Kubernetes 1.34.0 with KEDA 2.20.1 and Prometheus 3.11.3.

Three exact-model cloud placement misses raised
`statepool_pending_requests` from 0 to 3. KEDA activated the Worker Deployment
from 0 to 1; the HPA then raised it to 3; all three Pods became Ready. Delayed
compatible Worker registration cleared the pending signal. The Deployment
returned to zero, and every Pod preStop reported `safe_to_stop` with zero
active requests and zero unpersisted States both locally and at the StatePool
control plane.

Measured relative timings were 3.234 seconds to the first desired replica,
7.017 seconds to desired replica count 3, 14.380 seconds to three Ready Pods,
31.248 seconds until desired replicas returned to zero, and 62.260 seconds
until the last terminating Pod disappeared.

This is deliberately a **control-plane simulation**: the Worker image
implements the real registration, heartbeat, readiness and preStop contract
but performs no inference and requests no GPU. The separate RTX 4080 artifact
proves real RWKV State snapshot/restore, forced Worker-process replacement and
the real Sidecar safe-drain result. No claim combines the two into a measured
GPU Kubernetes SLO.

The full environment, live objects, 230-sample transition series, KEDA events,
image identities, three unique drain results, simulator source and checksums
are under
[`bench/artifacts/statepool-keda-kind-20260827/`](../../bench/artifacts/statepool-keda-kind-20260827/README.md).
