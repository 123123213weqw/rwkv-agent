# Related work and non-overlap

StatePool is intentionally a narrow integration layer. It does not reimplement
GPU virtualization, Kubernetes scheduling, autoscaling, object storage,
metrics storage, dashboards or a general LLM gateway.

## Why RWKV State is a cloud primitive

The original RWKV paper formulates the architecture so training can be
parallelized while autoregressive inference uses recurrent, constant-complexity
State ([Peng et al., 2023](https://arxiv.org/abs/2305.13048)). RWKV-7 retains
constant memory and constant inference time per token while making the State
transition more expressive
([RWKV-7 “Goose”](https://arxiv.org/abs/2503.14456)). The upstream repository
provides both GPT-mode State construction and RNN-mode continuation
([RWKV-v7](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7)).

Those properties create a serving option that is different from either keeping
a complete Transformer KV cache resident or discarding inference memory and
re-prefilling the transcript: persist a bounded recurrent State, release the
Worker, and restore that State on an exact-compatible Worker later.

The architecture does **not** make State self-describing or portable by itself.
StatePool therefore requires exact equality of model ID, immutable revision,
tokenizer and State ABI, plus checksum and version/fencing validation.

## Layer-by-layer comparison

| Project/approach | What it already solves | What StatePool consumes | What StatePool adds rather than rebuilding |
|---|---|---|---|
| [HAMi](https://github.com/Project-HAMi/HAMi) | heterogeneous accelerator sharing, isolation and device-aware Kubernetes placement | optional Worker resource annotations/profile | Session State location, exact State ABI, restore cost and dirty-State drain constraints |
| [AIBrix](https://github.com/vllm-project/aibrix) | cloud-native GenAI gateway, autoscaling, multi-engine serving, routing and KV-oriented infrastructure | future optional gateway/route adapter | long-lived RWKV recurrent State ownership, Lease/fencing and Hot/Warm/Cold lifecycle |
| [KServe](https://kserve.github.io/website/) | standardized predictive/generative model serving, rollout/routing and scale-to-zero | future optional `InferenceService`/gateway profile | State-safe scale-down and restore semantics for a personal Agent Session |
| [KEDA](https://keda.sh/docs/2.20/concepts/scaling-deployments/) | generic event-driven 0↔1 and HPA-backed 1↔N scaling | Prometheus scaler over StatePool metrics | State-aware demand/backlog metrics and a drain gate; KEDA remains the scaler |
| Prometheus/Grafana | time-series storage, PromQL and visualization | scrape endpoint and provisioned dashboard | domain metrics: State tier events, restoration, avoided Prefill, GPU seconds and estimated cost |
| PostgreSQL/S3 | durable transactions and immutable object storage | planned MetadataStore/StateStore adapters | schema, CAS/fencing rules, exact compatibility and lifecycle policy |
| Sticky Worker baseline | simplest State reuse with no transfer | benchmark baseline A | ability to release an idle Worker after a safe snapshot |
| Stateless re-prefill baseline | any compatible Worker can serve from transcript | benchmark baseline B and incompatibility fallback | avoid repeated Prefill while retaining Worker elasticity |

HAMi became a CNCF Incubating project in July 2026
([CNCF project page](https://www.cncf.io/projects/hami/)). That validates the
importance of accelerator sharing; it does not eliminate the application-level
question of what happens to a long-lived Agent State when a GPU Pod disappears.

AIBrix explicitly asks users to bring an inference engine and supplies the
infrastructure around it, including request routing, autoscaling, KV management
and model lifecycle
([AIBrix overview](https://aibrix.readthedocs.io/latest/getting_started/overview.html)).
StatePool follows the same compositional principle: its optional adapter should
feed AIBrix rather than fork it.

KEDA distinguishes the 0↔1 activation phase from HPA-managed 1↔N scaling. The
StatePool `ScaledObject` maps bounded pending demand to activation and estimated
decode backlog to scaling. It does not claim the YAML alone proves a safe N→0;
the Worker drain/snapshot handshake and a cluster test remain required.

## The unresolved gap

Existing systems typically schedule one of these:

- Pods and accelerator fractions;
- stateless inference requests;
- models/adapters;
- Transformer prefix/KV cache blocks;
- generic queue depth or request concurrency.

This project schedules a different object: an owner-scoped, versioned,
long-lived **Agent recurrent State** that may be Hot on a Worker, Warm on host
memory, Cold in an object store, or deliberately dropped for transcript
re-prefill. Its correctness constraints are Session Lease, monotonically
increasing fencing token, immutable snapshot, version CAS, exact State ABI and
privacy-aware placement.

## Claim matrix

| Claim | Current status |
|---|---|
| Optional process boundary, default local behavior unchanged | implemented and regression-tested |
| Worker registry and explainable placement | implemented in the development plugin |
| Single-process Lease/fencing/CAS | implemented and remotely tested |
| Atomic LocalFS snapshot/restore of generic State bytes | implemented and remotely tested |
| Compose, Helm, KEDA, ServiceMonitor and dashboard configuration | authored and statically validated |
| Live RWKV Albatross Sidecar CPU export/import boundary | implemented and conformance-tested |
| Live RWKV export/import after forced Worker process loss | measured on one RTX 4080 with PostgreSQL/S3; fresh process on the same physical GPU |
| PostgreSQL distributed Lease/CAS | implemented; two-client container integration passed |
| S3/MinIO Cold State adapter | implemented; MinIO container integration passed |
| Real GPU preStop safe-drain gate | measured after restore/release; rebuildable system root remained and dirty user State was zero |
| Measured live-Kubernetes/KEDA 0→1→N→0 | not measured |

This table is the release claim boundary; competition material must not promote
an item from “not implemented/measured” without linked evidence.
