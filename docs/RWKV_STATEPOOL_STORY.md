# From local RWKV Agent to stateful Serverless inference

## 1. The historical arc

RWKV began from a model-architecture question: can a sequence model keep the
parallel training advantages associated with Transformers while running
autoregressive inference as an RNN? The 2023 paper answered with a model that
can be expressed in parallel or recurrent form. RWKV-7 “Goose” extended the
State evolution and published models, code and corpus information under open
licenses. The community then carried RWKV from research code into CPU, mobile
and cross-platform local applications, including
[`rwkv.cpp`](https://github.com/RWKV/rwkv.cpp) and the local-first
[`RWKV_APP`](https://github.com/RWKV-APP/RWKV_APP).

The product implication is larger than “another model served behind an OpenAI
API.” A recurrent model has a compact, continuously updated computational
State. In a personal assistant, that State is valuable because the user returns
to the same Session repeatedly.

## 2. What RWKV can do well in this project

RWKV's strengths here are specific:

1. **Long-lived continuation:** append a new turn to an existing recurrent
   State instead of reconstructing all previous activations.
2. **Many isolated Sessions:** each owner has a separate bounded State slot;
   the runtime batches active rows without merging their contexts.
3. **Local/edge operation:** recurrent memory does not grow with history length,
   making desktop and edge deployments structurally attractive.
4. **Fast root-State forks:** stable system/tool prefixes can be prefetched once
   and copied into independent task States.
5. **State as an optimization object:** snapshot, restore, tier, meter and place
   State independently from model weights.

RWKV does not magically guarantee infinite factual memory, safe cross-model
State conversion, distributed consistency or cloud elasticity. Transcript/RAG
still carries auditable long-term information; only exact-compatible raw State
can move; the control plane must provide Lease/fencing and persistence.

## 3. The user problem

A personal assistant should feel continuously available, but continuously
reserving a GPU for every mostly-idle person is economically wrong. Two common
alternatives are also weak:

- keep a Sticky Worker alive so its State survives—good latency, poor idle cost;
- release the Worker and re-prefill the full transcript next time—good
  elasticity, repeated tokens and latency.

StatePool introduces a third path: preserve the bounded State, release the GPU,
then restore the exact-compatible State when the user returns.

## 4. The one-sentence story

> 每个人都可以长期拥有一个 AI 助手，但不需要长期占用一张 GPU。
>
> The assistant stays resident; the GPU does not have to.

This connects the existing local product and the cloud competition without
changing the product identity:

- local computer: privacy root, transcript, credentials, low-cost/low-latency
  edge Worker;
- cloud: elastic compatible Workers for bursts and heavy work;
- StatePool: protocol and policy that decide where a new request runs and how a
  versioned State survives Worker lifecycle;
- upstream cloud-native projects: Kubernetes/KEDA/HAMi/AIBrix/Prometheus rather
  than in-repository replacements.

Other models strengthen rather than dilute this story. An optional vLLM or
OpenAI-compatible Worker handles broad Transformer/model coverage; it replays
the transcript and uses only a same-Worker cache hint. RWKV remains the route
with a bounded, portable recurrent State and therefore demonstrates the deeper
scale-to-zero lifecycle. The product is one assistant control plane with
different truthful State capabilities, not one invented inference engine.

## 5. Application and market wedge

The initial wedge is not generic stateless chat. It is a workload with all four
properties:

1. the same user or Agent returns repeatedly;
2. prior computation is expensive enough to preserve;
3. traffic has long idle gaps or bursts;
4. privacy/SLO/cost can select local, edge or cloud execution.

Examples include coding/research assistants, private knowledge companions,
long-running operational Agents, intermittent edge assistants and multi-user
Agent hosting. The buyer/user value is measurable in avoided Prefill tokens,
GPU idle minutes, GPU-hours per 100 Sessions, restore P95 and cost per task—not
an abstract “AI cloud platform” claim.

## 6. Why this is defensible

The durable differentiation is the State/session contract, not a collection of
YAML files:

- exact model/revision/tokenizer/State-ABI compatibility;
- owner-scoped single writer;
- monotonically increasing fencing token;
- immutable checksum-verified State object;
- version CAS;
- Hot/Warm/Cold lifecycle and restore-cost-aware placement;
- drain completion before scale-down;
- State-aware FinOps.

Kubernetes, KEDA, object storage and dashboards can be replaced by compatible
upstream implementations. The protocol and benchmark remain the project.
