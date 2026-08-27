# OpenAI-compatible Worker adapter

## Purpose

Use an existing inference engine instead of implementing another one. The
adapter connects StatePool's exact-model placement and Worker lifecycle to an
upstream that already implements the OpenAI HTTP surface. vLLM is the primary
target, but the boundary is protocol-based and does not import its Python
package or source tree.

This keeps the product story honest:

- **RWKV route:** bounded recurrent State, `native_export`, fenced
  snapshot/restore, Hot/Warm/Cold lifecycle;
- **vLLM/OpenAI route:** broad Transformer/model support,
  `affinity_only`, full context replay, optional upstream prefix-cache benefit;
- **cross-model route:** transcript or Context Capsule only—never raw State/KV.

API compatibility, a checked model profile and State portability are three
different claims. A successful adapter conformance test proves only the first.

## Start

Start an upstream server independently, for example on port 8000. Then export
one exact logical model identity:

```bash
export RWKV_STATEPOOL_URL=http://127.0.0.1:8130
export RWKV_WORKER_ID=vllm-qwen-a
export RWKV_WORKER_ENDPOINT=http://127.0.0.1:8128
export RWKV_WORKER_ZONE=cloud
export RWKV_WORKER_MODEL_ID=qwen3.5-9b-vllm
export RWKV_WORKER_MODEL_REVISION=sha256:REPLACE_WITH_IMMUTABLE_REVISION
export RWKV_WORKER_TOKENIZER=REPLACE_WITH_IMMUTABLE_TOKENIZER_REVISION
export RWKV_WORKER_STATE_ABI=context-replay.v1
export RWKV_WORKER_STATE_SLOTS=16
export RWKV_WORKER_MAX_BATCH=16
export RWKV_OPENAI_UPSTREAM_URL=http://127.0.0.1:8000
export RWKV_OPENAI_UPSTREAM_MODEL=Qwen/Qwen3.5-9B
rwkv-openai-worker
```

The process first registers as `starting`. `/ready` becomes HTTP 200 only
after both the upstream `/health` probe and StatePool registration succeed.

With Compose, the adapter stays behind an explicit profile and expects the
upstream on the host by default:

```bash
docker compose -f deploy/statepool/compose.yaml \
  --profile openai-worker up -d statepool openai-worker
```

Replace both immutable revision placeholders before treating the profile as a
certified model deployment. The Helm equivalent is the opt-in
`values-openai-worker.yaml` overlay.

## Plan and call

```bash
python scripts/statepool_openai_client.py \
  --model-id qwen3.5-9b-vllm \
  --model-revision sha256:REPLACE_WITH_IMMUTABLE_REVISION \
  --tokenizer REPLACE_WITH_IMMUTABLE_TOKENIZER_REVISION
```

The result contains the explainable StatePool plan, completion and selected
Worker ID. Pass the returned ID as `--affinity-worker-id` on a later request.
The placement hint is non-authoritative and does not create a State object or
restore Lease.

## Drain contract

`POST /v1/statepool/drain` stops new admission, reports active requests, asks
StatePool to mark the Worker draining and returns `safe_to_stop` only when
in-flight work reaches zero. `unpersisted_states` is always zero because the
adapter owns no portable runtime State. The upstream engine remains under its
own operator's lifecycle policy.

## Current evidence boundary

The checked-in tests use a protocol-faithful fake HTTP upstream and prove
registration, health gating, model rewrite, proxy response, affinity header
and drain rejection. A live vLLM/model/GPU run is still required before a
specific model profile becomes `runtime_verified`.
