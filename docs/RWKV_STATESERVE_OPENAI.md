# RWKV StateServe: OpenAI/vLLM-compatible serving

RWKV StateServe exposes the existing RWKV recurrent scheduler through the HTTP
wire format used by OpenAI-compatible clients and vLLM deployments.  It is an
API compatibility layer, not a fork of vLLM and not a claim that RWKV uses a
Transformer KV cache.

The data path remains RWKV-native:

```text
OpenAI client
  -> /v1/chat/completions
  -> ContinuousBatchEngine
  -> AlbatrossChunkScheduler or HFRecurrentScheduler
  -> bounded RWKV recurrent State per active row
```

## Supported surface

| Surface | Status |
|---|---|
| `GET /v1/models` | supported |
| `POST /v1/completions` | synchronous and SSE streaming |
| `POST /v1/chat/completions` | synchronous and SSE streaming |
| `stream_options.include_usage` | supported |
| `system`, `developer`, `user`, `assistant` text messages | supported |
| RWKV `System/User/Assistant` prompt rendering | supported |
| `temperature=0`, `top_p=1`, `n=1` | supported deterministic profile |
| Sampling, logprobs, multimodal content | not yet supported; rejected explicitly |
| OpenAI tool-call wire format | not on the base endpoint; Agent tool routes remain separate |
| Native State snapshot/restore/fork | supported by the separate `/v1/states/*` contract when the backend declares it |

The standard chat endpoint creates an ephemeral inference row.  Long-lived
assistants should use the native State routes and the optional StatePool Cloud
Plugin rather than pretending an OpenAI request ID is a portable State ID.

## Start an HF recurrent profile

Use a symlink or directory name containing only Python-package-safe characters
for Transformers 4.x remote-code models, for example `rwkv7_15b_hf`.

```bash
export G1I_BACKEND=hf_recurrent
export G1I_HF_MODEL_PATH=/models/rwkv7_15b_hf
export G1I_MODEL_ID=rwkv7-g1g-1.5b-hf
export G1I_MODEL_REVISION=REPLACE_WITH_IMMUTABLE_REVISION
export G1I_TOKENIZER_ID=rwkv_vocab_v20230424
export G1I_CONTEXT=8192
export G1I_STATE_CAPACITY=16
export G1I_MAX_BATCH_SIZE=8
export G1I_PREFILL_CHUNK_SIZE=64

rwkv-stateserve --host 127.0.0.1 --port 18118
```

Bind to loopback by default.  Put authentication and TLS at the ingress or API
gateway before exposing the service outside a trusted network.

## Chat Completions

```bash
curl -sS http://127.0.0.1:18118/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"rwkv7-g1g-1.5b-hf",
    "messages":[
      {"role":"system","content":"Answer briefly."},
      {"role":"user","content":"What is RWKV?"}
    ],
    "temperature":0,
    "max_tokens":64
  }'
```

## SSE streaming

```bash
curl -N http://127.0.0.1:18118/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"rwkv7-g1g-1.5b-hf",
    "messages":[{"role":"user","content":"Say hello."}],
    "temperature":0,
    "max_tokens":16,
    "stream":true,
    "stream_options":{"include_usage":true}
  }'
```

The stream uses `chat.completion.chunk` objects and terminates with
`data: [DONE]`, matching OpenAI/vLLM client expectations.

## Python OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18118/v1", api_key="unused")
response = client.chat.completions.create(
    model="rwkv7-g1g-1.5b-hf",
    messages=[{"role": "user", "content": "What is RWKV?"}],
    temperature=0,
    max_tokens=64,
)
print(response.choices[0].message.content)
```

## Runtime verification

The maintained evidence bundle is
[`bench/artifacts/rwkv-stateserve-4090-20260830/`](../bench/artifacts/rwkv-stateserve-4090-20260830/README.md).
It records an RTX 4090 run of the RWKV-7 G1G 1.5B HF recurrent profile,
including synchronous chat, SSE chunks, usage accounting and an eight-client
continuous-batching probe.
