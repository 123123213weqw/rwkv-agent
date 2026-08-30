# RWKV-7 vLLM plugin

This package registers the Hugging Face architecture `RWKV7ForCausalLM` with
vLLM 0.7.x without forking vLLM. It reuses the model repository's trusted
RWKV-7 implementation and kernel bridge while vLLM owns request scheduling,
sampling, streaming, cancellation, and the OpenAI server.

The adapter maps each sequence to a bounded recurrent cache slot:

- vLLM `conv_state` stores the attention and FFN token shifts;
- vLLM `ssm_state` stores the FP32 RWKV-7 matrix state;
- prompt chunks are grouped by exact query length;
- decode tokens from concurrent requests run as one RWKV batch.

The plugin is MIT-licensed with this repository. Its runtime dependencies,
vLLM and Transformers, are Apache-2.0 projects and remain external packages;
no vLLM source is vendored here.

## Supported runtime profile

- vLLM `0.7.x` legacy engine (`VLLM_USE_V1=0`)
- one CUDA device / tensor parallel size 1
- eager execution (`--enforce-eager`)
- local Hugging Face RWKV-7 repositories using `RWKV7Cache`
- ordinary and chat completions, streaming, sampling, stop strings, request
  cancellation, and `n > 1` sequence forking through vLLM

CUDA graphs, tensor/pipeline parallelism, LoRA, quantization, prefix caching,
and speculative decoding are rejected rather than silently producing an
incorrect recurrent State.

Install the package into the same environment as vLLM:

```bash
python -m pip install --no-deps -e integrations/vllm_rwkv7
```

Use `deploy/agent/start_vllm_rwkv7.sh` from the repository root for the pinned
server profile.

```bash
export RWKV_VLLM_PYTHON=/path/to/venv/bin/python
export RWKV_VLLM_MODEL_PATH=/models/rwkv7_15b_hf
export RWKV_VLLM_MODEL_NAME=rwkv7-1.5b-vllm
export RWKV_VLLM_HOST=127.0.0.1
export RWKV_VLLM_PORT=8120
export RWKV_VLLM_MAX_MODEL_LEN=2048
export RWKV_VLLM_MAX_NUM_SEQS=8
export RWKV_VLLM_MAX_BATCHED_TOKENS=2048
bash deploy/agent/start_vllm_rwkv7.sh
```

The start script records the API-server process group under
`${XDG_RUNTIME_DIR:-/tmp}/rwkv-vllm/`. Stop the frontend and its spawned engine
together with:

```bash
RWKV_VLLM_PORT=8120 bash deploy/agent/stop_vllm_rwkv7.sh
```

## OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8120/v1", api_key="local")
response = client.chat.completions.create(
    model="rwkv7-1.5b-vllm",
    messages=[{"role": "user", "content": "What is RWKV?"}],
    temperature=0,
    max_tokens=64,
)
print(response.choices[0].message.content)
```

## Verified machine profile

The frozen `bench/artifacts/rwkv-vllm-plugin-4090-20260830/` run used an RTX
4090, vLLM 0.7.3, and the local RWKV-7 1.5B HF checkpoint. Greedy output
matched direct HF generation exactly for the parity prompt. One simultaneous
8-request sample completed 128 generated tokens in 2443 ms wall time; this is
a single functional sample, not a P95 latency claim.
