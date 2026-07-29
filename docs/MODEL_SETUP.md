# Model setup

## Verified checkpoint

The public Beta is verified with:

- model: `rwkv7-g1i_preview4922-13.3b-20260720-ctx12288.pth`;
- model page: [BlinkDL temp latest training models](https://huggingface.co/BlinkDL/temp-latest-training-models/blob/main/rwkv7-g1i_preview4922-13.3b-20260720-ctx12288.pth);
- context: 12,288 tokens;
- runtime mode: single CUDA device, no pipeline parallelism;
- tested hardware: one 32 GB V100.

The checkpoint and CUDA runtime are not distributed in this repository.

## Runtime contract

`G1I_RUNTIME_DIR` must point to an Albatross-compatible runtime containing:

```text
rwkv7_fast_v3a.py
rwkv/
```

The current integration expects the RWKV tokenizer pipeline used by the
official RWKV Gradio runtime. The official demonstration repository is
[RWKV-Gradio-1](https://huggingface.co/spaces/BlinkDL/RWKV-Gradio-1).

Runtime files, CUDA extensions and checkpoint versions must be kept together.
A random RWKV `.pth` file is not automatically compatible with G1I Tool Call
templates or the current state layout.

## Required environment

```bash
G1I_MODEL_PATH=/absolute/path/to/model.pth
G1I_RUNTIME_DIR=/absolute/path/to/albatross-runtime
G1I_MODEL_ID=rwkv7-g1i-preview4922-13.3b
G1I_CONTEXT=12288
CUDA_VISIBLE_DEVICES=0
RWKV_AGENT_TOOL_GATE_THRESHOLD=-3.2
```

The `-3.2` Search Gate threshold is calibrated only for Preview4922 13.3B. Do
not copy it to a different checkpoint without a routing regression test.

## Validation

```bash
rwkv-agent-service doctor
rwkv-agent-service start
curl -fsS http://127.0.0.1:8118/health | python -m json.tool
rwkv-agent doctor
```

Health must report the intended model ID, context, CUDA device, zero worker
error and an empty state pool when idle.

## Unsupported combinations

- custom multi-GPU pipeline parallelism is not a Beta release path;
- CPU inference is not supported by the G1I Sidecar;
- Hugging Face Transformers checkpoints are supported by the Legacy Web path,
  not automatically by the state-native Agent Sidecar;
- quantized and converted checkpoints require their own correctness benchmark.
