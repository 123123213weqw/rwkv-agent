# RWKV-7 vLLM plugin RTX 4090 evidence

This frozen bundle records the first real vLLM engine run of the out-of-tree
RWKV-7 adapter. It is separate from the earlier StateServe-compatible API:
vLLM itself owns the OpenAI routes, request scheduler, sampler, streaming, and
sequence lifecycle in this run.

Profile:

- NVIDIA GeForce RTX 4090, driver 550.142;
- vLLM 0.7.3 legacy engine, eager mode;
- RWKV-7 1.5B HF checkpoint, FP16 weights and FP32 recurrent matrix State;
- maximum 8 live sequences and 2048 model tokens.

Validated:

- model registration through `vllm.general_plugins`;
- `/v1/models`, `/v1/completions`, and `/v1/chat/completions`;
- sync and SSE streaming with final usage;
- OpenAI Python SDK 1.65.1 sync and stream parsing;
- `temperature`, `top_p`, deterministic seed, and `n=2` sequence forking;
- 8 simultaneous requests, all HTTP 200;
- deterministic request-State isolation before and after unrelated traffic;
- eight disconnected streams followed by eight successful requests;
- exact greedy continuation parity with direct HF generation.

The 8-request sample generated 16 tokens per request and completed in
2443.204 ms wall time. This is one functional sample rather than a statistically
valid throughput or latency benchmark.

Files containing requests and model text are test artifacts, not quality
evaluations of the small checkpoint.

