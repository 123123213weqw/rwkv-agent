# RWKV StateServe RTX 4090 verification — 2026-08-30

## Classification

Live, single-node GPU protocol and continuous-batching verification.  This is
not a vLLM engine benchmark, not an Albatross comparison and not a multi-node
result.

## Fixed environment

- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9;
- driver: 550.142; PyTorch CUDA runtime: 12.4;
- backend: `hf_recurrent`;
- model: RWKV-7 G1G 1.5B, 1,527,404,544 parameters, FP16;
- source checkpoint revision: `41251fab280e3fba70a3fc49e843f3a034d49d33`;
- source checkpoint SHA-256: `441f70b096ad62442b5c33128bfe717c5d8529915c45a9709d4482016e8a0482`;
- API profile: context 8192, State capacity 16, physical batch 8,
  prefill chunk 64, greedy decode.

Full package versions, model conversion manifest and executed-source hashes are
in [`environment.json`](environment.json).

## Executed checks

1. `GET /v1/models` returned the pinned model ID.
2. Non-streaming `POST /v1/chat/completions` returned a standard
   `chat.completion`, finish reason and token usage.
3. Streaming Chat Completions returned incremental
   `chat.completion.chunk` SSE events, a usage-only chunk and `[DONE]`.
4. OpenAI Python SDK 1.65.1 parsed synchronous and streaming responses.
5. Eight simultaneous HTTP clients completed successfully through one process.
6. Scheduler telemetry observed physical batch 8 (`B8T1` six times).
7. The affected remote CPU conformance suites passed 23/23 tests.

Observed eight-client wall time was 2,193.246 ms for eight bounded 8-token
requests.  This single run is functional evidence, not a P50/P95 claim.  The
pool reported one retained system gate State after requests completed; request
rows were released.  The recorded 4,087 MiB GPU memory sample includes loaded
weights and the retained gate State.

## Files

- [`chat.json`](chat.json): synchronous response;
- [`chat.sse`](chat.sse): raw SSE response;
- [`openai-client.json`](openai-client.json): official OpenAI Python SDK result;
- [`concurrency.json`](concurrency.json): eight-client results;
- [`health.json`](health.json): full scheduler/State telemetry;
- [`models.json`](models.json): model discovery response;
- [`gpu.csv`](gpu.csv): final NVIDIA telemetry sample;
- [`server.log`](server.log): server lifecycle log;
- [`remote-focused-pytest.txt`](remote-focused-pytest.txt): affected remote conformance suites.

Generated text is retained only to prove that the model executed.  It is not a
quality evaluation of the checkpoint.
