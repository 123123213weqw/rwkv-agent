# OpenAI-compatible StatePool Worker evidence

**Date:** 2026-08-27  
**Source:** `f914222f3d7b7b6068534af238c435fb77411e77`  
**Raw bundle:** [`bench/artifacts/statepool-openai-worker-20260827/`](../../bench/artifacts/statepool-openai-worker-20260827/README.md)

The optional standard-library adapter passed fake-upstream HTTP proxy,
logical-to-upstream model rewrite, StatePool registration/heartbeat, upstream
health gating, affinity header and drain rejection tests. The Rust placement
suite proves that `replay_only` and `affinity_only` never receive a raw-State
restore Lease, while legacy and explicit `native_export` RWKV Workers preserve
the existing exact-State lifecycle.

The default and OpenAI Helm overlays rendered on `WZU_Server`; the standalone
adapter image built and imported successfully. All Rust compilation, tests,
check and Clippy ran remotely. The complete Python suite passed 787 tests with
8 environment-dependent skips and 52 subtests.

This is adapter/control-plane conformance evidence, **not** a live vLLM,
Qwen3.5 GPU, throughput or model-quality result. A specific model profile stays
candidate-only until its immutable model/tokenizer revision, vLLM version,
hardware and raw run output are archived separately.
