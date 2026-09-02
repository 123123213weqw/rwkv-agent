# StatePool Cloud client integration

`rwkv-agent` keeps a narrow, disabled-by-default client boundary for the
independent [`statepool-cloud`](https://github.com/123123213weqw/statepool-cloud)
product. The cloud repository owns the service implementation, deployment,
OpenAI/vLLM Worker Adapter, observability, benchmarks and evidence.

## Default-off guarantee

Without `--cloud-plugin` or `RWKV_AGENT_CLOUD_PLUGIN=true`:

- no plugin HTTP client is constructed;
- no cloud endpoint is contacted;
- local model routing and State ownership are unchanged;
- PostgreSQL, S3, Kubernetes and vLLM are not runtime dependencies.

This behavior is a required regression test.

## What remains in this repository

- `crates/statepool-plugin-api`: versioned Rust wire types;
- `contracts/`: JSON Schema, OpenAPI and fixtures;
- `crates/agent-runtime/src/cloud_plugin.rs`: bounded HTTP client;
- `src/rwkv_agent/statepool_worker.py`: native RWKV Worker registration;
- `src/rwkv_agent/statepool_drain.py`: native Worker preStop helper;
- `crates/state-runtime` and `src/rwkv_runtime`: native State snapshot/restore.

## Start the external control plane

Follow the independent repository:

```bash
git clone https://github.com/123123213weqw/statepool-cloud.git
cd statepool-cloud
docker compose -f deploy/statepool/compose.yaml up --build statepool
```

Then enable the client explicitly:

```bash
rwkv-agent-server-rs \
  --cloud-plugin \
  --cloud-plugin-url=http://127.0.0.1:8130 \
  --cloud-plugin-fallback=local \
  --cloud-plugin-privacy=local_only \
  --cloud-model-id="$MODEL_ID" \
  --cloud-model-revision="$MODEL_REVISION" \
  --cloud-tokenizer="$TOKENIZER_REVISION" \
  --cloud-state-abi="$STATE_ABI"
```

The four model identity fields are atomic. Native State migration is allowed
only between exact-compatible identities. A Transformer/OpenAI-compatible
Worker uses `affinity_only` or `replay_only`; it never claims portable KV State.

## Compatibility

The current client contract is `statepool-plugin.v1`. Each release pins its
compatibility in [`COMPATIBILITY.md`](../COMPATIBILITY.md). Contract changes
must update Rust types, JSON Schema, OpenAPI, fixtures and both repositories'
CI before release.
