# Configuration

Configuration has two layers:

1. the private shell environment file for paths, ports and credentials;
2. the JSON Web configuration for bounded discovery, fetch and Evidence limits.

## Environment file

Default path:

```text
~/.config/rwkv-agent/rwkv-agent.env
```

Source it before starting the three processes. No repository service-manager
wrapper reads this file automatically.

| Variable | Required | Purpose |
|---|---:|---|
| `RWKV_AGENT_PROJECT_ROOT` | yes | Repository checkout |
| `RWKV_AGENT_PYTHON` | yes | Python with Agent dependencies |
| `G1I_MODEL_PATH` | yes | G1I checkpoint |
| `G1I_RUNTIME_DIR` | yes | Compatible Albatross runtime |
| `G1I_PERSISTENT_STATE_CAPACITY` | no | Sidecar recurrent-state ownership limit |
| `G1I_PERSISTENT_STATE_TTL_SECONDS` | no | Idle GPU State expiry |
| `CUDA_VISIBLE_DEVICES` | yes | Single verified CUDA device |
| `RWKV_AGENT_HOST` / `RWKV_AGENT_PORT` | no | Rust Server bind address; defaults to loopback `8122` |
| `RWKV_AGENT_MODEL_URLS` | no | Sidecar endpoint list |
| `RWKV_AGENT_DATA_PLANE_URL` | no | Retrieval/Evidence Provider endpoint |
| `RWKV_AGENT_SESSION_DIR` | no | Durable Rust Session and Task Ledger directory |
| `RWKV_AGENT_TOOL_GATE_THRESHOLD` | model-specific | Semantic Search Gate threshold |
| `RWKV_AGENT_CHAT_STATE_CAPACITY` | no | Rust Runtime hot-session State LRU size |
| `RWKV_AGENT_WEB_CONFIG` | no | JSON config path |
| `RWKV_AGENT_WEB_API_PROVIDERS` | no | Ordered structured providers |
| `RWKV_AGENT_KNOWLEDGE_ENDPOINT` | no | External local knowledge index; see [knowledge setup](KNOWLEDGE_SETUP.md) |
| `TAVILY_API_KEY` | no | Enables Tavily when present |
| `GITHUB_TOKEN` | no | Raises GitHub API allowance |

Use [`.env.example`](../.env.example) as the source of truth. Never place API
keys in JSON config or command-line arguments.

Direct chat keeps at most `RWKV_AGENT_CHAT_STATE_CAPACITY` opaque GPU State IDs
in the Rust Runtime LRU. The default is three so one B4 research request can
still allocate its root plus four branches within the Sidecar's default
eight-state ownership limit. Tool turns release the chat State while external
I/O runs; the next direct turn rebuilds once from the durable transcript.
Expired, restarted or capacity-constrained Sidecars fall back to transcript
prefill rather than losing the conversation.

## Web configuration

[`configs/production.example.json`](../configs/production.example.json) enables
realtime retrieval with conservative limits. Enhanced candidate admission,
query compaction, domain pivot and one-hop expansion stay disabled in the
public Beta because they have not passed all end-to-end release gates.

The Agent always enables a bounded realtime path when `web_search` executes.
The JSON values still control SearXNG, fallback engines, fetch limits, timeouts,
page sizes, cache and private-network rejection.

## Discovery providers

- `github`: repository/profile/release/commit structures;
- `mediawiki`: encyclopedic entities;
- `crossref`: papers and publication metadata;
- `tavily`: general Web discovery, enabled only when a key is present;
- configured SearXNG engines;
- bounded HTML fallback.

These are capabilities, not hard-coded topic routes. A stock query and a
software query share the same planning and Evidence pipeline.

## Model-specific settings

`G1I_MODEL_ID`, `G1I_CONTEXT` and `RWKV_AGENT_TOOL_GATE_THRESHOLD` must describe
the actual checkpoint. Changing them without rerunning Tool Call and routing
benchmarks may silently reduce quality.
