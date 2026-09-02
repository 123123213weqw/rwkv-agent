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

## Optional StatePool Cloud Plugin

The Rust server keeps the Cloud Plugin disabled unless
`RWKV_AGENT_CLOUD_PLUGIN=true`. Disabled mode constructs no plugin HTTP client
and requires no cloud dependencies.

| Variable | Default | Purpose |
|---|---|---|
| `RWKV_AGENT_CLOUD_PLUGIN` | `false` | Enable the out-of-process plugin |
| `RWKV_AGENT_CLOUD_PLUGIN_URL` | `http://127.0.0.1:8130` | Plugin endpoint |
| `RWKV_AGENT_CLOUD_PLUGIN_FALLBACK` | `local` | `local` or `fail_closed` |
| `RWKV_AGENT_CLOUD_PLUGIN_PRIVACY` | `local_only` | `local_only`, `hybrid` or `cloud_allowed` |
| `RWKV_AGENT_CLOUD_PLUGIN_LATENCY_SLO_MS` | `5000` | Placement latency objective |
| `RWKV_AGENT_CLOUD_PLUGIN_PREFERRED_ZONE` | unset | `local`, `edge` or `cloud` |
| `RWKV_AGENT_CLOUD_MODEL_ID` | unset | Exact model identity; required when enabled |
| `RWKV_AGENT_CLOUD_MODEL_REVISION` | unset | Immutable model revision; required when enabled |
| `RWKV_AGENT_CLOUD_TOKENIZER` | unset | Exact tokenizer identity; required when enabled |
| `RWKV_AGENT_CLOUD_STATE_ABI` | unset | Exact recurrent-State ABI; required when enabled |
| `RWKV_AGENT_CLOUD_STATE_LIFECYCLE` | `false` | Enable fenced State lifecycle transport independently of placement |
| `RWKV_AGENT_CLOUD_STATE_TARGET_TIER` | `cold` | Durable snapshot target: `warm` or `cold` |
| `RWKV_AGENT_CLOUD_LEASE_TTL_SECONDS` | `120` | Writer Lease TTL used by lifecycle operations |
| `RWKV_AGENT_CLOUD_LIFECYCLE_TIMEOUT_SECONDS` | `180` | Per-request snapshot/restore transfer timeout |

All four model identity fields are an atomic configuration unit. Supplying
only some of them is rejected at startup. `local_only` is enforced by the host
even if a plugin incorrectly returns a remote plan.

State lifecycle remains a second, default-off switch. When enabled, handshake
requires both `leases` and `state_lifecycle`; lifecycle requests validate the
complete model identity, State version, owner, fencing token and checksum.
Placement failures may fall back only before a State has been committed. Once
a request carries a committed `StateReference`, the host fails closed rather
than silently re-prefilling and executing the same Session locally.

See [STATEPOOL_CLOUD_PLUGIN.md](STATEPOOL_CLOUD_PLUGIN.md) for the protocol and
current implementation boundary.

The OpenAI/vLLM Worker Adapter and its deployment variables are maintained in
[`statepool-cloud`](https://github.com/123123213weqw/statepool-cloud). This
repository keeps only the native RWKV Sidecar Worker variables below.

Use [`.env.example`](../.env.example) as the source of truth. Never place API
keys in JSON config or command-line arguments.

Direct chat keeps at most `RWKV_AGENT_CHAT_STATE_CAPACITY` opaque GPU State IDs
in the Rust Runtime LRU. The default is three so one B4 research request can
still allocate its root plus four branches within the Sidecar's default
eight-state ownership limit. Tool turns release the chat State while external
I/O runs; the next direct turn rebuilds once from the durable transcript.
Expired, restarted or capacity-constrained Sidecars fall back to transcript
prefill rather than losing the conversation.

That paragraph describes the unchanged default. With
`RWKV_AGENT_CLOUD_STATE_LIFECYCLE=true`, a safe direct-chat turn is snapshotted
and committed before its Worker State is released; the LRU then holds a
versioned durable reference rather than a GPU allocation. The following turn
must restore that exact reference and is never allowed to silently fall back to
transcript. An uncertain post-execution commit marks the Session
`blocked_hot`, returns the completed answer with an observable persistence
error, and rejects later automatic execution until reconciliation.

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
