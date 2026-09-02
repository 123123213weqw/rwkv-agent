# Rust Service Pipeline

## Canonical flow

```text
rwkv-agent
  -> rwkv-agent-server-rs :8122
    -> AgentService
      -> TaskLedger + SessionStore
      -> AgentLoop + strict Tool/Answer parser
        -> RWKV Sidecar :8417
        -> Data Plane :8121
        -> optional bounded Command Sandbox
      -> versioned response or NDJSON event stream
```

The Rust control plane is authoritative for request identity, TaskSpec
normalization, budgets, tool sequencing, durable task checkpoints and State
release. The current CUDA Sidecar and retrieval/evidence Data Plane remain
narrow external providers until their separate Rust parity gates pass. The
former Python Controller has been removed rather than retained as a second
Agent loop.

## Endpoint contract

| Method and path | Role |
|---|---|
| `GET /live` | Process liveness; never calls an external Provider |
| `GET /ready` | Model, Data Plane, Sandbox, persistent-State capacity and Task Ledger readiness |
| `GET /v1/openapi.json` | Canonical OpenAPI 3.1 document for frontend tooling |
| `GET /v1/schema.json` | Versioned request, stream-event and error JSON Schema |
| `POST /v1/tasks` | Canonical synchronous TaskSpec or legacy-message run |
| `POST /v1/tasks/stream` | Canonical versioned NDJSON task event stream |
| `GET /v1/tasks` | Versioned, owner-scoped task summaries; requires `api_version`, `request_id`, and `owner_id` query fields |
| `GET /v1/tasks/{id}` | Owner-scoped durable task record |
| `POST /v1/tasks/{id}/resume` | Owner-scoped explicit resume |
| `POST /v1/tasks/{id}/cancel` | Owner-scoped idempotent cancellation |
| `POST /v1/research` | Bounded parallel recurrent-State research |
| `POST /v1/tools/call` | Strict direct tool call |
| `GET /v1/debug/traces*` | Opt-in, owner-scoped local Debug Trace query/API; absent by default |

`/health`, `/v1/agent/run`, `/v1/agent/run_stream`,
`/v1/agent/run_stateful` and `/v1/task-ledger/*` are compatibility aliases.
They share the canonical handlers and do not create a second Agent loop. The
embedded UI and external clients both use the owner-scoped canonical
`/v1/tasks` API.

Every canonical mutation carries:

```json
{
  "api_version": "rwkv-agent.service.v1",
  "request_id": "request-...",
  "owner_id": "session:...",
  "session_id": "..."
}
```

Task records persist the four identities. A repeated `request_id` with the
same Owner, Session and TaskSpec returns the completed record; a changed
payload is a conflict. Owner mismatch is returned as not found so one Owner
cannot inspect or mutate another Owner's task.

Stream events carry the same identities, a strictly increasing `sequence`, an
optional `task_id` and exactly one terminal `final` or `error` event. The
machine-readable contracts are `contracts/agent-service-v1.openapi.json` and
`contracts/agent-service-v1.schema.json`; the running server exposes the same
bytes at `/v1/openapi.json` and `/v1/schema.json`. See
[`HTTP_API.md`](HTTP_API.md) for the frontend integration contract.
Dropping the response body cancels the durable task immediately; an already
running Provider operation finishes only its atomic boundary, after which the
Controller performs no further model/tool action and releases the State.

## Error model

Errors retain the compatibility `error` string and add:

```json
{
  "status": "error",
  "error_detail": {
    "code": "unavailable",
    "message": "...",
    "retryable": true
  }
}
```

Codes are `invalid_request`, `not_found`, `conflict`, `unavailable`,
`unsupported`, `cancelled`, `deadline_exceeded` and `internal`. Missing Model
Sidecar or Data Plane capacity is `unavailable`; a disabled capability is
`unsupported`. Neither condition silently selects another model or Controller.

## Configuration and startup

Precedence is CLI flag, then `RWKV_AGENT_*` environment variable, then the
isolated default. Important identities and endpoints are:

- `--runtime-revision` / `RWKV_AGENT_RUNTIME_REVISION`
- `--model-urls` / `RWKV_AGENT_MODEL_URLS`
- `--data-plane-url` / `RWKV_AGENT_DATA_PLANE_URL`
- `--session-dir` / `RWKV_AGENT_SESSION_DIR`
- `--max-run-seconds` / `RWKV_AGENT_MAX_RUN_SECONDS`
- `--shutdown-grace-seconds` / `RWKV_AGENT_SHUTDOWN_GRACE_SECONDS`
- `--debug-mode` / `RWKV_AGENT_DEBUG_MODE` (`off`, `redacted`, `full`)
- `--debug-dir` / `RWKV_AGENT_DEBUG_DIR`
- `--debug-retention-hours` / `RWKV_AGENT_DEBUG_RETENTION_HOURS`
- `--debug-max-bytes` / `RWKV_AGENT_DEBUG_MAX_BYTES`
- `--debug-api` / `RWKV_AGENT_DEBUG_API` (loopback only)
- server `127.0.0.1:8122`, Data Plane `127.0.0.1:8121`, Sidecar
  `127.0.0.1:8417`

Debug Trace is release-default `off`. When explicitly enabled, responses and
stream metadata carry `trace_id` and `debug_capture`; `off` emits neither and
creates no files. See [`DEBUG_TRACE.md`](DEBUG_TRACE.md) for the body/privacy
boundary, bounded writer, on-disk schema, APIs, checksum verification and
cleanup procedure.

Start the Sidecar and Data Plane first, then `rwkv-agent-server-rs`. Check in
order:

```text
GET /live
GET /ready
rwkv-agent doctor
rwkv-agent task --spec task.json --task-id smoke-1
```

Command tools remain disabled unless both `--enable-command` and
`--command-workspace` are present. Missing Bubblewrap/AppArmor support keeps
the Sandbox unavailable; there is no host-shell fallback.

`/ready` fails closed when any configured Sidecar omits persistent-State
capacity, reports no free slot, or is unreachable. Graceful shutdown cancels
active tasks, waits for their bounded Provider boundary, releases cached chat
States and reports a failure instead of hiding an incomplete cleanup.

## Current boundary

This milestone validates service correctness, not load performance. It does
not claim GPU↔CPU snapshot/restore, high-concurrency capacity, full-screen TUI,
production deployment, real Webhooks or repository-wide Rust migration.
