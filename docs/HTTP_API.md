# RWKV Agent HTTP API v1

This is the frontend integration contract for the canonical Rust control plane.
The machine-readable documents are:

- [`../contracts/agent-service-v1.openapi.json`](../contracts/agent-service-v1.openapi.json)
- [`../contracts/agent-service-v1.schema.json`](../contracts/agent-service-v1.schema.json)
- runtime `GET /v1/openapi.json`
- runtime `GET /v1/schema.json`

The API version is always `rwkv-agent.service.v1`. The examples below use a
same-origin base URL. The server does not enable cross-origin access by default;
a separately hosted development frontend should proxy only the ordinary
canonical surface to the Rust server. The checked-in loopback-only
`rwkv-agent-web-dev-proxy-rs` does this without exposing compatibility aliases,
Debug Trace, direct Tool calls or Providers. See [`FRONTEND.md`](FRONTEND.md).

## 1. Frontend boundary

The browser talks only to `rwkv-agent-server-rs`:

```text
browser
  -> Rust agent-server
    -> AgentService / TaskLedger / AgentLoop
      -> RWKV Sidecar
      -> Retrieval Data Plane
      -> bounded Tool / Sandbox
```

Do not expose the Sidecar, Data Plane or command sandbox directly to a browser.
The Rust server defaults to `127.0.0.1:8122`; it has no built-in API key, OAuth
or user authentication. `owner_id` is a logical isolation key, not proof of
identity. Put an authenticated same-origin gateway in front before any remote
or multi-user deployment.

## 2. Canonical endpoints

The embedded browser routes are `/`, `/tasks` and `/status`; assets remain
same-origin under `/assets/`. These routes are UI navigation, not additional
control planes.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/live` | Process liveness; never contacts Providers |
| `GET` | `/ready` | Model, Data Plane, Sandbox, State capacity and ledger readiness |
| `GET` | `/v1/openapi.json` | OpenAPI 3.1 contract |
| `GET` | `/v1/schema.json` | Request, stream-event and error JSON Schema |
| `GET` | `/v1/tasks` | Owner-scoped task summaries |
| `POST` | `/v1/tasks` | Synchronous durable task run |
| `POST` | `/v1/tasks/stream` | Streaming durable task run over NDJSON |
| `GET` | `/v1/tasks/{task_id}` | Safe durable task detail |
| `POST` | `/v1/tasks/{task_id}/resume` | Resume from the durable checkpoint |
| `POST` | `/v1/tasks/{task_id}/cancel` | Idempotent cancellation request |
| `POST` | `/v1/research` | Bounded stateful research |
| `POST` | `/v1/tools/call` | Direct operator-level tool invocation |

Normal frontend traffic should use `/v1/tasks/stream`, `/v1/tasks` and the task
detail/control endpoints. `/v1/tools/call` is for an operator console; an Agent
chat UI should let the runtime select tools.

## 3. Identity and idempotency

Canonical mutation bodies carry:

```json
{
  "api_version": "rwkv-agent.service.v1",
  "request_id": "request-019f...",
  "owner_id": "user-42",
  "session_id": "chat-019f..."
}
```

| Field | Lifetime | Rule |
|---|---|---|
| `api_version` | application | Literal `rwkv-agent.service.v1` |
| `request_id` | one logical request | Generate a new UUID-like value for each action; reuse only for an intentional idempotent replay |
| `owner_id` | signed-in user/workspace | Persist across sessions; 1-128 ASCII letters, digits or `._:-/` |
| `session_id` | conversation | Persist for all turns in one conversation; 1-128 non-control characters |
| `task_id` | durable task | Optional client ID or server-generated ID; persist after `task_created` |

A repeated `request_id` with the same owner, session and TaskSpec replays the
durable result. Reusing it with a changed payload returns `conflict`. Cross-owner
task access returns `not_found` instead of revealing that the task exists.

The run/research/tool decoders still accept omitted `request_id` and `owner_id`
for legacy clients, but canonical frontend code must send both. List, detail,
resume and cancel always require them.

## 4. Liveness and readiness

### `GET /live`

Always returns HTTP `200` while the process can serve requests:

```json
{
  "status": "alive",
  "api_version": "rwkv-agent.service.v1",
  "control_plane": "rust",
  "runtime_revision": "0.3.0-beta.2"
}
```

### `GET /ready`

Returns `200` when all required dependencies are ready and `503` otherwise.
The body includes `components`, `configuration`, model identity, State capacity,
tools and sandbox status. Frontends must use the HTTP status as the readiness
decision and may render the component body for diagnostics.

Do not use `/health`: it is a compatibility route that returns the readiness
body with HTTP `200` even when the service is unavailable.

## 5. Run a task

### Simple task

```http
POST /v1/tasks/stream
Content-Type: application/json
Accept: application/x-ndjson
```

```json
{
  "api_version": "rwkv-agent.service.v1",
  "request_id": "request-001",
  "owner_id": "user-42",
  "session_id": "chat-001",
  "message": "Explain the current repository architecture."
}
```

### Workspace task

`working_directory` selects the configured sandbox workspace. It must not be a
free-form field for untrusted public users.

```json
{
  "api_version": "rwkv-agent.service.v1",
  "request_id": "request-002",
  "owner_id": "user-42",
  "session_id": "chat-001",
  "message": "Fix the inventory total and run the verifier.",
  "working_directory": "/srv/workspaces/inventory"
}
```

### Structured long task

```json
{
  "api_version": "rwkv-agent.service.v1",
  "request_id": "request-003",
  "owner_id": "user-42",
  "session_id": "chat-002",
  "task_spec": {
    "schema_version": 1,
    "objective": "Repair the inventory calculation and verify the result.",
    "acceptance_criteria": [
      "The two known source defects are fixed",
      "All specified verification commands pass",
      "The final answer summarizes changed files and verification"
    ],
    "constraints": [
      "Do not modify generated or protected files"
    ],
    "verification_commands": [
      "python3 verify.py"
    ],
    "requires_mutation": true,
    "working_directory": "/srv/workspaces/inventory",
    "stages": [
      {
        "id": "inspect",
        "objective": "Inspect the implementation and reproduce the failure"
      },
      {
        "id": "repair",
        "objective": "Apply the minimal repair",
        "depends_on": ["inspect"],
        "requires_mutation": true
      },
      {
        "id": "verify",
        "objective": "Run the exact verifier and report the result",
        "depends_on": ["repair"]
      }
    ]
  }
}
```

When both `message` and `task_spec.objective` are present, their trimmed values
must match. The same rule applies to top-level and TaskSpec working directories.
Unknown JSON fields are rejected.

`POST /v1/tasks` accepts the same body but waits for the final response instead
of producing a stream.

## 6. NDJSON streaming

`POST /v1/tasks/stream` returns:

```http
Content-Type: application/x-ndjson; charset=utf-8
Cache-Control: no-store
X-Accel-Buffering: no
```

This is not SSE and not WebSocket. Each non-empty line is one JSON object with
the following envelope:

```json
{
  "api_version": "rwkv-agent.service.v1",
  "request_id": "request-003",
  "owner_id": "user-42",
  "session_id": "chat-002",
  "task_id": "task-...",
  "sequence": 1,
  "type": "task_created"
}
```

Known event types:

| Type | Important payload | Meaning |
|---|---|---|
| `task_created` | `task_id`, `status` | Durable task exists; persist its ID immediately |
| `stage_started` | `stage_id`, `stage_index`, `stage_count` | TaskSpec stage began |
| `phase` | `phase=routing\|tool\|decoding` | Coarse UI progress |
| `delta` | `text`, `delta`, `replace`, `output_tokens` | Model output update |
| `stage_completed` | stage fields | TaskSpec stage checkpoint committed |
| `final` | `response` | Unique successful terminal event |
| `error` | `error`, optional `error_detail` | Unique failed terminal event |
| `runtime_event` | extension data | Forward-compatible unknown runtime event |

`sequence` is strictly increasing per stream. A well-formed stream has exactly
one terminal `final` or `error`; ignore data after the terminal event. Preserve
unknown non-terminal event types for logging instead of failing the whole UI.

Browser parser:

```ts
export async function streamTask(
  body: unknown,
  onEvent: (event: Record<string, unknown>) => void,
  signal?: AbortSignal,
) {
  const response = await fetch("/v1/tasks/stream", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "application/x-ndjson",
    },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(await response.text());
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminal = false;

  while (!terminal) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = done ? "" : lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      onEvent(event);
      terminal = event.type === "final" || event.type === "error";
      if (terminal) break;
    }
    if (done) break;
  }

  if (!terminal) throw new Error("task stream ended without a terminal event");
}
```

Aborting or dropping the response asks the runtime to cancel the durable task.
The current atomic Provider/tool boundary may finish, but no later action should
start and allocated recurrent State is released.

## 7. Task list and detail

All three query identity fields are required:

```http
GET /v1/tasks?api_version=rwkv-agent.service.v1&request_id=request-004&owner_id=user-42
```

```json
{
  "api_version": "rwkv-agent.service.v1",
  "request_id": "request-004",
  "owner_id": "user-42",
  "status": "ok",
  "durable": true,
  "counts": {
    "total": 3,
    "running": 1,
    "complete": 1,
    "failed": 1
  },
  "tasks": [
    {
      "id": "task-001",
      "session_id": "chat-002",
      "message": "Repair the inventory calculation…",
      "kind": "agent",
      "status": "running",
      "route": "tool",
      "state": "active",
      "tool_count": 2,
      "created_unix_ms": 1788336000000,
      "elapsed_ms": 4120,
      "error": null,
      "revision": 3,
      "recovery_count": 0
    }
  ]
}
```

The current task wall is bounded to 100 records and has no public pagination
cursor. Use task detail for durable stage/event state:

```http
GET /v1/tasks/task-001?api_version=rwkv-agent.service.v1&request_id=request-005&owner_id=user-42
```

The safe response contains TaskSpec counts/previews, stage status, ledger events,
revision, recovery count, trace identity, error and a final summary. It does not
return the complete private prompt or full tool bodies.

## 8. Cancel and resume

Both endpoints use the owner-scoped control body:

```json
{
  "api_version": "rwkv-agent.service.v1",
  "request_id": "request-006",
  "owner_id": "user-42"
}
```

```http
POST /v1/tasks/task-001/cancel
POST /v1/tasks/task-001/resume
```

Cancel is idempotent. Resume does not repeat a stage already committed as
succeeded; it continues from the durable Task Ledger boundary. Refresh task
detail after either operation instead of deriving lifecycle state solely from
the button response.

## 9. Research and direct tools

### Research

```json
{
  "api_version": "rwkv-agent.service.v1",
  "request_id": "request-007",
  "owner_id": "user-42",
  "session_id": "research-001",
  "message": "Compare the current official release notes and repository tag.",
  "branch_width": 4,
  "max_rounds": 2
}
```

`branch_width` is `1..4`; `max_rounds` is `1..3`. Research is synchronous in
service v1.

### Direct tool call

```json
{
  "api_version": "rwkv-agent.service.v1",
  "request_id": "request-008",
  "owner_id": "user-42",
  "session_id": "chat-002",
  "name": "web_search",
  "arguments": { "query": "RWKV latest official release" }
}
```

Do not expose an unrestricted tool form to ordinary users. `run_command` also
requires the server to start with command execution enabled, an explicit
workspace and an available sandbox; there is no host-shell fallback.

## 10. Errors and retries

Non-streaming errors use:

```json
{
  "api_version": "rwkv-agent.service.v1",
  "request_id": "request-008",
  "status": "error",
  "error": "model sidecar unavailable",
  "error_detail": {
    "code": "unavailable",
    "message": "model sidecar unavailable",
    "retryable": true
  }
}
```

| Code | Typical HTTP status | Retry automatically |
|---|---:|---|
| `invalid_request` | 400 | No |
| `not_found` | 404 | No |
| `conflict` | 409 | No |
| `unavailable` | 503 | Yes, bounded backoff |
| `unsupported` | 501 | No |
| `cancelled` | 409 | No |
| `deadline_exceeded` | 504 | Yes, bounded backoff |
| `internal` | 500 | No |

Use `error_detail.retryable` as the machine decision. Keep `error` for a concise
UI message and operator logs.

## 11. Debug API

Debug routes exist only when the server starts with `--debug-api`, and startup
rejects that option on a non-loopback listen address:

```text
GET /v1/debug/traces
GET /v1/debug/traces/{trace_id}
GET /v1/debug/traces/{trace_id}/events
GET /v1/debug/traces/{trace_id}/files/{kind}
```

They require the versioned request/owner query identity. Trace listing supports
`filter_request_id`, `task_id`, `session_id`, `after_trace_id` and `limit`.
Event listing supports `after_sequence` and `limit`. Full debug captures may
contain prompts, file content and tool arguments; keep this surface out of the
ordinary frontend. See [`DEBUG_TRACE.md`](DEBUG_TRACE.md).

## 12. Compatibility routes

The following routes remain only for old clients and are excluded from the
canonical OpenAPI document:

```text
GET  /health
GET  /v1/task-ledger
GET  /v1/task-ledger/{task_id}
POST /v1/task-ledger/{task_id}/resume
POST /v1/task-ledger/{task_id}/cancel
POST /v1/agent/run
POST /v1/agent/run_stream
POST /v1/agent/run_stateful
POST /v1/agent/gate
```

They call the same Rust handlers; there is no second Agent loop. New frontend
code must not use them. The embedded UI is also pinned to the canonical routes.

## 13. Recommended frontend state model

Persist these records independently:

```ts
type Conversation = {
  ownerId: string;
  sessionId: string;
};

type RunningTask = {
  taskId: string;
  requestId: string;
  sessionId: string;
  lastSequence: number;
  stageId?: string;
  phase?: "routing" | "tool" | "decoding";
  status: "running" | "complete" | "failed" | "cancelled";
};
```

The UI should persist `task_id` as soon as `task_created` arrives, render stream
events optimistically, and reconcile with `GET /v1/tasks/{task_id}` after a
terminal event, reconnect, cancel or resume.
