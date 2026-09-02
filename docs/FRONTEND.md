# RWKV Agent frontend

The browser UI is a thin same-origin client of the canonical Rust Controller.
It does not contain a second Agent loop, Task Ledger, session database, direct
Sidecar client, direct Data Plane client, or unrestricted command surface.

## Product layout

The current layout adapts reusable interaction patterns studied in
[x-harness-rs](https://github.com/123123213weqw/x-harness-rs):

- a collapsible workspace sidebar with a prominent new-session action;
- a centered conversation surface and persistent bottom composer;
- compact Agent/Research mode selection next to the composer;
- a durable Task wall with a contextual detail drawer;
- Context, Runtime and Task inspector tabs;
- trajectory cards for phases, stages and Tool Calls;
- responsive drawer navigation on narrow screens.

The implementation, DOM structure, CSS tokens, TypeScript client and RWKV
visual identity are maintained in this repository. No compiled upstream UI,
plugin graph, RPC client, brand asset, package namespace or Node production
server is shipped. Third-party design-study attribution is retained in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Source map

```text
web/index.html                 semantic shell and accessible view structure
web/app.css                    RWKV light/dark tokens, layout and responsive UI
web/src/api-client.ts          typed canonical API and strict NDJSON parser
web/src/app.ts                 identity, view, Task and inspector state
web/dist/*.js                  checked-in browser output embedded by Rust
web/tests/api-client.test.mjs  contract/parser tests
web/tests/mock-controller.mjs  deterministic browser smoke fixture
crates/agent-server/src/lib.rs embedded same-origin production surface
crates/agent-server/src/bin/web_dev_proxy.rs
                               loopback-only development proxy
```

The checked-in JavaScript is generated from TypeScript without a framework or
runtime package dependency:

```bash
tsc --project web/tsconfig.json
node --test web/tests/api-client.test.mjs
```

Release serving remains Rust-only. `rwkv-agent-server-rs` embeds the generated
assets at compile time. The optional development proxy is also Rust:

```bash
rwkv-agent-web-dev-proxy-rs \
  --target http://127.0.0.1:8122 \
  --web-root web \
  --port 5173
```

The proxy is loopback-only and forwards only the ordinary frontend contract:
`/live`, `/ready`, canonical Task list/run/stream/detail/control routes and
`/v1/research`. It rejects compatibility aliases, Debug Trace routes, direct
Tool calls, arbitrary Workspace paths and non-loopback upstreams.

## Browser state

The frontend persists these logical identities in browser-local storage:

- one `owner_id` for the local browser profile;
- one active `session_id` per conversation;
- a bounded list of recent local session labels;
- the latest `task_id` immediately after `task_created`.

Every API action receives a fresh `request_id`. Recent sessions are navigation
metadata only; the browser does not become a transcript source of truth. Task
refresh and reconnect always reconcile against owner-scoped
`GET /v1/tasks/{task_id}`.

## Stream and lifecycle rules

`web/src/api-client.ts` treats `/v1/tasks/stream` as NDJSON, not SSE or a
WebSocket. It supports arbitrarily split lines, preserves unknown non-terminal
events, requires strictly increasing sequence numbers, and accepts one terminal
`final` or `error`. A Stop action aborts the stream; the Rust Controller then
persists cancellation at its next boundary and releases recurrent State. The UI
reconciles the durable Task instead of assuming that closing the response was
instant cancellation.

Task detail uses the safe Controller projection only. It renders Stage status,
attempts, event timeline, revision, recovery count and final summary. It never
requests raw Debug files or full private tool bodies from the Task Ledger.

## System status

The status view calls both endpoints independently:

- `/live` proves that the Rust process can serve requests;
- `/ready` reports Model, Data Plane, Sandbox, State capacity, Task Ledger,
  runtime revision and Agent limits, including an HTTP `503` body when degraded.

The compatibility `/health` route is not used.

## Security boundary

- Ordinary UI code calls no `/v1/debug/*`, `/v1/tools/call`, compatibility alias,
  Sidecar, Data Plane or arbitrary command endpoint.
- Model and tool output is inserted with `textContent`; it is never interpreted
  as HTML. The browser smoke fixture includes hostile-looking markup to keep
  this boundary visible.
- The embedded and development static surfaces set a same-origin Content
  Security Policy and `nosniff`.
- `owner_id` is logical isolation, not authentication. Keep the server on
  loopback or add a separately authorized authenticated gateway before remote
  multi-user use.
- Research is intentionally bounded to four branches and two rounds in the MVP.
