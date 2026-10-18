# RWKV Agent Rust Server

`rwkv-agent-server-rs` is the canonical Rust HTTP control plane. It embeds the
RWKV browser workspace at `/`, Task wall at `/tasks` and readiness dashboard at
`/status`.

- `GET /live` — process liveness only
- `GET /ready` — dependency readiness; returns `503` while unavailable
- `GET /v1/openapi.json` and `/v1/schema.json` — machine-readable frontend contract
- `POST /v1/tasks` and `POST /v1/tasks/stream`
- `GET /v1/tasks?api_version=...&request_id=...&owner_id=...`
- `GET /v1/tasks/{task_id}` with the same versioned owner identity
- `POST /v1/tasks/{task_id}/resume|cancel`
- `POST /v1/research`
- `POST /v1/tools/call`
- optional owner-scoped `GET /v1/debug/traces*` on loopback

`GET /health`, `/v1/agent/run`, `/v1/agent/run_stream`,
`/v1/agent/run_stateful` and `/v1/task-ledger/*` are compatibility aliases;
they invoke the same Rust service methods rather than a parallel lifecycle.
The versioned request, event and error contracts are
[`contracts/agent-service-v1.openapi.json`](../../contracts/agent-service-v1.openapi.json)
and [`contracts/agent-service-v1.schema.json`](../../contracts/agent-service-v1.schema.json).
The human frontend guide is [`docs/HTTP_API.md`](../../docs/HTTP_API.md).

The embedded UI itself uses `/live`, `/ready`, canonical `/v1/tasks*` routes and
bounded `/v1/research`.
Compatibility aliases are not part of the frontend contract.

For frontend-only iteration, `rwkv-agent-web-dev-proxy-rs` serves the checked-in
TypeScript output from `web/` and proxies a strict allowlist to a loopback Rust
Controller. It rejects Debug, compatibility and direct Tool routes. See
[`docs/FRONTEND.md`](../../docs/FRONTEND.md).

It defaults to the isolated port `8122`, Sidecar `8417` and Python data plane
`8121`. The default semantic Gate threshold is the frozen Preview4922 13.3B
calibration value `-3.2`; deployments using another checkpoint must recalibrate
it. When a pasted document is active, a separately calibrated `-5.5` threshold
keeps document questions on the semantic tool path while greetings stay chat;
this is still the model Gate, not keyword routing. It does not replace or
restart another Controller.

```bash
rwkv-agent-data-plane --port 8121 --model-urls http://127.0.0.1:8417
rwkv-agent-server-rs \
  --port 8122 \
  --runtime-revision <runtime-commit-or-release> \
  --model-urls http://127.0.0.1:8417 \
  --data-plane-url http://127.0.0.1:8121

RWKV_AGENT_ENDPOINT=http://127.0.0.1:8122 rwkv-agent doctor
```

Local diagnostics are release-default `off`. `--debug-mode redacted|full`
enables the bounded Rust writer; `--debug-api` registers query/raw-file routes
only when the server listens on loopback. See
[`docs/DEBUG_TRACE.md`](../../docs/DEBUG_TRACE.md). Full traces contain private
request/tool bodies and are never training data.

CLI flags override `RWKV_AGENT_*` environment variables, which override the
isolated defaults. `GET /ready` identifies the runtime revision and reports
Model Sidecar, Data Plane, Sandbox, persistent-State capacity and Task Ledger
independently. Missing dependencies never select a different model path
silently. Readiness fails closed when capacity is unknown or exhausted;
liveness remains Provider-independent.

Command execution remains disabled unless both `--enable-command` and
`--command-workspace` are supplied. There is no unsafe fallback when Bubblewrap
is missing. On Ubuntu hosts that enable
`kernel.apparmor_restrict_unprivileged_userns=1`, an already loaded,
administrator-approved AppArmor profile that grants `userns` may be selected
with `RWKV_AGENT_BWRAP_APPARMOR_PROFILE=<profile>`. The runtime invokes it as
`aa-exec -p <profile> -- bwrap ...`; it never changes AppArmor policy, sudo,
capabilities or kernel settings itself.

See [`docs/SERVICE_PIPELINE.md`](../../docs/SERVICE_PIPELINE.md) for the request
flow, startup order, error codes and troubleshooting sequence.
