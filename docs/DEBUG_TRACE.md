# Rust Debug Trace runbook

Debug Trace is an opt-in, owner-scoped local diagnostic capture for the Rust
service pipeline. It is separate from the body-free
`long-lived-runtime-trace.v1` operational record. Full captures can contain
prompts, raw generations, tool arguments/results, command stdout/stderr and
final task bodies; they must not be committed, published, or used as
training/distillation data.

## Modes and configuration

The release default is `off`. In that mode no trace ID, directory, file, body
copy, writer queue, or query route is created.

| CLI flag | Environment | Default |
|---|---|---|
| `--debug-mode` | `RWKV_AGENT_DEBUG_MODE` | `off` |
| `--debug-dir` | `RWKV_AGENT_DEBUG_DIR` | `var/debug-traces/` |
| `--debug-retention-hours` | `RWKV_AGENT_DEBUG_RETENTION_HOURS` | `24` |
| `--debug-max-bytes` | `RWKV_AGENT_DEBUG_MAX_BYTES` | `2147483648` |
| `--debug-api` | `RWKV_AGENT_DEBUG_API` | `false` |

`redacted` retains identities, state lifecycle, stages, sizes, token counts,
timing, statuses and error codes while replacing body strings/objects with
size metadata. `full` retains the complete diagnostic payload. The raw API can
only be enabled on a loopback listen address; startup rejects any other
combination.

Example isolated launch:

```bash
rwkv-agent-server-rs \
  --host 127.0.0.1 --port 8122 \
  --debug-mode full \
  --debug-dir var/debug-traces \
  --debug-api
```

`GET /ready` reports `components.debug_trace` with `enabled`, `mode`,
`directory`, `writeable`, `queue_depth`, and `incomplete_total`. An enabled
but unwritable trace store does not bypass task State cleanup; the response is
annotated with `debug_capture.status=incomplete`.

## On-disk contract

Each unpredictable 128-bit `trace_id` owns one directory:

```text
<trace_id>/
  manifest.json
  service-events.jsonl
  model.jsonl
  tools.jsonl
  state.jsonl
  stream.jsonl
  task-record.json
  final-response.json
  SHA256SUMS
```

All incremental events also appear in `service-events.jsonl`. Their sequence
starts at one, increases strictly, begins with one `trace_started`, and ends
with one `trace_finished`. A Tokio bounded queue applies backpressure. Writer,
flush, sync, checksum, and rename errors make the manifest incomplete rather
than silently dropping data. Active directories use `.partial`; startup
recovers them as explicit `process_restart_partial_recovery` manifests.
Retention removes finalized owner-only directories by age and total bytes.
Symlinks and non-allowlisted raw-file names are rejected.

The schema is `contracts/debug-trace-v1.schema.json`.

## Owner-scoped API

The routes exist only when the debug API is explicitly enabled:

- `GET /v1/debug/traces?api_version=...&request_id=...&owner_id=...`
  supports `filter_request_id`, `task_id`, `session_id`, `after_trace_id`, and
  bounded `limit`.
- `GET /v1/debug/traces/{trace_id}` returns the manifest.
- `GET /v1/debug/traces/{trace_id}/events?after_sequence=N&limit=M` supports
  incremental TUI consumption.
- `GET /v1/debug/traces/{trace_id}/files/{kind}` accepts only
  `service-events`, `model`, `tools`, `state`, `stream`, `task-record`,
  `final-response`, or `checksums`.

Owner mismatches return Not Found. There is no directory listing, arbitrary
path, or range interface. Disconnecting a debug consumer has no relationship
to task cancellation; disconnecting the task stream retains the existing
cancel-and-State-release behavior.

## Verification and cleanup

Verify checksums locally inside one finalized trace directory:

```bash
sha256sum -c SHA256SUMS
```

Stop the isolated service before deleting active `.partial` data. To disable
capture, restart the isolated service without Debug flags (or set mode to
`off`); do not modify public `8118/8120`. Remove local captures only after any
required failure artifact has been reviewed:

```bash
rm -rf var/debug-traces
```
