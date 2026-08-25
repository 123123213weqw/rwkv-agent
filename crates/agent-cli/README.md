# RWKV Agent CLI

`rwkv-agent` is the terminal client for the Agent HTTP API. It contains no model
runtime, crawler, index or sandbox implementation; those live in the server and
data-plane packages listed in [`docs/CODEMAP.md`](../../docs/CODEMAP.md).

Common commands:

```bash
rwkv-agent health
rwkv-agent ready
rwkv-agent doctor
rwkv-agent ask "hello"
rwkv-agent task --spec task.json --task-id inventory-fix-1
```

The default endpoint is the isolated Rust service at `http://127.0.0.1:8122`.
`ask`, `task`, `research` and direct tool calls attach Service v1
`request_id/owner_id/session_id` identities. `task` is the canonical long-task
entry and reads a strict TaskSpec v1 JSON file; `ask` is an explicit legacy
message-to-TaskSpec conversion performed by the server contract.

With `--json`, successful and failed HTTP calls both write one complete JSON
document to stdout. Structured server `error_detail`, `request_id` and Owner
identity are preserved for automation; when local Debug Trace is enabled,
`trace_id` and `debug_capture` are preserved as well. Human mode renders the
concise message. Debug capture is configured on the server, not in the CLI.

Installation and release packaging remain under [`cli/`](../../cli/README.md)
for compatibility with existing users.
