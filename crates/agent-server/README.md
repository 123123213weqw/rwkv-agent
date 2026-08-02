# RWKV Agent Rust Server

`rwkv-agent-server-rs` exposes the existing CLI-compatible endpoints while the
control plane runs in Rust:

- `GET /health`
- `POST /v1/agent/gate`
- `POST /v1/tools/call`
- `POST /v1/agent/run`
- `POST /v1/agent/run_stateful`

It defaults to the isolated port `8122`, Sidecar `8417` and Python data plane
`8121`. The default semantic Gate threshold is the frozen Preview4922 13.3B
calibration value `-3.2`; deployments using another checkpoint must recalibrate
it. When a pasted document is active, a separately calibrated `-5.5` threshold
keeps document questions on the semantic tool path while greetings stay chat;
this is still the model Gate, not keyword routing. It does not replace or
restart the Python Controller on port `8120`.

```bash
rwkv-agent-data-plane --port 8121 --model-urls http://127.0.0.1:8417
cargo run -p rwkv-agent-server -- \
  --port 8122 \
  --model-urls http://127.0.0.1:8417 \
  --data-plane-url http://127.0.0.1:8121

RWKV_AGENT_ENDPOINT=http://127.0.0.1:8122 rwkv-agent doctor
```

Command execution remains disabled unless both `--enable-command` and
`--command-workspace` are supplied. There is no unsafe fallback when Bubblewrap
is missing.
