# RWKV Agent quickstart

The supported service path is:

```text
rwkv-agent -> Rust Server :8122 -> RWKV Sidecar :8417
                               -> Retrieval Data Plane :8121
```

The old Python Controller, Web preview and service-manager wrapper have been
removed. All three backend processes bind to loopback by default.

## 1. Requirements

- Linux CUDA host with a compatible RWKV checkpoint and Albatross runtime;
- Python 3.10 or newer for the two narrow external providers;
- Rust toolchain for the Server and CLI;
- Bubblewrap when the optional command tool is enabled.

## 2. Install providers

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[realtime,agent]'
```

Set the checkpoint and runtime identity for the Sidecar:

```bash
export G1I_MODEL_PATH=/absolute/path/to/model.pth
export G1I_RUNTIME_DIR=/absolute/path/to/albatross-runtime
export G1I_MODEL_ID=your-verified-model-id
export G1I_CONTEXT=16384
export CUDA_VISIBLE_DEVICES=0
```

Model-specific State capacity and Search Gate thresholds must come from a
verified profile. Do not reuse a threshold calibrated for another checkpoint.

## 3. Build the Rust binaries

```bash
cargo build --release --locked -p rwkv-agent-server -p rwkv-agent-cli
```

Repository contributors must follow `AGENTS.md`; its remote-only Rust build
policy takes precedence over the example above.

## 4. Start the split stack

Terminal 1:

```bash
source .venv/bin/activate
rwkv-g1i-sidecar --host 127.0.0.1 --port 8417
```

Terminal 2:

```bash
source .venv/bin/activate
rwkv-agent-data-plane \
  --host 127.0.0.1 \
  --port 8121 \
  --model-urls http://127.0.0.1:8417
```

Terminal 3:

```bash
./target/release/rwkv-agent-server-rs \
  --host 127.0.0.1 \
  --port 8122 \
  --model-urls http://127.0.0.1:8417 \
  --data-plane-url http://127.0.0.1:8121 \
  --session-dir ./var/sessions
```

Command execution remains disabled unless both `--enable-command` and an
isolated `--command-workspace` are supplied.

## 5. Verify and use

```bash
./target/release/rwkv-agent --endpoint http://127.0.0.1:8122 health
./target/release/rwkv-agent --endpoint http://127.0.0.1:8122 doctor
./target/release/rwkv-agent --endpoint http://127.0.0.1:8122 ask "你好"
```

Use `GET /live` for process liveness and `GET /ready` for Provider,
Sandbox, State-capacity and Task-Ledger readiness. See
[SERVICE_PIPELINE.md](SERVICE_PIPELINE.md) for TaskSpec, streaming,
cancellation and Debug Trace endpoints.

For remote access, keep the Server on loopback and forward only the Rust port:

```bash
ssh -N -L 8122:127.0.0.1:8122 user@gpu-host
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8122 rwkv-agent doctor
```

The Beta Server has no public authentication, TLS or rate limiting. Stop the
processes in reverse order: Rust Server, Data Plane, then Sidecar.
