# Quickstart

This guide starts the `v0.3.0-beta.1` Agent on one local CUDA host. The current
public service script starts one model Sidecar and one Controller. It never
creates SSH tunnels or contacts a private server.

## Requirements

- Linux with an NVIDIA CUDA GPU;
- Python 3.10 or newer;
- Rust toolchain for the terminal client;
- `curl`;
- RWKV G1I model checkpoint;
- compatible Albatross runtime containing `rwkv7_fast_v3a.py`.

The verified Preview4922 13.3B setup fits one 32 GB V100. Other cards and
quantized runtimes require separate validation.

## Install

```bash
git clone https://github.com/123123213weqw/rwkv-search.git
cd rwkv-search
python -m venv .venv
source .venv/bin/activate
pip install -e '.[realtime,agent]'

cd cli
./install.sh
cd ..
```

Ensure `$HOME/.local/bin` is in `PATH`.

## Configure

From the repository root:

```bash
rwkv-agent-service init
$EDITOR ~/.config/rwkv-agent/rwkv-agent.env
```

At minimum, replace:

```bash
RWKV_AGENT_PROJECT_ROOT=/absolute/path/to/rwkv-search
RWKV_AGENT_PYTHON=/absolute/path/to/rwkv-search/.venv/bin/python
G1I_MODEL_PATH=/absolute/path/to/model.pth
G1I_RUNTIME_DIR=/absolute/path/to/albatross-runtime
```

The environment file is mode `0600`. Do not commit it.

## Validate and start

```bash
rwkv-agent-service doctor
rwkv-agent-service start
rwkv-agent doctor
```

The two local endpoints are:

- Sidecar: `http://127.0.0.1:8118`;
- Controller: `http://127.0.0.1:8120`.

## Use

```bash
rwkv-agent chat
rwkv-agent ask "Explain recurrent state in RWKV."
rwkv-agent tool web-search "RWKV latest official repository update"
rwkv-agent research --branches 4 --rounds 2 \
  "Compare the latest official progress across RWKV repositories."
```

Interactive commands include `/status`, `/web`, `/knowledge`, `/research`,
`/longtext`, `/session` and `/json`.

## Optional SearXNG

```bash
cd deploy/searxng
# Replace the example secret_key in settings.yml before use.
docker compose up -d
curl 'http://127.0.0.1:8888/search?q=rwkv&format=json'
```

If SearXNG is unavailable, the Agent uses configured structured providers and
the bounded fallback engine. Search quality depends on network egress.

## Remote GPU host

Run the installation and service on the GPU host, then forward the Controller:

```bash
ssh -N -L 8120:127.0.0.1:8120 user@gpu-host
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8120 rwkv-agent doctor
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8120 rwkv-agent chat
```

Do not bind the Beta Controller to a public interface without adding your own
authentication, TLS, rate limiting and outbound network policy.

## Stop

```bash
rwkv-agent-service status
rwkv-agent-service stop
```

Only processes referenced by PID files under `RWKV_AGENT_STATE_DIR` are stopped.
