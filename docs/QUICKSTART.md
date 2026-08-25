# RWKV Agent quickstart

This guide starts the `v0.3.0-beta.2` Agent on one local CUDA host. The
canonical path is the Rust CLI on `8122` → Rust Server → external model Sidecar
and Data Plane. The existing `rwkv-agent-service` script remains a compatibility
bootstrap for the Python Controller on `8120`; it never creates SSH tunnels or
contacts a private server. See [Rust Service Pipeline](SERVICE_PIPELINE.md) for
the canonical startup and readiness contract.

## Requirements

- Linux with an NVIDIA CUDA GPU;
- Python 3.10 or newer;
- Rust toolchain for the terminal client;
- `curl`;
- RWKV G1I model checkpoint;
- compatible Albatross runtime containing `rwkv7_fast_v3a.py`.

The verified Preview4922 13.3B setup fits one 32 GB V100. Other cards and
quantized runtimes require separate validation.

## Client-only setup

The Rust client does not need CUDA or model weights. Install it on a laptop and
connect to an already configured Controller:

```bash
git clone https://github.com/123123213weqw/rwkv-agent.git
cd rwkv-agent
./cli/install.sh --client-only

ssh -N -L 8122:127.0.0.1:8122 user@gpu-host
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8122 rwkv-agent doctor
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8122 rwkv-agent
```

Prebuilt CLI archives, when attached to a tagged Beta release, contain the
binary, CLI guide, license and an adjacent SHA-256 file. See
[`cli/README.md`](../cli/README.md) for installation and supported targets.

The remaining sections install the complete backend on a Linux CUDA host.

## Full backend install

```bash
git clone https://github.com/123123213weqw/rwkv-agent.git
cd rwkv-agent
python -m venv .venv
source .venv/bin/activate
pip install -e '.[realtime,agent]'

./cli/install.sh
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
RWKV_AGENT_PROJECT_ROOT=/absolute/path/to/rwkv-agent
RWKV_AGENT_PYTHON=/absolute/path/to/rwkv-agent/.venv/bin/python
G1I_MODEL_PATH=/absolute/path/to/model.pth
G1I_RUNTIME_DIR=/absolute/path/to/albatross-runtime
```

The environment file is mode `0600`. Do not commit it.

## Canonical Rust service pipeline

Start the configured model Sidecar, then the external Data Plane, and finally
the Rust control plane. The tagged release archives contain the CLI; build the
Server from the same tagged source tree:

```bash
cargo build --release --locked -p rwkv-agent-server
export PATH="$PWD/target/release:$PATH"

rwkv-agent-data-plane --port 8121 --model-urls http://127.0.0.1:8417
rwkv-agent-server-rs \
  --port 8122 \
  --runtime-revision <runtime-commit-or-release> \
  --model-urls http://127.0.0.1:8417 \
  --data-plane-url http://127.0.0.1:8121

RWKV_AGENT_ENDPOINT=http://127.0.0.1:8122 rwkv-agent doctor
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8122 rwkv-agent
```

Command execution remains disabled unless a bounded Sandbox and Workspace are
explicitly configured. Debug Trace remains `off` unless explicitly enabled.

## Compatibility lifecycle script

```bash
rwkv-agent-service doctor
rwkv
```

`rwkv` can automatically start the configured local Sidecar and compatibility
Controller when they are offline, then enter interactive chat. Service lifecycle
commands are retained for existing installations and troubleshooting; they do
not replace the canonical Rust Server lifecycle above.

The compatibility endpoints are:

- Sidecar: `http://127.0.0.1:8118`;
- Controller: `http://127.0.0.1:8120`.

## Use

```bash
rwkv
rwkv ask "Explain recurrent state in RWKV."
rwkv tool web-search "RWKV latest official repository update"
rwkv research --branches 4 --rounds 2 \
  "Compare the latest official progress across RWKV repositories."
```

Interactive commands include `/status`, `/web`, `/knowledge`, `/research`,
`/longtext`, `/session` and `/json`.

## Optional local knowledge service

`knowledge_search` requires a separately running Elasticsearch-compatible
FineWiki index. The public service script does not download or start it.

See [Local knowledge service setup](KNOWLEDGE_SETUP.md) for the official
FineWiki, Elasticsearch, Embedding and Reranker download links, storage sizes,
copyable download commands and index-building steps.

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

Run the installation and canonical service pipeline on the GPU host, then
forward the Rust Server:

```bash
ssh -N -L 8122:127.0.0.1:8122 user@gpu-host
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8122 rwkv-agent doctor
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8122 rwkv-agent
```

Do not bind the Beta Server or compatibility Controller to a public interface
without adding your own authentication, TLS, rate limiting and outbound network
policy.

## Stop

```bash
rwkv-agent-service status
rwkv-agent-service stop
```

Only processes referenced by PID files under `RWKV_AGENT_STATE_DIR` are stopped.
