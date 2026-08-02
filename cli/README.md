# RWKV Agent terminal client

`rwkv-agent` is the cross-platform terminal client for an RWKV Agent HTTP
backend. The client itself does not load model weights and can run on a laptop;
the current self-hosted backend requires a separately configured Linux CUDA
machine.

## Install only the client

From a source checkout with a Rust toolchain:

```bash
./cli/install.sh --client-only
```

The default prefix is `$HOME/.local`. Use another location when needed:

```bash
./cli/install.sh --client-only --prefix "$HOME/.rwkv-agent"
```

Prebuilt release archives contain one `rwkv-agent` binary. Extract an archive
for your platform, verify its adjacent SHA-256 file, then install it:

```bash
install -d "$HOME/.local/bin"
install -m 0755 rwkv-agent "$HOME/.local/bin/rwkv-agent"
```

The first release workflow builds these CLI targets:

- `aarch64-apple-darwin` for Apple Silicon macOS;
- `x86_64-unknown-linux-gnu` for 64-bit glibc Linux.

Windows, Intel macOS and Linux ARM64 are not part of the first verified binary
set. They may still build from source but are not release-tested yet.

## Connect to a backend

The default endpoint is `http://127.0.0.1:8120`:

```bash
rwkv-agent
```

To use a backend on another machine, keep the Controller bound to loopback and
forward it over SSH:

```bash
ssh -N -L 8120:127.0.0.1:8120 user@gpu-host
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8120 rwkv-agent doctor
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8120 rwkv-agent
```

You can also pass the endpoint explicitly:

```bash
rwkv-agent --endpoint http://127.0.0.1:8120 ask "你好"
```

The Beta Controller has no built-in public authentication or rate limiting. Do
not expose port 8120 directly to the Internet. Use loopback, an SSH tunnel, a
private network, or an authenticated reverse proxy.

## Commands

```text
rwkv-agent                         interactive chat
rwkv-agent doctor                  verify Controller, model and tools
rwkv-agent health                  print backend health
rwkv-agent ask <message>           send one ordinary Agent turn
rwkv-agent research <question>     bounded parallel-state Web research
rwkv-agent tool web-search <query> call Web search directly
rwkv-agent tool knowledge-search <query> call local knowledge directly
rwkv-agent --json ...              print complete machine-readable JSON
```

Interactive slash commands include `/status`, `/tools`, `/web`, `/knowledge`,
`/research`, `/longtext`, `/session`, `/json`, `/clear` and `/help`.

Color is enabled only on an interactive terminal. Set `NO_COLOR=1` to disable
it. Piped output and `--json` never contain ANSI color sequences.

## Full self-hosted installation

To install the client plus the local `rwkv` launcher and service manager:

```bash
./cli/install.sh
rwkv-agent-service init
```

Continue with the repository
[Quickstart](https://github.com/123123213weqw/rwkv-agent/blob/main/docs/QUICKSTART.md)
for model, runtime, SearXNG and local knowledge configuration.

## Source location

The Rust workspace lives at the repository root so the active Agent code is
visible without entering a directory named only for the client:

- `../crates/agent-cli` is the released terminal binary;
- `../crates/agent-core` contains the runtime-neutral strict protocol, typed tool
  registry, bounded same-State loop, cancellation/budget policy and lifecycle
  events.
- `../crates/agent-runtime` owns routing, recurrent chat State, durable transcripts,
  the Tool/Observation loop, parallel-State research and the safe command policy;
- `../crates/agent-server` exposes the CLI-compatible Rust HTTP control plane.

This `cli/` directory retains installer, release packaging, local lifecycle and
integration-smoke scripts for compatibility. See the repository
[`DEVELOPING.md`](../DEVELOPING.md) and [`code map`](../docs/CODEMAP.md).

Python continues to own the RWKV CUDA runtime, search, Evidence and long-text
data plane. The Rust server defaults to isolated port `8122`; the existing
Python Controller on `8120` remains compatible and unchanged until a separately
authorized deployment switch.

Run the split stack in an isolated environment:

```bash
rwkv-agent-data-plane --port 8121 --model-urls http://127.0.0.1:8417
cargo run -p rwkv-agent-server -- --port 8122 \
  --model-urls http://127.0.0.1:8417 \
  --data-plane-url http://127.0.0.1:8121
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8122 rwkv-agent doctor
```

## Build and test

```bash
cargo fmt --check
cargo test --locked
cargo clippy --all-targets -- -D warnings
cargo build --release --locked
bash cli/tests/cli_smoke.sh
```

Create a release archive for the current native Rust target:

```bash
./cli/package-release.sh
```
