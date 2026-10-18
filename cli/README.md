# RWKV Agent terminal client

`rwkv-agent` is the cross-platform Rust client for the canonical Rust Agent
Server. It does not load model weights and can run on a laptop.

## Install

Prebuilt release archives contain the binary, `README.md` and `LICENSE`.
Verify the adjacent SHA-256 file, then install the binary:

```bash
install -d "$HOME/.local/bin"
install -m 0755 rwkv-agent "$HOME/.local/bin/rwkv-agent"
```

From a source checkout with an existing release build:

```bash
./cli/install.sh --skip-build
```

The installer installs only `rwkv-agent`; it does not recreate the removed
`rwkv` launcher or `rwkv-agent-service` Python Controller wrapper.

## Connect

The default endpoint is `http://127.0.0.1:8122`:

```bash
rwkv-agent doctor
rwkv-agent ask "你好"
```

For a remote GPU host, keep the Rust Server on loopback:

```bash
ssh -N -L 8122:127.0.0.1:8122 user@gpu-host
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8122 rwkv-agent doctor
```

Do not expose the Beta Server directly to the Internet; it has no built-in
authentication, TLS or rate limiting.

## Commands

```text
rwkv-agent                         interactive chat
rwkv-agent doctor                  verify Server, model and tools
rwkv-agent health                  print backend health
rwkv-agent ask <message>           send one ordinary Agent turn
rwkv-agent research <question>     bounded parallel-state Web research
rwkv-agent task --spec <file>      submit a canonical TaskSpec
rwkv-agent tool web-search <query> call Web search directly
rwkv-agent --json ...              print machine-readable JSON
```

See [the quickstart](../docs/QUICKSTART.md) for the Sidecar, Data Plane and
Rust Server startup order. Source ownership is documented in
[CODEMAP.md](../docs/CODEMAP.md).

## Build and release

```bash
cargo fmt --check
cargo test --locked
cargo clippy --all-targets -- -D warnings
cargo build --release --locked
bash cli/tests/cli_smoke.sh
./cli/package-release.sh
```

Repository contributors must obey `AGENTS.md`; its remote-only Rust build
policy overrides the local commands above.
