# RWKV Agent CLI

`rwkv-agent` is the terminal client for the Agent HTTP API. It contains no model
runtime, crawler, index or sandbox implementation; those live in the server and
data-plane packages listed in [`docs/CODEMAP.md`](../../docs/CODEMAP.md).

Common commands:

```bash
cargo run -p rwkv-agent-cli -- --endpoint http://127.0.0.1:8122
cargo test -p rwkv-agent-cli
bash cli/tests/cli_smoke.sh
```

Installation and release packaging remain under [`cli/`](../../cli/README.md)
for compatibility with existing users.
