# CLAUDE.md

Read `AGENTS.md` and `docs/TODO.md` before acting. The TODO is the sole
project-status source.

## Active architecture

```text
Rust CLI -> Rust Server -> Rust Runtime/Core
                         -> Python RWKV Sidecar
                         -> Python Retrieval/Evidence Data Plane
```

The Python Controller, old `rwkv-search` application, packaged Web preview and
automatic service wrapper have been removed. Do not recreate a parallel Agent
loop. Active ownership is listed in [docs/CODEMAP.md](docs/CODEMAP.md).

## Development

- Python providers: `src/rwkv_agent/sidecar.py` and
  `src/rwkv_agent/data_server.py`.
- Rust control plane: `crates/agent-{core,runtime,server,cli}/`.
- Runtime-only data belongs in ignored `var/`, `data/`, `bench/runs/` or
  `target/`.
- Frozen benchmark evidence must not be rewritten.
- Repository contributors must obey the remote-only Rust compilation policy in
  `AGENTS.md`.

The canonical startup and verification sequence is
[docs/QUICKSTART.md](docs/QUICKSTART.md).
