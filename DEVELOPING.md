# Developing RWKV Agent

All development commands run from the repository root. The Rust control plane
and Python data plane deliberately share one repository but have separate
runtime responsibilities; see [`docs/CODEMAP.md`](docs/CODEMAP.md) before
changing an entry point.

## Active implementation

- Rust owns the Agent protocol, tool registry and loop, recurrent-State
  lifecycle, sessions, research orchestration, command sandbox policy and HTTP
  control plane.
- Python owns CUDA/model integration, live and local retrieval, long-text
  processing, Evidence reduction and claim validation.
- `cli/` contains the Rust client installer, release packager and smoke
  fixtures, not a second service lifecycle.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[realtime,segment,dev]' fastapi uvicorn
```

Install the terminal client:

```bash
./cli/install.sh
```

## Common commands

```bash
./scripts/dev check       # static checks and public-release audit
./scripts/dev test        # Python and Rust tests
./scripts/dev build       # Python package and Rust release workspace
./scripts/dev data-plane  # isolated Python data plane on 127.0.0.1:8121
./scripts/dev server      # isolated Rust control plane on 127.0.0.1:8122
./scripts/dev cli         # client connected to the isolated Rust server
```

The data plane expects a model Sidecar at `http://127.0.0.1:8417` unless
`RWKV_AGENT_MODEL_URLS` is set. The Rust development server is the only Agent
Controller.

## Direct commands

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:. python -m pytest -q
ruff check src bench benchmarks tests scripts
python -m compileall -q src
python scripts/check_public_release.py

cargo fmt --check
cargo test --locked
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release --locked
bash cli/tests/cli_smoke.sh
bash cli/tests/install_smoke.sh
```

## Change boundaries

Do not combine a source-layout refactor with changes to model prompts, Tool Call
semantics, retrieval ranking, model weights or production deployment. Raw model
traces, fetched pages, credentials, server addresses, runtime state and
license-restricted benchmark cases must remain outside Git. Reviewed aggregate
benchmark summaries belong under `bench/baselines/`; reproducible runners belong
under `benchmarks/`; raw results belong under ignored run directories.
