# StatePool OpenAI-compatible Worker audit — 2026-08-27

## Scope

This bundle validates commit `f914222f3d7b7b6068534af238c435fb77411e77`
(`feat: add capability-aware OpenAI worker adapter`). It covers the additive
Worker contract, three State-capability modes, the standard-library HTTP
adapter, fake-upstream proxy lifecycle, default/overlay Helm rendering and the
standalone OCI image.

It does **not** claim a live vLLM model/GPU result, Transformer KV migration,
model-quality certification or production throughput. The HTTP integration
test uses a protocol-faithful fake upstream. The Qwen3.5 profile remains a
candidate until a separately archived live-runtime run passes.

## Verdict

- Python: **787 passed, 8 skipped, 52 subtests passed**.
- Rust on `WZU_Server`: workspace tests, check and Clippy `-D warnings` passed.
  The two durable-adapter tests remained ignored because this audit did not
  inject the PostgreSQL/S3 test secrets; those adapters have separate evidence.
- Contracts: 7 schemas and 14 examples validated; OpenAPI retains 12 paths.
- Deployment: 9 Compose services, 12 Grafana panels and 400 SBOM components
  passed static checks; Compose resolved successfully.
- Helm: default chart and `values-openai-worker.yaml` rendered remotely. The
  adapter Pod has `state-mode=affinity_only`, CPU-only proxy resources and the
  Python drain command; it does not request a GPU for the external engine.
- OCI: `openai-worker-adapter` built remotely as
  `sha256:fafc9a17812723bb06c1b9389ad68863c2c7f76e9c6945427e4769acf87083fb`.
  Lightweight module import and drain-client availability passed.
- Local Rust execution: none. Only `cargo fmt --check` ran locally; every Rust
  compile/check/test ran on `WZU_Server` under the repository policy.

## Capability invariants proved

| Worker mode | Affinity hint | Transcript replay | Native restore Lease |
|---|---:|---:|---:|
| `replay_only` | no | yes | never |
| `affinity_only` | yes | yes | never |
| `native_export` | yes | only on fallback | exact-compatible only |

Legacy v1 RWKV Worker JSON without `state_capability` still deserializes as
`native_export`. The new OpenAI adapter emits `affinity_only` explicitly and
returns `X-StatePool-Worker-Id`; it cannot promote itself to native export.

## Primary commands

Local non-Rust validation:

```bash
PYTHONPATH=.:src uv run --offline pytest -q
uv run --offline ruff check <changed maintained Python files>
uv run --offline --with jsonschema python scripts/check_statepool_contracts.py
uv run --offline --with pyyaml python scripts/check_statepool_deploy.py
uv run --offline python scripts/check_public_release.py
docker compose -f deploy/statepool/compose.yaml config --quiet
cargo fmt --all -- --check
```

Remote Rust and deployment validation:

```bash
cargo test --workspace --all-targets
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
helm template demo deploy/statepool/helm/statepool
helm template demo deploy/statepool/helm/statepool \
  -f deploy/statepool/helm/statepool/values-openai-worker.yaml
docker build --target openai-worker-adapter \
  -t rwkv-openai-worker-adapter:audit \
  -f deploy/statepool/Dockerfile .
```

See the adjacent raw logs and rendered manifests. `SHA256SUMS` authenticates
all files in this bundle except itself.
