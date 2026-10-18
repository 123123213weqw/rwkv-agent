# Controller FinOps / UsageRecord evidence — 2026-08-27

## Scope

This slice connects successful direct-chat State lifecycle execution to the
StatePool `UsageRecord` contract without changing the plugin-disabled path.
It is conformance and accounting evidence, not a real GPU utilization claim.

Implemented and verified:

- `ExecutionPlan.worker_zone` preserves the selected Local/Edge/Cloud zone;
- the Controller keeps the actual Worker identity/zone across Hot reuse and
  durable restore;
- plugin-selected Local Worker endpoints are honored instead of silently using
  the default Sidecar;
- each successful direct-chat turn submits one validated UsageRecord when the
  handshake advertises `finops`;
- actual restore/snapshot wall time, per-turn State bytes, output token count,
  estimated input/prefill tokens, provider model wall time and plan cost/queue
  estimates are attributed to the record;
- reporting failure is visible as `trace.finops.status=report_failed` and does
  not fail an already successful inference;
- authoritative snapshot/restore byte counters are not incremented again from
  UsageRecord, preventing double counting;
- Prometheus exposes actual completed inference count by zone and the Grafana
  dashboard uses those counters rather than plan counts.

## Local non-compiling gates

Executed from the repository root:

```text
cargo fmt --all -- --check
# exit 0

git diff --check
# exit 0

uv run --with jsonschema python scripts/check_statepool_contracts.py
StatePool contracts valid: 7 schemas, 14 examples, 12 API paths

uv run --with pyyaml python scripts/check_statepool_deploy.py
StatePool deploy assets valid: 8 Compose services, 11 dashboard panels, 400 SBOM components

uv run python scripts/check_public_release.py
Public release audit passed: Python 0.3.0b2, Rust 0.3.0-beta.2, 620 public text files scanned.
```

`cargo fmt` is permitted locally by `AGENTS.md`; no local Rust compile, check or
test command was run.

## Remote Rust validation

Current source, including uncommitted changes, was synchronized with:

```text
rsync -az --delete \
  --exclude='.git/' --exclude='target/' --exclude='node_modules/' \
  --exclude='.venv/' --exclude='.env' --exclude='.env.*' \
  --exclude='var/' --exclude='data/' \
  ./ WZU_Server:~/codex-build/rwkv-agent/
```

Commands executed on `WZU_Server`:

```text
CC=/usr/bin/cc \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/cc \
cargo check --workspace --all-targets --locked

CC=/usr/bin/cc \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/cc \
cargo test --workspace --locked
```

Final result: both commands exited 0 with no warnings. Relevant test evidence:

```text
rwkv-agent-runtime unit tests: 71 passed
mock_full_path: 17 passed
  direct_chat_persists_releases_restores_and_advances_fenced_state
  plugin_selected_local_worker_endpoint_is_used_for_create_and_restore
  finops_reporting_failure_never_turns_successful_inference_into_failure
statepool-cloud-plugin: 11 passed
  usage_accounts_actual_zone_without_double_counting_lifecycle_bytes
statepool-plugin-api: 6 passed
  usage_validation_rejects_unknown_enums_and_non_finite_metrics
```

The two environment-gated PostgreSQL/S3 adapter tests remained ignored in this
workspace run because their explicit `RWKV_STATEPOOL_TEST_*` endpoints were not
set. They were not represented as passing by this evidence file.

## Metric semantics / non-claims

- `statepool_gpu_seconds_total` currently sums Sidecar-reported model wall time
  as a one-active-GPU proxy. It is not GPU SM utilization, energy, or billing.
- input and avoided-prefill token counts are character/4 estimates until
  provider tokenizer telemetry is wired.
- queue time and cost are ExecutionPlan estimates.
- no real Worker-kill GPU restore or live Kubernetes KEDA scale-to-zero test is
  claimed by this slice.
