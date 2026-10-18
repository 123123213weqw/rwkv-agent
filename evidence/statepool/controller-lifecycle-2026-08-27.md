# Controller automatic State lifecycle evidence — 2026-08-27

## Scope

This evidence covers the Rust Controller orchestration boundary only. It is a
deterministic mock/CPU transport result, not a real-GPU migration or
Kubernetes scale-to-zero claim.

Verified sequence:

```text
plan → acquire(v0) → prefill/continue → renew → Sidecar snapshot
→ plugin CAS commit(v1) → Sidecar release → Lease release
→ plan(StateReference v1) → acquire(v1) → plugin read
→ Sidecar restore → continue → commit(v2) → release
```

The test also injects an HTTP failure after inference and before a confirmed
State commit. The first answer is returned with `residency=blocked_hot`; the
following request is rejected before prefill, restore, or continuation, proving
that the Controller does not silently re-execute an uncertain turn.

## Commands

Repository source, including uncommitted changes, was synchronized to
`WZU_Server:~/codex-build/rwkv-agent/` with `.git`, `target`, `.venv`,
`node_modules`, `.env*`, `var` and `data` excluded. Rust commands ran only on
the remote host with the system linker:

```text
CC=/usr/bin/cc \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/cc \
cargo check --workspace --all-targets --locked

CC=/usr/bin/cc \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/cc \
cargo test --workspace --locked
```

## Result

- workspace check: passed;
- `rwkv-agent-runtime` unit tests: 70 passed;
- `mock_full_path`: 15 passed;
- Controller lifecycle success test: passed;
- uncertain-commit no-double-execution test: passed;
- full Rust workspace: passed;
- PostgreSQL/S3 environment-gated tests: 2 ignored in this run and covered by
  the separate durable-adapter evidence.

Authoritative test names:

- `direct_chat_persists_releases_restores_and_advances_fenced_state`;
- `uncertain_lifecycle_commit_blocks_automatic_double_execution`;
- `disabled_plugin_never_builds_http_and_returns_original_local_path`.

## Invariants directly asserted

- first turn commits State version 1 and frees the Worker State;
- second turn restores without another prefill, commits version 2, and frees
  the restored Worker State;
- cached durable State consumes zero Hot Worker slots in Controller readiness;
- model identity, owner, checksum, version and fencing token are validated;
- committed State planning cannot fall back locally;
- uncertain commit blocks the next automatic model continuation;
- lifecycle disabled retains the original hot-State reuse and shutdown-release
  behavior.
