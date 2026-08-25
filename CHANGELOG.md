# Changelog

All notable public changes are recorded here.

## Unreleased

## 0.3.0-beta.2 - 2026-08-25

### Added

- canonical Service v1 TaskSpec, request identity, Owner-scoped task APIs and
  structured error contracts for the Rust control plane;
- durable Task Ledger v2 semantics for idempotent requests, resume, cancel,
  process-restart recovery and streaming disconnect cleanup;
- independent liveness/readiness diagnostics with explicit Model, Data Plane,
  Sandbox, State-capacity and Task Ledger status;
- opt-in `off | redacted | full` Rust Debug Trace with bounded asynchronous
  writing, rotation, recovery, checksums and Owner-scoped local APIs.

### Changed

- renamed the user-facing product to **RWKV Agent** while retaining RWKV Search
  as the internal retrieval subsystem and keeping repository, package and API
  compatibility identifiers unchanged;
- made `rwkv` the primary documented user command; lifecycle commands remain
  available for administrators;
- increased the launcher's health-check tolerance for SSH-forwarded remote
  Agents so normal tunnel latency does not trigger a false local autostart;
- made `rwkv-agent-server-rs` authoritative for the new service lifecycle while
  retaining the current CUDA Sidecar and Data Plane as narrow external Python
  providers and the port-8120 Controller as a compatibility path.

### Known limitations

- Debug Trace and its HTTP API are disabled by default; `full` mode contains
  private prompt, model and tool bodies and is intended only for local diagnosis;
- model inference, retrieval and evidence providers still require the external
  Python runtime until separate Rust parity gates pass;
- this release validates service correctness, cancellation and State cleanup,
  not GPU-to-CPU snapshot/restore, production authentication, high-availability
  or a new high-concurrency performance claim.

## 0.3.0-beta.1 - 2026-07-29

### Added

- Preview4922 13.3B Agent release profile with 12,288-token context;
- semantic Search Gate, strict three-tool protocol and reasoning-boundary cleanup;
- state-native bounded parallel Web research;
- generic local Sidecar/Controller lifecycle script;
- Rust CLI `doctor` command and service environment preflight;
- public production config and environment templates;
- 200-case unified Agent regression summaries;
- user Quickstart, model, deployment, configuration, troubleshooting and known
  issue documentation.

### Changed

- Python package version unified with Rust CLI Beta release;
- Rust CLI and Agent backend are now the primary documented interface;
- private host-specific lifecycle assumptions removed from public scripts;
- Legacy Web UI retained but no longer presented as the current Agent path.

### Known limitations

- malformed Tool Call JSON remains possible;
- open-Web and exact-page recall remain below stable-release targets;
- HTTP API is loopback-only and has no built-in authentication or rate limiting;
- only one specific 13.3B single-V100 runtime is release-verified.
