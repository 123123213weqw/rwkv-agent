# Changelog

All notable public changes are recorded here.

## Unreleased

### Changed

- renamed the user-facing product to **RWKV Agent** while retaining RWKV Search
  as the internal retrieval subsystem and keeping repository, package and API
  compatibility identifiers unchanged;
- made `rwkv` the primary documented user command; lifecycle commands remain
  available for administrators;
- increased the launcher's health-check tolerance for SSH-forwarded remote
  Agents so normal tunnel latency does not trigger a false local autostart.

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
