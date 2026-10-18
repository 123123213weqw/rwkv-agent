# Release checklist

## Correctness and compatibility

- [ ] plugin-disabled regression demonstrates original local behavior;
- [ ] all JSON Schemas, OpenAPI and checked-in examples validate;
- [ ] Rust format/check/test/clippy run under the repository build policy;
- [ ] Python/public release checks pass;
- [ ] Compose config and both gated/default Helm renders validate;
- [ ] protocol changes have ADR, compatibility and migration notes;
- [ ] stale Lease, owner mismatch, version conflict and checksum corruption tests pass.

## Claims and evidence

- [ ] every measured number links hardware, command, raw log and commit;
- [ ] estimates are labeled and include formula/source/date/currency;
- [ ] no configuration-only integration is described as runtime verified;
- [ ] GPU concurrency wording distinguishes resident States from physical batch;
- [ ] live cross-Worker restore is claimed only after kill/restore/continue evidence;
- [ ] benchmark artifacts contain no prompts, credentials or private paths.

## Supply chain

- [ ] `third_party/COMPONENTS.yaml`, NOTICE and compatibility versions updated;
- [ ] external images use exact tags and release artifacts use registry digests;
- [ ] dependency license audit and SBOM generated;
- [ ] container runs non-root with read-only root filesystem where applicable;
- [ ] release archives/model references use checksums and exclude weights.

## Operations and documentation

- [ ] README, Quickstart, configuration, known limitations and roadmap updated;
- [ ] liveness/readiness and failure/fallback semantics documented;
- [ ] backup/restore/GC/drain runbooks match the released adapter;
- [ ] Grafana dashboard imports and all PromQL returns the intended type;
- [ ] security review covers authentication, owner identity and State privacy;
- [ ] rollback command and compatibility window are documented.

## Publication

- [ ] clean tag built from a clean tree;
- [ ] image digests, SBOM and checksums attached;
- [ ] changelog and claim matrix frozen;
- [ ] release notes name every unimplemented/unverified gate;
- [ ] competition evidence register references only immutable published artifacts.
