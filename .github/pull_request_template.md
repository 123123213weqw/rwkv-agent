## Problem and boundary

Describe the user-visible problem and whether this changes core, an optional
adapter, deployment configuration, benchmark or documentation.

## Compatibility

- [ ] Default local mode remains unchanged, or the breaking change is explicit.
- [ ] Protocol/schema changes include fixtures, ADR/migration and versioning.
- [ ] Optional dependency failure has tested fallback/fail-closed semantics.

## Validation

List exact commands, environment and outputs. Rust commands must follow the
repository build-host policy used by maintainers.

- [ ] Contract/deployment static checks pass.
- [ ] Relevant unit/integration regressions pass.
- [ ] Documentation and compatibility matrix are updated.

## Claims and supply chain

- [ ] Measured and estimated results are distinguished with raw evidence.
- [ ] No upstream source is copied when a protocol/adapter/configuration works.
- [ ] New dependency/image has version, license and `third_party` entry.
- [ ] No secrets, private prompts/States, weights or local paths are included.
