# Project governance

RWKV State Agent is currently a maintainer-led project. Governance is kept
small and explicit rather than implying an organization that does not exist.

## Roles

- **Maintainer:** merge/release authority; owns security response, compatibility
  policy and claim review.
- **Reviewer:** trusted contributor who can approve an area but cannot publish a
  release without a Maintainer.
- **Contributor:** anyone submitting Issues, documentation, tests, adapters or
  code under the project license.

Roles are earned through sustained, reviewed contributions. Additions/removals
are recorded in a public pull request updating this file or `CODEOWNERS` when
that file is introduced.

## Decision process

- normal changes: one Maintainer review and green required checks;
- public protocol or compatibility changes: ADR, contract fixture and migration
  note, with a minimum seven-day public review window before merge;
- security fixes: embargoed review is allowed, followed by a public advisory;
- release claims: must link raw evidence and distinguish measured from estimated;
- unresolved architectural disagreement: document options in an ADR; the
  Maintainer decides for the current release and records the rationale.

## Maintained surface

The project maintains:

- local Agent Controller/runtime behavior;
- Stateful Inference Session and StatePool plugin contracts;
- exact State compatibility, Lease/fencing/CAS and lifecycle logic;
- explainable placement and State-aware FinOps;
- thin optional adapters and deployment examples.

It does not maintain Forks of Kubernetes, KEDA, HAMi, AIBrix, KServe,
PostgreSQL, S3/MinIO, Prometheus or Grafana. Upstream-specific code must stay in
an optional adapter with a pinned compatibility row and can be retired without
breaking local mode.

## Compatibility and deprecation

- default local behavior is the compatibility anchor;
- versioned contracts are additive within a v1 line;
- removing a public field/endpoint requires a new contract major version;
- an adapter supports only versions listed in `COMPATIBILITY.md`;
- deprecated versions receive at least one documented release window when
  practical; pre-release security/correctness issues may shorten that window.

## Community expectations

Use the Code of Conduct norms of respectful, technical collaboration. Reports
must not include credentials, private prompts/States, model weights or private
server details. Maintainers do not promise fixed response times; release notes
must describe actual maintenance capacity honestly.
