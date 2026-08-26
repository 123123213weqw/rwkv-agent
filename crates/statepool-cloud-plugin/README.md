# RWKV StatePool Cloud Plugin

Optional, out-of-process cloud extension for `rwkv-agent`. It currently
provides:

- plugin-v1 handshake and capability negotiation;
- dynamic, TTL-bounded Worker registration;
- exact `model_id`/`revision`/`tokenizer`/`state_abi` compatibility filtering;
- privacy- and state-affinity-aware placement;
- single-writer Leases with monotonically increasing fencing tokens;
- version compare-and-swap for immutable State metadata;
- checksum-verified, atomic LocalFS snapshot/restore for protocol and local
  integration tests;
- Worker drain admission control;
- bounded in-memory usage aggregation and Prometheus metrics.

The local and Cloud Lite profiles advertise `leases` and `state_lifecycle`.
The LocalFS/InMemory profile covers local conformance; the PostgreSQL/S3
profile provides transactionally fenced metadata and immutable object storage.
An exact-compatible RTX 4080 forced-Worker-process-loss lifecycle has passed
against the Cloud Lite profile. The service deliberately does not advertise
the stronger reserved `remote_state` capability until Controller restart
reconstruction and live multi-node rollout gates are implemented and verified.

The normal Agent server does not start or contact this process unless
`--cloud-plugin` is explicitly enabled.
