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

The local profile advertises `leases` and `state_lifecycle`. These capabilities
describe real single-process semantics and generic binary State persistence;
they do **not** claim durable PostgreSQL metadata, S3 persistence, a
multi-replica distributed Lease service, or live RWKV Sidecar export/import.
The service deliberately does not advertise `remote_state` until those gates
are implemented and verified.

The normal Agent server does not start or contact this process unless
`--cloud-plugin` is explicitly enabled.
