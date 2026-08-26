# RWKV StatePool Cloud Plugin

Optional, out-of-process cloud extension for `rwkv-agent`. It currently
provides:

- plugin-v1 handshake and capability negotiation;
- dynamic, TTL-bounded Worker registration;
- exact `model_id`/`revision`/`tokenizer`/`state_abi` compatibility filtering;
- privacy- and state-affinity-aware placement;
- Worker drain admission control;
- bounded in-memory usage aggregation and Prometheus metrics.

It does **not** yet claim durable PostgreSQL metadata, S3 State persistence,
distributed leases or live RWKV snapshot/restore. The service deliberately
does not advertise the `remote_state` or `leases` capabilities until those
gates are implemented and verified.

The normal Agent server does not start or contact this process unless
`--cloud-plugin` is explicitly enabled.
