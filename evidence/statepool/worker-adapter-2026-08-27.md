# StatePool Worker adapter evidence — 2026-08-27

This slice was validated as a control-plane lifecycle on `WZU_Server`. It is
not a real-GPU or Kubernetes scale-to-zero claim.

Verified behavior:

- Sidecar Worker mode remains absent when `RWKV_STATEPOOL_URL` is empty;
- exact model/tokenizer/State ABI and actual runtime capacity/load are reported;
- registration is recovered if the plugin forgets a Worker;
- a drain decision is sticky against a delayed `ready` heartbeat;
- drain closes new inference admission but keeps snapshot/release available;
- resident recurrent States are conservatively counted as unpersisted;
- a dirty Worker reports `draining`, an expired dirty Worker reports
  `deadline_exceeded`, and only zero work plus zero dirty States reports
  `safe_to_stop`;
- the preStop client polls until the Controller makes States durable and
  released, otherwise exits non-zero.

Reproduce the cross-process smoke after starting the plugin:

```bash
PYTHONPATH=src python3 scripts/statepool_worker_smoke.py \
  --plugin-url http://127.0.0.1:8130
```

Raw output: `worker-registration-drain-2026-08-27.txt`.

Validation totals for the same source slice:

- Python Worker/State/lifecycle tests: 19 passed;
- Rust StatePool plugin tests on `WZU_Server`: 10 passed, 2 external-service
  integrations intentionally ignored in this run;
- Helm 3.19.0 lint and Worker+KEDA rendering: passed.

The Helm validator ran in `alpine/helm:3.19.0`, observed image digest
`sha256:aef9b56f64e866207d9591d0abd8f6d767b36aadd12edf68f8a719716d9d29c9`.
