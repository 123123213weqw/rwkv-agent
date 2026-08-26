# Real GPU forced Worker-loss restore — 2026-08-27

The Cloud Lite path has now passed one real-GPU process-loss experiment. An
RWKV-7 G1I 1.5B Albatross Worker on an RTX 4080 created and advanced a recurrent
State, exported 12,911,277 bytes, and committed them through the StatePool
PostgreSQL/S3 profile. The source PID was then terminated without releasing its
hot State. A fresh compatible Worker process started on the same physical GPU,
read the current fenced State from MinIO, validated the exact model identity
and SHA-256, restored a distinct GPU State ID, continued decoding from 42 to 66
seen tokens, released the restored State, and passed the real preStop drain
client with zero active requests and zero unpersisted user States.

Authoritative raw evidence and the claim boundary are in
[`bench/artifacts/statepool-4080-worker-kill-20260827/README.md`](../../bench/artifacts/statepool-4080-worker-kill-20260827/README.md).

This proves exact-compatible recovery across an actual killed Worker process.
It does not prove cross-model State migration, simultaneous cross-node GPU
migration, production S3 latency, or a live Kubernetes/KEDA 0→1→N→0 cycle.
