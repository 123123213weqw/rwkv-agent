# Real GPU StatePool forced-Worker-loss evidence — 2026-08-27

## Verdict

`worker/result.json` records a passing, exact-model lifecycle on a real NVIDIA
RTX 4080:

```text
source Worker process
  -> prefill + continue
  -> 12,911,277-byte recurrent-State snapshot
  -> PostgreSQL Lease/fencing + S3 atomic commit
  -> SIGTERM source process (source PID proven dead)
  -> start a fresh compatible Worker process
  -> PostgreSQL current-State lookup + S3 read
  -> checksum/model/owner validation
  -> restore into a new GPU State ID + continue
  -> release restored user State
  -> StatePool preStop drain reports safe_to_stop
  -> stop replacement process (target PID proven dead)
```

The source and replacement used one physical RTX 4080 sequentially and reused
the same HTTP endpoint. They are different OS processes and different GPU
State identities. This proves process-loss recovery on an exact-compatible
Worker; it is not a simultaneous cross-node transfer benchmark.

## Immutable identity and storage proof

| Item | Observed value |
|---|---|
| Model | `rwkv7-g1i-20260805-1.5b` |
| Revision | `g1i-1.5b-20260805-ctx16384` |
| Tokenizer | `rwkv_vocab_v20230424` |
| State ABI | `rwkv7-albatross-fp32io16-state-v1` |
| Source State | `state-1ca41f5098a541cab011786ba5904738` |
| Restored State | `state-bd1fdfaf478148f7a366ba2304513377` |
| Durable version / fencing token | `1 / 1` |
| Snapshot bytes | `12,911,277` |
| SHA-256 | `c5efb72b2487a6065edd7da807da0382088e0bf6df79d43cb221228ec0f97711` |
| Durable object | `s3://rwkv-statepool/recurrent-state/e4de9d44ad8673ef822d7e3d4d509ecc/v1-f1-c5efb72b2487a606.state` |

`statepool-server/postgres-session.txt` shows the same current State reference
and no active Lease. `statepool-server/minio-object-stat.txt` independently
shows the 12 MiB immutable object. Plugin Prometheus counters advance from zero
to one committed snapshot, one completed restore, and exactly 12,911,277 bytes
in each direction.

## Measured lifecycle

| Measurement | Result |
|---|---:|
| Worker-local snapshot | 81.266 ms |
| Fresh replacement process ready | 4,011.946 ms |
| Worker-local restore | 104.246 ms |
| Seen tokens before / after restore | 42 / 66 |
| Source process after forced stop | dead |
| Target process before drain | alive |
| Target process after final stop | dead |
| preStop drain exit code | 0 |
| preStop drain result | `safe_to_stop`, 0 active, 0 unpersisted |
| GPU memory with source / after final stop | 3,282 MiB / 198 MiB |

The replacement process benefited from already compiled CUDA extensions and a
warm filesystem page cache, so its 4.012 s startup is a warm-process restart,
not a cold-node startup number.

The 49.041 s S3 commit and 60.636 s S3 read travelled from the 4080 host over
a reverse SSH tunnel, through the capture workstation, and then through a
second SSH tunnel to the WZU Server Compose network. They prove end-to-end byte
correctness but **must not** be presented as MinIO or production-network
latency. The 120.723 s total is consequently not a serving SLO baseline.

The base model's eight-token text was not evaluated as a semantic recall
benchmark. The correctness assertions here are process death, exact model
identity, digest/size equality, fenced durable version, distinct restored GPU
State identity, monotonic `seen_tokens`, continued decoding, release, and safe
drain.

## Safe drain with a reconstructible system root

Both source and replacement health snapshots report one allocated immutable
tool-gate root with `persistent_states.reconstructible=1`. After the restored
user State is released, the replacement still owns that root, but its Worker
capability reports `unpersisted_state_slots=0`. The real preStop client then
returns `safe_to_stop` and the control plane agrees. This distinguishes a
rebuildable system cache from dirty user/session State without weakening the
rule that every other resident State blocks scale-down.

## Topology

- GPU Worker: `WZU_4080`, Ubuntu 24.04-family kernel, RTX 4080 16 GiB,
  driver 595.84, Python 3.12.2, PyTorch 2.11.0+cu130.
- Model runtime: Albatross `rwkv7_fast_v3a`, `fp32io16` recurrent-State ABI.
- StatePool: `WZU_Server` Docker Compose Cloud Lite profile.
- Metadata: PostgreSQL 17.6.
- Object store: MinIO release `2025-04-22T22-12-26Z`.
- StatePool binary SHA-256:
  `18893ac60984dbe1f4e5c3efa3e2c32f90de2e2de680bb5b344fa610ee27990b`.

Model, runtime, source and launcher hashes are retained under `worker/`.

## Driver command

The essential command was:

```bash
PYTHONPATH="$ROOT:$ROOT/src" python scripts/statepool_live_lifecycle_demo.py \
  --plugin-url http://127.0.0.1:18131 \
  --source-worker-url http://127.0.0.1:18218 \
  --target-worker-url http://127.0.0.1:18218 \
  --source-worker-id worker-4080-source \
  --target-worker-id worker-4080-target \
  --session-id gpu-4080-20260826T180745Z \
  --owner-id demo:statepool-gpu-4080 \
  --max-tokens 8 --target-tier cold \
  --source-stop-command "/bin/kill 3720167" \
  --target-start-command "$RUN/start-worker.sh target" \
  --source-down-timeout-seconds 30 \
  --target-ready-timeout-seconds 300 \
  --output "$RUN/result.json"
```

The source and replacement launch environment is retained verbatim in
`worker/start-worker.sh`. CUDA 13.0 headers/tools were assembled from the
installed NVIDIA conda packages because that host's CUDA toolkit is split
across packages; `worker/software-versions.txt` records the resulting compiler
and runtime versions.

## Artifact index

- `worker/result.json`: authoritative lifecycle result.
- `worker/source.log`, `worker/driver-and-target.log`: both process logs.
- `worker/health-*.json`: exact model, State allocator and Worker lifecycle.
- `worker/plugin-workers-*.json`: source offline, target ready/draining and
  dirty-State capacity reported to the plugin.
- `worker/plugin-metrics-*.prom`: before/after counter evidence.
- `worker/prestop-drain.json`, `worker/drain-status.json`: safe scale-down gate.
- `worker/gpu-*.csv`, `worker/nvidia-smi-before.txt`: raw GPU telemetry.
- `statepool-server/postgres-session.txt`: durable metadata and fencing state.
- `statepool-server/minio-object-stat.txt`: durable S3 object existence/size.
- `statepool-server/plugin-image-inspect.json` and
  `plugin-binary-sha256.txt`: executed image identity.
- `statepool-server/cargo-{check,test}.log`: full remote Rust validation.

`SHA256SUMS` covers every retained artifact other than itself.
