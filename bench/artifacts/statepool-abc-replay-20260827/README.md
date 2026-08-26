# StatePool A/B/C FinOps replay — 2026-08-27

## Classification

**Simulation replay from measured primitives.** This directory is not a live
100-Session GPU benchmark and must not be presented as one.

The replay holds one explicit workload constant across three policies:

- 100 Sessions;
- 8 active Sessions per Worker, hence 13 Workers at peak;
- two 60-second active windows separated by a 300-second idle gap;
- 512 historical tokens per Session at the second window;
- a scenario input price of CNY 2.00 per allocated GPU-hour.

## Result

| Scenario | Policy | Modeled GPU-hours / 100 Sessions | Modeled idle GPU-minutes | Repeated Prefill tokens | State transfer | Estimated GPU cost |
|---|---|---:|---:|---:|---:|---:|
| A | Sticky Worker | 1.516667 | 65.0000 | 0 | 0 | CNY 3.033333 |
| B | Stateless re-prefill | 0.831877 | 17.6813 | 51,200 | 0 | CNY 1.663754 |
| C | StatePool | 0.831877 | 17.6813 | 0 | 3,873,383,100 bytes | CNY 1.663754 |

Within these assumptions, C reduces modeled allocated GPU-hours and estimated
GPU cost by **45.151%** versus A, while avoiding **51,200** repeated Prefill
tokens versus B. C writes 2,582,255,400 State bytes and reads 1,291,127,700
bytes in exchange.

B and C intentionally have the same modeled GPU allocation. The replay does
not know the GPU time required to re-prefill B, so it does not invent an extra
cost advantage for C. Likewise, storage/request/egress charges are excluded.

## Measured inputs

The script reads and hashes three independent evidence classes:

1. **KEDA control plane:** kind/Kubernetes 1.34 + KEDA 2.20.1 measured 14.380
   seconds from demand to three Ready simulated Workers and 40.803 seconds from
   pending-clear to the last Pod disappearing.
2. **GPU State primitive:** one RTX 4080 lifecycle measured a 12,911,277-byte
   exact RWKV State, 81.266 ms Worker-local snapshot and 104.246 ms
   Worker-local restore.
3. **100-Session contract correctness:** the frozen `keep_hot`,
   `drop_reprefill`, and `move_cpu` contract runs each completed 300 events with
   zero cross-talk, zero failed events and zero State leaks.

The `move_cpu` contract run is a correctness analogue for snapshot/restore; it
is not relabelled as the PostgreSQL/S3 implementation. The separate GPU and
KEDA inputs are not relabelled as one end-to-end topology.

## Formulae

```text
peak_workers = ceil(100 / 8) = 13

A allocated seconds = 13 × (60 + 300 + 60)

B/C allocated seconds =
  13 × 2 × (14.380 scale-to-ready + 60 active + 40.803 shutdown)

re-prefill tokens = 100 × 512 = 51,200

C State bytes =
  (100 × 2 snapshots + 100 × 1 restore) × 12,911,277
```

`modeled_idle_gpu_minutes` counts A's idle gap. For B/C it counts the measured
shutdown/grace interval while Pods still exist; startup time is treated as
initialization rather than idle.

## Explicit non-measurements

The replay does not report:

- GPU average utilization for A/B/C;
- production restore P50/P95;
- B's Prefill throughput or GPU seconds;
- storage, API request or network charges;
- real GPU Kubernetes scaling.

The CNY 2.00 price is a scenario parameter copied from the RTX 4080 Worker
registration, not a current market-price claim.

## Reproduce

The manifest records clean Git commit
`a7bf0100a1c9793977ea8a48cf72e36118ed64e3` and hashes the generator plus all
source evidence.

```bash
python scripts/statepool_finops_replay.py \
  --output-dir bench/artifacts/statepool-abc-replay-20260827 \
  --sessions 100 \
  --sessions-per-worker 8 \
  --active-seconds 60 \
  --idle-seconds 300 \
  --windows 2 \
  --history-tokens 512 \
  --gpu-price-cny-per-hour 2
```

- `summary.json`: inputs, derivation, results and claim limits;
- `results.csv`: compact A/B/C table;
- `manifest.json`: commit, generator hash and source hashes;
- `SHA256SUMS`: artifact integrity.
