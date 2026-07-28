# RWKV7 Albatross production chunk scheduler

Scheme-B implementation: retain the existing Albatross
`faster3a_2607` runtime and add a production-oriented recurrent-state scheduler
around its existing vectorized continuation interface:

```python
logits = model.forward(tokens_BxT, existing_state_B)
```

The scheduler now backs the desktop `rwkv_agent.sidecar` through
`ContinuousBatchEngine`. The integrated path has been tested in an isolated
G1I process; ports 8118, 8119 and 8120 have not been switched or restarted.

## Why no Albatross kernel patch is included

Unlike the older HF dispatch described by PR #58, current Albatross already
accepts an existing recurrent state together with an entire `[B,T]` token
chunk. Its `forward()` selects a BnTn path and mutates the supplied state.

The missing production work was above the kernel:

1. split long inputs into fixed exact chunks;
2. batch requests sharing the same chunk length;
3. keep independent state rows in a bounded slab;
4. gather/scatter non-contiguous rows safely;
5. batch T=1 decode while preserving per-request ownership.

## Code

- `src/rwkv7_scheduler/state_pool.py`
  - fixed-capacity recurrent-state slab;
  - generation-protected opaque handles;
  - stale-handle rejection;
  - reusable contiguous workspaces;
  - zero-on-reuse;
  - shared model execution lock.
- `src/rwkv7_scheduler/scheduler.py`
  - fair fixed-quantum chunk waves;
  - exact-length tails without padding;
  - vectorized continuation into an existing state;
  - continuous active-row greedy decode;
  - cancellation, queue and context limits;
  - shape/resource metrics.
- `tests/test_scheduler.py`
  - six CPU tests using an Albatross-compatible fake recurrent model.
- `benchmarks/scheduler/run_g1i_production_matrix.py`
  - real G1I 7.2B correctness, quantum autotune and throughput matrix.

## Important state-layout rule

Albatross single-GPU state is:

```text
shift   [L, 2, B, C]
wkv     [L, B, H, N, N]
elapsed [B]
```

A narrow slice through the middle batch dimension is generally
**non-contiguous**, even if its slots are consecutive. Passing that view to
custom CUDA kernels as a flat state caused real greedy mismatches during the
first benchmark.

The final pool borrows a slab view only when every resulting tensor reports
`is_contiguous()`. All other batches use a contiguous reusable workspace and
`index_select/index_copy_`.

## Scheduling policy

For each scheduling round:

1. every request with at least `prefill_chunk_size` tokens receives at most one
   full chunk;
2. newly created short tails are grouped by exact tail length;
3. no padding token is ever advanced through RWKV state;
4. the next round continues remaining long requests.

This bounds starvation while retaining large BnTn chunks.

```python
pool = AlbatrossStatePool(
    model,
    capacity=32,
    max_batch_size=8,
)
pool.prewarm([1, 2, 4, 8])

scheduler = AlbatrossChunkScheduler(
    model,
    pool=pool,
    config=SchedulerConfig(
        prefill_chunk_size=64,
        max_batch_size=8,
        max_queue_size=32,
        max_input_tokens=12288,
    ),
    token_device="cpu",  # current G1I uses CPU embeddings
)

scheduler.admit_many(
    [
        ("request-a", token_ids_a),
        ("request-b", token_ids_b),
    ]
)
scheduler.prefill(["request-a", "request-b"])
outputs = scheduler.greedy_decode(
    ["request-a", "request-b"],
    max_new_tokens=32,
)

# Tool result or another continuation chunk reuses the same state.
scheduler.continue_tokens("request-a", observation_token_ids)
```

## V100 G1I result

Environment:

- G1I preview3260 7.2B, context 12,288;
- Tesla V100 PCIe 32GB;
- Albatross `faster3a_2607`;
- `fp32io16`, CPU embedding, `batched_rkv=off`;
- existing 8118 Sidecar remained loaded on the same GPU;
- two complete timing repetitions per point;
- eight generated tokens after prefill;
- natural Chinese/English prompt tokens;
- serial baseline uses the same chunk quantum and one recurrent state at a
  time.

### Quantum autotune

Variable prompt lengths:

```text
384, 512, 640, 768, 896, 1024, 1152, 1280
```

| Quantum | Speedup vs serial | Exact greedy |
|---:|---:|---:|
| 64 | **2.633x** | 2/2 |
| 128 | 2.161x | 2/2 |
| 256 | 1.717x | 2/2 |
| 512 | 1.471x | 2/2 |

The frozen V100 choice is `64` for this model/runtime. It is a benchmarked
default, not a universal constant for another GPU or model width.

### Production matrix with quantum 64

| Case | Input lengths | Speedup | Scheduled input tok/s | Exact greedy |
|---|---|---:|---:|---:|
| Equal short B8 | 8 × 512 | **3.091x** | 3,218 | 2/2 |
| Equal medium B8 | 8 × 2,048 | **3.122x** | 4,033 | 2/2 |
| Variable short B8 | 41–179 | **1.918x** | 879 | 2/2 |
| Variable medium B8 | 384–1,280 | **2.594x** | 2,995 | 2/2 |
| Variable long B8 | 2,048–6,144 | **2.704x** | 3,339 | 2/2 |

All autotune and matrix points were exact greedy across both repetitions.

### Memory

- G1I model allocation: 14,299,201,536 bytes.
- One recurrent state: 34,078,724 bytes.
- Eight-slot slab: 272,629,792 bytes.
- Cached B1–B8 workspaces observed in the matrix: 1,226,834,064 bytes.
- Process peak CUDA allocation: 16,657,019,392 bytes.

Workspace caching intentionally trades roughly 1.23GB for stable serving
latency and no per-call CUDA state allocations. A smaller deployment can
prewarm/cache fewer batch sizes.

## Verification

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
ruff check src tests bench
```

Current result:

- scheduler/state unit tests: 7/7;
- Ruff: passed;
- PyCompile: passed;
- G1I production matrix: all exact;
- result SHA-256:
  `b95baf73accebf7cfd7eaaa9cb5d206fa841d40624d6be07cd34da3d0a053e1c`.

Raw result: `benchmarks/scheduler/g1i_production_matrix_final.json`.

## Remaining integration boundaries

- real-G1I validation of the new opt-in HTTP `state_id` endpoints;
- adaptive branch stopping and claim-level verification;
- CPU pinned-memory offload;
- multi-GPU pipeline-parallel state layout;
- production deployment or service restart.

The local State-native MVP now implements same-Sidecar owner-scoped Prefill,
GPU Fork, Batch Continue, TTL and Release plus an opt-in four-branch/two-round
Agent endpoint. It remains unvalidated on the real 7.2B runtime and is not part
of the deployed default path; see `STATE_NATIVE_AGENT.md`.

The scheduler fails closed for pipeline-parallel/list-based state rather than
silently treating it as the single-GPU layout.
