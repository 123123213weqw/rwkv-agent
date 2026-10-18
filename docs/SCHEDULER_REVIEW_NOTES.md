# Review notes

## Main invariants

1. Padding is never forwarded into RWKV state.
2. A `StateHandle` is valid only for its slot generation and device.
3. Reused slots are zeroed before reassignment.
4. Duplicate state slots cannot appear in one model batch.
5. All forwards sharing one model/pool use the same execution lock.
6. Continuation cannot exceed the configured total context.
7. A final non-EOS generated token is committed to state by default.
8. Only contiguous state tensors may be passed directly to Albatross kernels.

## First benchmark defect found and fixed

The initial optimization borrowed every consecutive slab slice. Because batch
is a middle dimension in both shift and WKV state, most narrow slices retain
the full slab's outer stride and are non-contiguous. Equal B8 cases happened to
use the full slab and passed, while some variable-length batches produced
greedy mismatches.

The final implementation checks all three tensor layouts. Narrow batches use
preallocated contiguous workspaces and explicit scatter-back. The complete
natural-prompt matrix then passed every greedy comparison.

## Review focus

- Whether the 1.23GB all-shape workspace cache is acceptable for the intended
  concurrency.
- Whether production should freeze quantum 64 or keep per-GPU startup
  autotuning.
- Whether short unique tails should wait briefly for another identical tail or
  run immediately as they do now.
- The desktop G1I Sidecar now exposes the pool through an internal continuous
  batch worker. Production deployment and persistent HTTP state remain separate
  decisions.
