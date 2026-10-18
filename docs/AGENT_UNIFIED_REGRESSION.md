# RWKV-Agent Unified Regression v1

This is a compact daily regression set built from the already frozen FitGen
Dev split. It does not replace the full 430-case Dev benchmark or either
locked release test.

## Size

| Dataset | Track | Cases |
|---|---|---:|
| BFCL | Tool protocol | 40 |
| WebWalkerQA | Web research | 40 |
| FRAMES | Multi-hop web research | 40 |
| LongBench v2 | Long text | 40 |
| ALCE | Citation grounding | 40 |
| **Total** |  | **200** |

BFCL uses equal 10-case coverage for `simple`, `multiple`, `parallel`, and
`parallel_multiple`. The other datasets preserve the frozen Dev distribution
across their applicable language, difficulty, domain, reasoning, context, and
subset fields through deterministic multi-stratum selection.

## Build

```bash
python benchmarks/create_unified_regression_set.py \
  --dev-dir /private/fitgen_v1/training/dev \
  --output-dir /private/agent-unified-regression-v1
```

The output contains:

- `cases.jsonl`: the 200 complete private benchmark cases;
- `<dataset>.jsonl`: five 40-case files in the layout accepted by the existing
  end-to-end Runner;
- `index.jsonl`: IDs, tracks, languages, and selected strata;
- `manifest.json`: input/output SHA-256, source counts, selected IDs, and
  distributions;
- `README.md`: scope and publication boundary.

The builder accepts only a frozen `training/dev` directory, rejects paths
containing `locked`, checks global ID uniqueness, and refuses to overwrite an
existing output directory. Individual rows retain their public dataset's
original split label (`v3-core`, `main`, `test`, `asqa`, or `qampari`); Dev
membership is defined by the frozen FitGen directory and its source hashes.

## Publication boundary

The full combined JSONL remains private because ALCE metadata does not declare
a publishable license. The manifest, index, and generation code may be shared.
Fit-ID, Structural-OOD, and Fresh-Web Gold are never read by this builder.
