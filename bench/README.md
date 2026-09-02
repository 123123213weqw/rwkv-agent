# Benchmark surface

`bench/` keeps reproducible datasets, generic metrics, current runners and
reviewed historical evidence. Runtime outputs belong in ignored
`bench/runs/`.

## Current runnable groups

- realtime retrieval: `run_realtime_retrieval_bench.py`,
  `retrieval_{schema,metrics,failure_attribution}.py`;
- candidate admission/reranking: `run_candidate_*_bench.py`;
- Web extraction: `run_web_extraction_bench.py` and
  `web_extraction*.py`;
- SearXNG engine diagnostics: `run_searxng_engine_bench.py`;
- long knowledge: `run_long_knowledge_{bench,hybrid_bench,dense_bench}.py`
  and their generic metrics/schema helpers;
- recurrent Agent runtime: `long_lived_runtime_v1/` and the Rust runners in
  `crates/state-runtime/`.

Model-training, FitGen, old Python Controller, legacy Web-product and one-off
shadow/replay runners are no longer executable release surface. Their reviewed
results remain in `baselines/` and their source revisions remain available
from Git.

## Evidence boundary

- `baselines/` and `artifacts/` are immutable reviewed evidence;
- frozen SHA/Manifest files are not regenerated during repository cleanup;
- duplicated files inside checksummed historical bundles are intentional and
  are not deduplicated in place;
- model weights, page bodies, private traces, credentials and machine-local run
  logs must not enter Git.

## Basic checks

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/test_realtime_retrieval_bench_data.py \
  tests/test_retrieval_metrics.py

PYTHONPATH=src:. python bench/run_realtime_retrieval_bench.py \
  --config configs/benchmark.json \
  --case-id retrieval-zh-001 \
  --case-id retrieval-en-006
```

Gold fields are read only after retrieval for scoring. They must never be
added to prompts, generated queries, discovery requests, ranking features or
training data.
