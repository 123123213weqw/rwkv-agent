# RWKV-Agent Unified Regression v1

A deterministic 200-case daily regression set selected only from the frozen FitGen Dev split. It contains 40 cases each from BFCL, WebWalkerQA, FRAMES, LongBench v2, and ALCE. Locked Fit-ID, Structural-OOD, and Fresh-Web Gold are not read.

`cases.jsonl` stays private because ALCE does not declare a publishable license. `manifest.json` and `index.jsonl` contain the reproducibility metadata.

The isolated 200-record Train fitting probe is documented in `fit200-v1.md`; its machine-readable metrics and artifact hashes are in `fit200-v1-summary.json`. This probe does not replace Fit-ID, Structural-OOD, Fresh-Web, or end-to-end Agent evaluation.

The same 200 cases were rerun end-to-end with Preview4922 13.3B in `e2e-p0-13b-preview4922-summary.json`. The compact 7.2B comparison is in `e2e-p0-13b-vs-7b-comparison.json`. The BFCL AST value in the compact summary is an offline replay of the stored decoded calls with pinned `bfcl-eval==2026.3.23`; no model generation was repeated.

`public-results-manifest.json` records the byte size and SHA-256 of every
publishable aggregate result in this directory.
