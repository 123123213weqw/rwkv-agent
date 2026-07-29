# Known issues and optimization backlog

This file distinguishes a usable Beta from remaining quality and production
work. Items are based on the fixed 200-case Preview4922 13.3B regression and
the current deployment audit.

## P0: correctness and reliability

- **Malformed Tool Calls:** 3/40 BFCL generations were invalid JSON; protocol
  validity was 92.5%. Add constrained decoding or one bounded structural repair
  and keep strict schema validation after repair.
- **Unsupported answered claims:** FRAMES unsupported-claim rate on answered
  cases was 15.52%. Strengthen claim-to-Evidence entailment and suppress only
  unsupported clauses rather than accepting the complete answer.
- **Public API security:** Controller has no authentication, user authorization,
  TLS or rate limiting. Loopback-only is the supported Beta deployment.
- **No high availability:** one Sidecar is a single point of failure. Add warm
  standby, readiness-aware routing and bounded overload rejection.

## P1: search quality

- **Weak open-Web discovery:** WebWalker Evidence coverage and domain recall
  both remained 20%; exact-page recall was 2.5%.
- **Exact-page retrieval:** FRAMES improved to 14.79%, which is useful but still
  too low for precise recent facts.
- **Evidence selection:** stronger models answer more often but can select a
  plausible non-answer passage. Reranking must optimize answer-bearing passages,
  source authority, freshness and diversity together.
- **Provider reproducibility:** the older 7.2B run manifest did not record API
  provider environment variables. Future runs must record provider names,
  response snapshots or replay hashes without recording credentials.
- **Dynamic pages:** static extraction is the fast path; bounded browser fallback
  for the small set of JavaScript-only pages is not yet a release feature.

## P1: latency and cost

- FRAMES P95 was 33.2 seconds on one V100;
- WebWalker P95 was 24.6 seconds;
- state research lacks evidence-sufficiency early stop;
- no cross-request model prefix cache is exposed at the product layer;
- no user-visible progressive search timeline is available in the current CLI.

Optimize with bounded parallel discovery/fetch, shared result cache, early stop,
streaming progress and latency budgets. Do not remove Evidence validation merely
to answer faster.

## P2: product experience

- the current Agent experience is CLI-first;
- Legacy Web Preview does not yet use the complete current Agent backend;
- no file upload, PDF/Office ingestion or persistent document library;
- pasted long text is process memory and is lost on restart;
- no source side panel, regenerate, stop-generation or per-source inspection UI;
- no packaged Linux service unit, container image or prebuilt release binaries.

## P2: model/runtime portability

- only Preview4922 13.3B on one 32 GB V100 is release-verified;
- Search Gate threshold `-3.2` is checkpoint-specific;
- CPU, quantized checkpoints and custom pipeline parallelism are unsupported;
- the external Albatross runtime is not vendored and must match the checkpoint;
- other GPUs need memory, kernel and correctness validation.

## Quality gates for stable v0.3.0

Before removing the Beta label:

1. Tool Call protocol validity at least 99.5% on the frozen suite;
2. no regression in BFCL official AST or LongBench choice accuracy;
3. materially higher WebWalker exact-page and domain recall;
4. unsupported-claim rate below 5% on answered multi-step cases;
5. authenticated deployment example with rate limits;
6. clean-install and 24-hour soak test on a documented supported host;
7. frontend connected to the same Agent contracts, or clearly kept out of the
   stable release surface.
