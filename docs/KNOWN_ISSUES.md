# Known issues and optimization backlog

This file distinguishes a usable Beta from remaining quality and production
work. Items include the fixed Preview4922 13.3B regression, the later
time-isolated Fresh-Web-200-v1 blind run and the current deployment audit.

## P0: correctness and reliability

- **Malformed Tool Calls:** 3/40 BFCL generations were invalid JSON; protocol
  validity was 92.5%. Add constrained decoding or one bounded structural repair
  and keep strict schema validation after repair.
- **Unsupported answered claims:** FRAMES unsupported-claim rate on answered
  cases was 15.52%. Strengthen claim-to-Evidence entailment and suppress only
  unsupported clauses rather than accepting the complete answer.
- **Public API security:** the Rust Server and compatibility Controller have no
  authentication, TLS or rate limiting. Service v1 `owner_id` is an isolation
  key, not authentication. Loopback-only is the supported Beta deployment.
- **Full Debug Trace privacy:** Debug Trace and its API are disabled by default.
  Explicit `full` capture stores prompt, model and tool bodies for local
  diagnosis and must not be published, exposed remotely or used as training
  data.
- **No high availability:** one Sidecar is a single point of failure. Add warm
  standby, readiness-aware routing and bounded overload rejection.

## P1: search quality

- **Fresh-Web answer quality:** the sealed 200-case run completed every request
  and reached 81% Gold-domain recall, but answer Token F1 was only 11.19% and
  citation presence was 58.5%; it did not pass the release gate.
- **Exact-page retrieval:** Fresh-Web Gold exact-URL recall was 16.75% and cited
  exact-URL recall was 9.5%. Most failed cases reached the correct domain but
  did not retain the exact answer-bearing page through Evidence and citation.
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
- Fresh-Web P95 was 42.33 seconds and all 200 requests exceeded its 20-second
  budget;
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
- no packaged Linux service unit or container image;
- two-target CLI archives are published from reviewed Beta tags; the Rust Server
  and external Provider stack are currently source/configuration deployments.

## P2: model/runtime portability

- the public Preview4922 13.3B profile is release-verified on one 32 GB V100;
- the canonical service lifecycle also passed an isolated correctness smoke with
  the 7.2B RWKV Provider on one RTX 4080, but this is not a general portability
  or performance claim;
- Search Gate threshold `-3.2` is checkpoint-specific;
- CPU, quantized checkpoints and custom pipeline parallelism are unsupported;
- the external Albatross runtime is not vendored and must match the checkpoint;
- model inference, retrieval and evidence still use external Python Providers
  until their separate Rust parity gates pass;
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
