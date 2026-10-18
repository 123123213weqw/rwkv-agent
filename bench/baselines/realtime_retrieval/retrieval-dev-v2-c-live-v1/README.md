# Realtime Retrieval Dev v2 — C Live Baseline v1

This public-safe baseline records the first full live run of the bounded C feedback
retrieval path on the 100-case development set. It is an isolated benchmark, not a
production deployment.

## Conditions

- RWKV7 G1I 7.2B with greedy decoding and the flat P4 `web_search(query)` action.
- Direct Bing CN HTML discovery through the existing experiment-host network path.
- At most one initial query and one conditional feedback query.
- Q1 and Q2 candidates are merged before a single fetch/extraction phase.
- Runtime input excludes expected domains, target paths, categories, styles, source
  labels, pair IDs, and all other benchmark gold metadata.

## Headline result

- Candidate Domain Recall@10: **42%**; organization-level Domain Recall@10: **51%**.
- Result Domain Recall@10: **30%**.
- Candidate Target Page Recall@20: **4%**; Result Target Page Recall@20: **3%**.
- Non-empty result rate: **86%**; garbage-result rate: **1%**.
- Fetch success: **348/553 (62.93%)**.
- Mean latency: **6.751 s**; P95 latency: **11.853 s**.
- Feedback Q2 executed for **46/100** cases; no case exceeded two discovery calls.

Inside this single C run, merging Q2 increased Domain@10 from 37 to 42 cases and
Target@20 from 2 to 4 cases, with no already-hit case regressing. This is only a
stage attribution, not an independent A/C experiment.

## Main failure buckets

The expected exact domain was absent in 49 cases; nine more found only an
organization-related domain. Thirty-eight found an expected domain but not the target
page. Four found a target candidate, and one of those was lost before final results.
Thus discovery of authoritative target pages remains the dominant quality bottleneck.

## Interpretation limits

The development set contains manually curated realistic queries, not real user logs
or a blind test. It has 50 bilingual topic pairs, so language comparisons are paired
but topic diversity is 50. Small category groups are diagnostic only. Labels were not
changed after this run; possible label issues are listed only as review work. Live
search results are volatile, and this v2 baseline is not directly comparable with the
older 50-case benchmark. No production service, configuration, or search algorithm was
changed by this run.

Raw queries, generated searches, URLs, snippets, page content, and fetch traces remain
under ignored `bench/runs/` and are intentionally excluded here.
