# C Feedback Integration Smoke v1

This frozen artifact verifies the default-off `CFeedbackPlanner` inside the real
`RealtimeSearchEngine` on the RTX 4080 experiment host. It does not enable or
restart any production, chat, frontend, proxy, or SearXNG service.

## Conditions

- RWKV7 G1I 7.2B, greedy decoding, 192-token action ceiling.
- Flat P4 `web_search(query)` Tool Call for Q1 and at most one Q2.
- Direct Bing CN HTML discovery, IPv4, same outlet family as milestone 4C-6.
- Q1 and Q2 candidates are URL-deduplicated and RRF-merged before one fetch/extract phase.
- The runtime receives only the user query and live candidates. Benchmark domains,
  target URL patterns, categories, and labels are used only after retrieval for metrics.

## Result

Five prior feedback-sensitive cases were run. Four executed Q2 and all four exposed
`model_feedback` candidate provenance. The fifth generated a strict Q2 but the generic
duplicate-query guard rejected it. No case exceeded two discovery requests. The safe
trace contains stage, accepted query, timing, token count, validation, and gate state;
it excludes raw tokens, raw output, and private reasoning.

The five-case recall numbers are diagnostic only and must not be compared with the
50-case quality benchmark. `comparison.json` compares only protocol and request budget.
Raw queries, URLs, snippets, page content, model paths, and token traces remain under
ignored `bench/runs/` and are not part of this public baseline.
