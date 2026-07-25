# Agent Web Shadow v1

This is the sanitized frozen summary for milestone 5G. The desktop Agent kept the existing `web_search(query)` and visible `W1..W5` Legacy evidence unchanged while an optional, bounded, asynchronous Enhanced arm recorded private traces. Queue saturation, timeout, discovery, fetch, and logging failures are isolated to Shadow. The Shadow is disabled by default.

## Benchmark

- Exact frozen dataset: 50 manually reviewable realtime web retrieval cases (25 Chinese, 25 English).
- Dataset SHA-256: `6900404d43deac290b599f10ee3b1f6e2fb8d8db06f821b346809049ab2e57dc`.
- Same metrics and evaluator for both arms; execution order alternated 25/25.
- No G1I answer generation was called.
- The experiment host had no healthy local SearXNG, so both arms used the existing Bing HTML fallback. This result is **not** a SearXNG or metasearch engine-quality conclusion.

## Result

| Metric | Legacy | Enhanced |
|---|---:|---:|
| Candidate Domain Recall@5 | 28% | 24% |
| Candidate Domain Recall@10/20 | 32% | 26% |
| Candidate Target Page Recall@10/20 | 2% | 2% |
| Result Domain Recall@5/10/20 | 24% | 20% |
| Result Target Page Recall@10/20 | 0% | 0% |
| Non-empty result rate | 86% | 58% |
| Garbage result rate | 17.56% | **1.08%** |
| Fetch success rate | 72.99% | 58.77% |
| Mean / P95 latency | 2820.5 / 7223.5 ms | 2680.7 / 7072.9 ms |

Generic admission filtering removed nearly all obvious garbage, but it also over-filtered or failed to recover enough useful results. Enhanced lost three paired Domain Recall@10 cases and sixteen paired non-empty cases, with no paired Domain Recall@10 win. Alternating order showed a live-upstream/order effect, but Enhanced remained weaker even when it ran first.

## Decision

- Shadow/failure-isolation integration: **passed**.
- Enhanced as the Agent-visible default: **rejected**.
- Production or service state changed: **no**.
- Next work must first improve discovery/fetch recall and rerun this exact benchmark with a healthy SearXNG instance; it requires separate owner authorization.

Full URLs, queries, fetched content, and per-case traces remain in ignored experiment storage and are intentionally excluded from this public baseline.
