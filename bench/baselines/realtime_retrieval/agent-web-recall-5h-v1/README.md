# Agent Web Recall Repair 5H v1

This is the sanitized frozen result for milestone 5H. It repairs the recall and empty-result regressions found in the previous Agent Web Shadow run without changing the Agent-visible Legacy `W1..W5` path.

## What changed

- Long English requests are compacted into subject-first search queries; this is generic lexical cleanup, not topic or site routing.
- Candidate reranking may reorder the admitted first page but cannot evict its original Top-10 set.
- Legacy and Enhanced adapters share a bounded process-local discovery cache when they issue the same external query.
- Failure traces now distinguish discovery-empty, admission-empty, fetch failure, post-fetch rejection, and final-ranking-empty.
- If Enhanced produces no public Evidence, Shadow records an auditable Legacy-Evidence fallback. It still does not change the visible answer.

## Engine finding

A current SearXNG build was deployed only in the isolated experiment environment. Bing and DuckDuckGo initially returned useful pages, but sustained runs produced connection timeouts/resets and anti-bot responses through the available proxy exits. Smaller engines did not provide sufficient fresh bilingual recall. That diagnostic was rejected rather than presented as a healthy multi-engine result.

The accepted frozen run therefore uses the existing Bing HTML fallback with no paid search API. It proves the query/admission/fetch repair, not durable SearXNG operation.

## Same-set result

| Metric | Paired Legacy | Enhanced |
|---|---:|---:|
| Candidate Domain Recall@5 | 26% | **48%** |
| Candidate Domain Recall@10/20 | 30% | **50%** |
| Candidate Target Page Recall@10/20 | 8% | **14%** |
| Result Domain Recall@10/20 | 24% | **40%** |
| Result Target Page Recall@10/20 | 6% | **10%** |
| Non-empty fetched results | 72% | **88%** |
| Public Evidence non-empty | 96% | **98%** |
| Garbage result rate | 9.15% | **1.12%** |
| Fetch success | 56.52% | **71.35%** |
| Mean / P95 latency | 3291.3 / 8008.4 ms | 3301.5 / **6901.9 ms** |

Paired outcomes were 10-0 for Enhanced on candidate Domain Recall@10, 8-0 on result Domain Recall@10, and 8-0 on non-empty fetched results. English candidate Domain Recall@10 rose from 0% to 40%; Chinese remained 60% in both arms while sharing the same discovery response.

Compared with the earlier frozen 5G Legacy reference, Enhanced also exceeds candidate Domain Recall@10 (50% vs. 32%) and non-empty rate (88% vs. 86%), while reducing garbage from 17.56% to 1.12%.

## Decision

- Retrieval repair gate: **passed** on the frozen 50-case set.
- Durable SearXNG/multi-engine gate: **not passed**.
- Automatic visible-default switch: **not performed**.
- Production/service state changed: **no**.

Full queries, URLs, bodies, and per-case traces remain in ignored experiment storage and are excluded from this public baseline.
