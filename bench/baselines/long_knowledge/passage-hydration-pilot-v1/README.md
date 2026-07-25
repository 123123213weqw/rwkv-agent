# FineWiki Passage Hydration Pilot v1

This isolated pilot tests the prerequisite between the frozen 5B page-level Hybrid retriever and chat Evidence. It does **not** change Router, chat, frontend, production Candidate Index behavior, or answer generation.

## Design

1. Reconstruct the frozen `Lexical + Dense RRF + CrossEncoder` page order.
2. Take the top eight pages without using qrels at runtime.
3. Run a page-ID-restricted chunk query against the existing FineWiki lexical index.
4. Compare page lead, lexical chunk, CrossEncoder chunk, and a bounded `lead + CrossEncoder` Evidence payload.
5. Measure character-trigram recall against MIRACL's positive passage text from a different Wikipedia snapshot.

The primary metric is gold-passage **recall**, rather than F1, because Evidence may intentionally include both a lead and a query-specific passage. F1 penalizes useful additional context.

## Pilot scope

- Chinese: 40 deterministic cases, 116/116 requested positive passages available.
- English: 21 evaluable cases. Hugging Face Dataset Viewer exposes a partial English conversion; only 36/111 requested positive passages were available, so this is not a full English benchmark.
- Every selected case already has a correct page in frozen Hybrid Top-8. Therefore this experiment isolates passage hydration and must not be quoted as page-retrieval recall.

## Result

| Language | Chunks/page | Mean latency | Lead+Cross mean gold recall | Recall >= 0.8 |
|---|---:|---:|---:|---:|
| zh | 6 | 288.9 ms | 73.76% | 67.50% |
| zh | 12 | 351.3 ms | 75.18% | 67.50% |
| zh | 20 | 412.0 ms | 75.18% | 67.50% |
| en | 6 | 390.4 ms | 79.87% | 57.14% |
| en | 12 | 516.3 ms | 82.80% | 61.90% |
| en | 20 | 607.1 ms | 82.80% | 61.90% |

Depth 12 is the pilot knee point. Relative to depth 6, bounded lead+Cross Evidence improves mean gold recall from 73.76% to 75.18% in Chinese and from 79.87% to 82.80% in the evaluable English subset. Depth 20 does not improve the combined result and adds latency.

Keeping the lead is important: a pure CrossEncoder can prefer a historically relevant list over the lead containing the current direct answer. The bounded combination preserves direct definitions/current summaries while adding a question-specific section.

## Artifacts

- Public safe metrics: `comparison.json`
- Implementation: `bench/long_knowledge_passage.py`
- Runner: `bench/run_long_knowledge_passage_bench.py`
- Gold preparation: `scripts/prepare_miracl_passage_gold.py`
- Unit tests: `tests/test_long_knowledge_passage.py`
- Full selected text, qrels subsets, private gold passages, and per-query traces remain under ignored `bench/runs/` and the V100 data disk.

## Stop condition

This pilot selects a passage hydration candidate only. It has not been connected to `EvidenceBuilder` or chat. The next milestone requires separate owner review and authorization.
