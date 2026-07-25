# FineWiki hybrid retrieval v1

Milestone 5B public-safe result for isolated page-level long-term knowledge retrieval. The experiment uses the frozen milestone 5A MIRACL Chinese/English development cases and project compatibility cases without adding their queries or qrels to either index.

## Result

| Benchmark | Language | Lexical Hit@10 | Lexical + Cross-Encoder | Dense | Equal RRF | Equal RRF + Cross-Encoder |
|---|---|---:|---:|---:|---:|---:|
| MIRACL dev | zh | 43.51% | 50.13% | 61.58% | 65.14% | **67.18%** |
| MIRACL dev | en | 53.69% | 60.45% | 82.23% | 77.97% | **85.11%** |
| Project compat v1 | zh | 76.19% | 80.95% | 90.48% | 85.71% | **90.48%** |
| Project compat v1 | en | 45.00% | 50.00% | **85.00%** | 75.00% | 80.00% |

On MIRACL, the final equal-RRF plus Cross-Encoder path raises Hit@10 by 23.66 percentage points in Chinese and 31.41 points in English. Recall@10 rises by 19.47 and 25.16 points. The candidate ceiling also improves: Top-100 Hit rises from 67.43% to 77.61% in Chinese and from 66.96% to 91.74% in English.

The experiment therefore confirms two separate bottlenecks:

1. Page-level dense retrieval recovers many pages that lexical retrieval never placed in Top 100.
2. The Cross-Encoder moves already-recalled pages into the first ten and especially improves rank one.

## Method

- Lexical: the frozen FineWiki Candidate Index, Top 100.
- Dense: multilingual E5-small compatible 384-dimensional embeddings over title, aliases, headings and lead passage; Elasticsearch `int8_hnsw`, `num_candidates=1000`.
- Fusion: reciprocal-rank fusion with `k=60` and equal channel weights.
- Rerank: BGE reranker v2 M3 over the first 50 fused candidates.
- All qrels are used only after retrieval for scoring.

Weight sweeps were run on the same development sets. They are diagnostic, not held-out selection. The headline keeps equal RRF weights instead of publishing language-specific tuned weights as a production configuration.

## Admission result

A generic maximum reranker-score threshold of zero rejected 6/7 project missing probes. It lost 0.51 points of Chinese MIRACL Hit@10 and 2.88 points of English MIRACL Hit@10. Seven missing probes are not enough to calibrate a production abstention policy, so no threshold was enabled.

## Resource boundary

The page vector indexes contain 1,256,291 Chinese pages (5.7 GB) and 6,498,758 English pages (20.7 GB). The measured dense plus Cross-Encoder process peaked at 3,292 MiB GPU memory. No model weights, raw hits, queries, server paths or endpoint details are included here.

This milestone did not modify production Candidate Index defaults, chat routing, answers, services or frontend.
