# FineWiki Shadow Passage Pair v1

This frozen result validates the integration boundary between the existing
FineWiki Shadow retriever and the 5C `lead + CrossEncoder` passage hydrator. It
does not run answer generation and does not change Router, visible Evidence,
chat output, frontend, production configuration, or service state.

## Paired path

For every sampled chat-search request:

1. the existing FineWiki Shadow retriever returns its legacy `CandidateHit`s;
2. the same top eight page IDs are queried for up to 12 chunks per page;
3. `BAAI/bge-reranker-v2-m3` selects a question-specific chunk;
4. a bounded, at-most-3,200-character `lead + selected chunk` Evidence variant
   is created;
5. legacy and hydrated Evidence are written into one Shadow trace;
6. the original `CandidateHit`s remain the only data available to the explicit
   live-FineWiki adapter, so user-visible output cannot change in this stage.

Passage failure is caught inside the Shadow worker and falls back to legacy
Evidence. It cannot fail the primary chat request.

## Fixed experiment

- Dataset: 41 positive cases from the frozen project compatibility set
  (21 Chinese, 20 English).
- Indexes: existing read-only FineWiki Chinese and English lexical indexes.
- Candidate pages: existing Shadow retrieval, top eight.
- Passage depth: 12 chunks per page.
- Answer model: not executed.
- Full queries, retrieved text, URLs, and per-case traces remain under ignored
  `bench/runs/` and the V100 data disk.

## Result

| Metric | Result |
|---|---:|
| Hydration completed | 41/41 |
| Legacy fallback | 0/41 |
| Page order preserved | 100% |
| Legacy Page Hit@8 | 51.22% |
| Hydrated Page Hit@8 | 51.22% |
| Evidence text changed | 117/299 (39.13%) |
| Empty legacy retrieval | 1/41 |
| Visible output changed | No |

Page Hit@8 must remain identical because this stage is only allowed to hydrate
text for already-ranked pages. It does not claim to improve page retrieval.

After one model load per language process, mean hydration overhead was 279.0 ms
for Chinese and 349.6 ms for English; warm P95 was 435.3 ms and 587.0 ms. Cold
model loads were 7.45 s and 7.75 s. A future always-on Shadow deployment would
need explicit model warm-up before its latency can be judged against an online
SLO.

## Important limitation

Hydration cannot repair a wrong page. For example, an ambiguous Python query
can still retrieve snake or comedy pages; this stage only replaces the text
inside those returned pages. The unchanged 51.22% Page Hit@8 makes that
boundary explicit. The earlier 5C MIRACL passage-gold pilot measures passage
coverage only after the correct page is already present.

## Decision and stop condition

The old/new Evidence pairing, bounded passage payload, failure isolation, and
unchanged visible path work as designed. The feature remains disabled by
default. No production service was restarted.

The next experiment requires separate owner authorization. It should compare
offline answers generated from legacy versus hydrated Evidence on cases whose
retrieved page is relevant, while separately retaining wrong-page failures.
