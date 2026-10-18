# Agent Hybrid Knowledge Shadow v1

This frozen benchmark validates the desktop Agent integration only. It does not call the RWKV G1I 7.2B model and does not change the live 8120 service. The visible `knowledge_search` result stays on the legacy lexical FineWiki path while the Hybrid arm runs as a bounded, failure-isolated shadow.

## Frozen setup

- 24 manually reviewable compatibility queries: 12 Chinese and 12 English.
- Same query and known relevant page IDs for both arms.
- Legacy: existing lexical Candidate Index, Top-5.
- Hybrid: lexical Top-100 + page E5 Dense Top-100, equal-weight RRF, CrossEncoder Top-50 rerank, Top-5 page-restricted 12-chunk `lead_plus_cross` hydration.
- Full queries, Evidence text, model paths, endpoint and per-case traces remain in ignored experiment storage.

## Result

| Metric | Legacy | Hybrid |
|---|---:|---:|
| Hit@1 | 15/24 (62.5%) | 15/24 (62.5%) |
| Hit@5 | 18/24 (75.0%) | 21/24 (87.5%) |
| Mean latency | 851.9 ms | 1704.8 ms |
| P95 latency | 2179.9 ms | 3088.7 ms |
| Empty | 0 | 0 |
| Hybrid fallback | - | 0 |

Chinese Hit@5 improved from 9/12 to 12/12. English Hit@5 stayed 9/12. At Hit@1, Hybrid won six cases and regressed six cases, so the net metric did not improve. Therefore the isolated Shadow integration passes its safety/reproducibility gate, but Hybrid is **not** approved as the Agent default.

Notable generic failure classes are entity/list-page over-ranking, ambiguous species/title interpretation, and CrossEncoder replacement of an exact entity page with a broader explanatory page. These need ranking review before production switching.
