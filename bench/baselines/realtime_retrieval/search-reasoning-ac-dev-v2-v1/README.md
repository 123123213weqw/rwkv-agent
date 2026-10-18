# Search Reasoning A/C — Dev v2 v1

This public-safe artifact contains a paired 100-case Discovery benchmark of A
(`direct`) and C (`feedback`). Both use the same RWKV7 G1I 7.2B greedy P4 action and
the same live Bing CN discovery outlet. For every case, C reuses A's exact Q1
candidates and may add at most one Q2, so repeated Q1 search volatility cannot be
misreported as feedback gain.

## Result

- Domain Recall@10: A **36%**, C **39%** (+3 points; 3 gains, 0 regressions).
- Target Page Recall@20: A **2%**, C **4%** (+2 points; 2 gains, 0 regressions).
- Domain Recall@20: A **36%**, C **40%** (+4 points; 4 gains, 0 regressions).
- Average discovery queries: A **0.97**, C **1.36** (+0.39).
- Average model-plus-search latency: A **1.598 s**, C **3.067 s** (+1.469 s).
- Gate triggered on 79 cases, generated 76 feedback actions, and executed 39 Q2 searches.

All Domain@10 and Target@20 gains occurred in the English half; Chinese was unchanged.
This supports C over A on the same cases, but C's absolute Target@20 of 4% is still far
from a production-quality target-page discovery system.

## Boundary

This is a Discovery-only experiment. It does not fetch pages or measure extraction,
evidence, answer quality, or final-result recall. The 100-case development set is
manually curated realistic data, not user logs or a blind test, and should not be
cross-compared numerically with the older 50-case set. No production service, runtime
rule, retrieval algorithm, or benchmark label was changed.

Raw queries, generated actions, URLs, snippets, model paths, and live diagnostics stay
under ignored `bench/runs/` and are excluded from this directory.
