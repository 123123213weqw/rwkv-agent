# Search Reasoning A/B/C/D v1

Frozen public result for four model-driven query/search protocols on the same 50-case Chinese/English realtime retrieval benchmark. Every arm used the same RWKV G1I checkpoint, greedy decoding, Bing CN HTML discovery, benchmark cases, and deterministic hard-constraint merge. Expected domains, target paths, categories, and labels were never shown to the model or runtime query logic.

- **A / direct:** current P4, one search.
- **B / short CoT:** requested short reasoning then one search.
- **C / feedback:** A plus at most one model-generated follow-up after a generic evidence-sufficiency gate.
- **D / bounded ReAct:** up to three reasoning/search/observation rounds.

Headline result: C raised Domain Recall@10 from 56% to 62% and Target Page Recall@20 from 10% to 14%, using 1.50 searches and 2.97 seconds on average versus A's 1.00 search and 1.64 seconds. B kept Domain Recall@10 at 56% and only moved Target@20 to 12%. D fell to 48% Domain@10, never emitted a valid stop action, and averaged 4.48 seconds.

A separate 384-token sensitivity check covered only the eight D cases that executed zero searches under the primary 192-token cap. It recovered three cases, but four still consumed the larger budget before producing a valid first action. This supports keeping C as the next shadow candidate and excluding current D from the ordinary fast path.

Raw token traces, generated queries, URLs, snippets, model/runtime paths, and search responses remain under ignored `bench/runs/` and are intentionally not frozen here. This result does not authorize production or shadow integration.
