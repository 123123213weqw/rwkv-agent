# FineWiki English × MIRACL dev baseline v1

Frozen page-level long-term-knowledge retrieval baseline over the full English FineWiki index and the public MIRACL English development qrels. Raw queries, retrieved chunks and evidence remain outside this directory; the public test set is under `bench/external/miracl-v1`.

The unconditional metrics are primary. `conditional_on_index_coverage` only diagnoses ranking after at least one positive page is known to exist in the target snapshot.
