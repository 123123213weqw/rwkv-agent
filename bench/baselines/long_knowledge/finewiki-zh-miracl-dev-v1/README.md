# FineWiki Chinese × MIRACL dev baseline v1

Frozen page-level long-term-knowledge retrieval baseline. It uses the full Chinese FineWiki index and the public MIRACL Chinese development qrels. Raw queries, results and evidence remain outside this directory; the public test set is under `bench/external/miracl-v1`.

The unconditional metrics are primary. `conditional_on_index_coverage` only diagnoses ranking after at least one positive page is known to exist in the target snapshot.
