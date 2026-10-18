# RWKV Search long-knowledge compatibility v1

This is a small, manually curated regression set for product-specific failure modes. It complements, but
never replaces, the larger MIRACL human-qrel benchmark.

- 48 cases: 24 Chinese and 24 English.
- Direct entities, aliases, natural questions, descriptions, comparisons, ambiguity, noisy spacing, and
  cross-language expressions.
- Seven explicit local-missing probes. They are excluded from Hit/Recall/MRR/nDCG and reported with a
  separate expected-missing accuracy.
- Positive labels use stable Wikipedia page IDs. Labels were reviewed independently of current retrieval
  ranks; the test queries and labels are never indexed.
- The query text is manually curated from observed project failure patterns and controlled probes. It is
  not presented as production user logs.

License for the query/qrel file: CC0-1.0. Wikipedia page identity and linked content remain subject to
Wikipedia's licenses.
