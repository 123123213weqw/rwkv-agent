# Candidate Rerank BGE-M3 v1

Frozen public summary for an offline rerank A/B over the exact two-run Bing candidate snapshot from milestone 4C-3. No search request was repeated.

Strategies:

- `raw`: Bing order.
- `admission`: generic hard filtering, metadata scoring and domain diversity.
- `semantic`: BAAI/bge-reranker-v2-m3 cross-encoder only.
- `hybrid`: hard filtering plus equal-weight semantic and metadata fusion, followed by domain diversity.

The cross-encoder received only the frozen P4 query and each candidate's title, normalized URL source, and snippet. Benchmark expected domains and target paths were used only after ranking for evaluation. Full URLs, snippets, model scores tied to URLs, and raw traces remain in ignored remote run data.

The hybrid candidate passes the offline gates, but this baseline does not authorize production or shadow integration.
