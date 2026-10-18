# Query Coordinator P0 isolated A/B

- `comparison.json`: machine-readable comparison and verdict.
- `raw/`: immutable full response traces for old branch-aware v4, pre-P0 current, and P0.
- Runtime: V100, RWKV 13.3B greedy, isolated ports 8517/8520. Production port 8120 was not changed.

P0 validates and coordinates query views. It does not claim to fix entity resolution, evidence admission, or answer entailment. The end-to-end quality gate is still failing.
