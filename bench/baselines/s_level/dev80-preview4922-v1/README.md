# Dev80 Preview4922 S-Level gap baseline

This is a conservative partial projection of the frozen 13.3B Dev80 run onto
`RWKV-Agent-S-Level-v1`. It is a gap report, not a claim that Dev80 is the final
500-case release benchmark.

Sources:

- WebWalkerQA Dev80 case SHA-256:
  `38a5e79a3ba4cdef40d4e4fc1d717f0c0cb36365052618f164487612f526e28b`
- Retrieval snapshot SHA-256:
  `1fbe8bebac8f030b2de12be7abf27046619cd102a3ff05531ff51da6f20f53c6`
- Runtime: RWKV7 Preview4922 13.3B, no Tavily, one independent Live run.

Results:

- Both profiles: 41 gates, 4 passed, 37 failed, 18 missing measurements.
- Domain candidate recall: 95.0%.
- Exact page discovery macro recall: 39.375%.
- Final Evidence exact recall: 30.625%.
- Citation exact page recall: 23.125%.
- Unsupported claim rate: 12.6786%.
- Search latency: P50 28.249s, P95 38.223s.
- Status success: 100%; State Leak: 0.

Missing measurements fail closed. In particular this adapter does not infer
factual accuracy from token F1 and does not infer three-run stability from one
Live run.

Artifact SHA-256:

- `measurements.json`: `1289103270443227befaf7aa5fad4062f863a8261c30148b0b2d38f1c50f8159`
- `production-report.json`: `d1cfd1c0fce4c5b880f938eaa77b44b3a7db534ad0b1e9f3270fcd83c8b1dc75`
- `s-level-report.json`: `34ea9339d6fb4f569467679f9ce93c5d16b541c0e8ca3a251f451a32ab3a260d`
