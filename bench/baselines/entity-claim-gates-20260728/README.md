# Entity Admission + Claim Support Gates (2026-07-28)

This frozen comparison isolates two production-quality safeguards for the RWKV state-native search Agent:

1. **Entity-aware evidence admission** rejects results that do not mention the stable subject anchor.
2. **Claim-to-evidence verification** releases only claims supported by their cited Evidence IDs; unsupported route, number and URL claims are dropped or cause safe abstention.

Additional source-layer fixes preserve a bounded structured-source lane, keep one latest record per provider-declared stage, cache identical GitHub entity expansion across parallel branches, and retain GitHub profile, repository-index and latest-commit records through final Evidence merging.

## Result

- Q1 no longer returns `Leo501` / `Zhuangfei Xu` or Cocos Creator contamination. It returns **Peng Bo**, the 35-repository GitHub index, and retrieves the latest commit record (`2026-07-23T08:56:31Z`). The generated latest-update sentence mixed that commit with an unsupported release claim, so the claim gate correctly removed the whole sentence. This remains the next answer-generation issue.
- Q2 no longer releases a fabricated metro/flight route. Both primary and fallback generations failed support verification, so the endpoint returns `insufficient_evidence`.

See `comparison.json` for checkable fields and `raw/` for complete requests, Evidence, generated answers and traces.
