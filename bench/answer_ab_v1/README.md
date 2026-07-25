# Answer Evidence A/B v1

This is a small, manually reviewable regression set for the offline
Legacy-Evidence versus Hydrated-Evidence answer experiment.

- Cases: 24 (12 Chinese, 12 English)
- Source: the project's frozen real-knowledge regression cases
- Scope: positive knowledge questions with manually recorded relevant page IDs
- Not included: user logs, model outputs, webpage bodies, endpoints, or credentials
- Dataset SHA-256:
  `5c6156d06491a105f1b1b5f0f4b4efba8abda4d80a5a8559b1210fe2041fcb06`

Each row records the source compatibility-case ID, language, answer query,
relevant page IDs, required answer terms, and forbidden terms. The relevant
page IDs are used only for evaluation and never appear in the answer prompt.

This set is intentionally small. Results must be reported separately for
correct-page and wrong-page retrieval buckets and must not be presented as a
general production answer-quality score.
