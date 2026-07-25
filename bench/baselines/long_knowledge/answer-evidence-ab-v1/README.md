# Legacy vs Hydrated Evidence Answer A/B v1

This frozen offline experiment asks whether the 5D `lead + CrossEncoder`
Evidence variant improves final grounded answers relative to legacy Evidence.
It preserves the same retrieved pages, page order, answer prompt, model,
greedy decoding, and scoring rules in both arms.

## Fixed experiment

- Cases: 24 manually reviewable project regression questions (12 Chinese,
  12 English).
- Retrieval buckets: 17 cases contain a manually known relevant page in the
  top five Evidence items; seven do not.
- Answer model: `rwkv7-g1i-preview3260-7.2b`.
- Prompt: the current production grounded-answer prompt and one production
  citation-repair attempt.
- Input: top five Evidence items, each compacted to at most 900 characters.
- Output limit: 256 tokens per primary or repair generation.
- Dataset SHA-256:
  `5c6156d06491a105f1b1b5f0f4b4efba8abda4d80a5a8559b1210fe2041fcb06`.

The full Evidence, queries, prompts, model outputs, and endpoint assignments
remain in ignored benchmark traces. The known relevant-page list is not
included in the prompt.

## Result

The Evidence pairing itself was valid: page order was identical for every
case and Evidence text changed in 20/24 cases. The answer-quality comparison,
however, failed its answer-model admission gate:

| Metric | Legacy | Hydrated |
|---|---:|---:|
| Accepted grounded answer | 4/24 (16.67%) | 2/24 (8.33%) |
| Citation repair attempted | 21/24 (87.50%) | 22/24 (91.67%) |
| Correct-page strict grounded answer | 0/17 | 0/17 |
| Correct-page relevant citation | 0/17 | 0/17 |
| Wrong-page safe abstention | 3/7 | 1/7 |
| Mean model latency | 6.812 s | 7.409 s |
| P95 model latency | 8.991 s | 8.890 s |

The model usually produced a factual paragraph without an allowed `[S#]`
citation. The single repair attempt generally repeated the paragraph or
source text without repairing the citation. Therefore this run cannot
determine whether Hydrated Evidence improves grounded answer quality. A
zero-versus-zero strict score is an answer-format/model failure, not evidence
that the two Evidence strategies are equivalent.

An earlier smoke prompt using a synthetic `tool_result` demonstration was
rejected before freezing because the model sometimes copied the demonstration
answer. It is retained only in ignored diagnostic traces.

## Limitations

- The G1I 7.2B model is a tool-call preview model, not the project's intended
  13.3B chat answer model.
- The 24-case set is small.
- Relevant page IDs are known positives, not exhaustive qrels. The seven-case
  unmatched bucket may contain alternative supporting pages and must not be
  treated as a definitive hallucination set.
- This experiment did not run the final service-level extractive fallback.

## Decision

Do not enable Hydrated Evidence in production and do not optimize passage
selection from this answer result. The next separately authorized experiment
should first require the actual chat answer model to pass a small grounded
answer-format gate, then rerun this exact paired A/B with that fixed model.
No production service or user-visible answer was changed.
