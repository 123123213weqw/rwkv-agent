# Long-text QA

## Scope

`long_text_qa(question)` answers from one long text pasted into the current
Agent session. It is a chat capability, not a file reader or upload subsystem:

```text
paste text -> transient session buffer -> ask question -> strict function call
           -> chunks -> independent RWKV states -> candidate/null
           -> grounded quotes -> deterministic reducer -> L# Evidence -> answer
```

The implementation borrows the reusable Chunk-State Map-Reduce pattern from the
supplied Three Body batch-QA project. It does **not** copy that reproduction's
gold-positive rules, fixed task IDs, eleven question-specific extractors or
tuned multi-state coefficients.

## Chat flow

1. Paste a UTF-8 text of at least 4,000 characters as one chat message.
2. The Controller stores one text for that `session_id` in a bounded RAM-only
   LRU buffer and immediately acknowledges it. No model inference is run.
3. The SQLite conversation transcript receives only a name/character-count
   placeholder. The Rust CLI also excludes long pasted messages from its
   persistent readline history.
4. Ask a normal question in the same session.
5. The greedy router sees only that question plus
   `Active pasted long text: yes` and emits `long_text_qa(question)`.
6. Tool execution obtains the source text internally from the session buffer.

Pasting another long text replaces the previous text in that session. Sessions
are isolated. The buffer holds at most 32 sessions and one million characters
per session by default; it is cleared when the Agent process exits. It is not
profile memory, embedding memory or a persistent document store.

## Runtime

1. Split the active text into 1,200-character chunks with 160-character
   overlap.
2. Rank chunks using character/word query features and corpus-local IDF.
3. Select Top-16 by default.
4. Submit up to eight chunk prompts concurrently through `ModelClient`.
5. Each completion runs on the greedy `ContinuousBatchEngine`; the Sidecar owns
   the actual GPU State slot.
6. Parse the `{"answer":`-prefixed output. A non-null candidate is accepted only
   when an exact quote or derived answer fragment exists literally in the
   source chunk.
7. Deduplicate candidates, rank by retrieval, question overlap and repeated
   support, and emit `L1...L8`.
8. Send the top candidate as `answer_hint` with its Evidence ID to the normal
   final-answer stage.

No padding tokens are advanced through RWKV state. Chunk workers use isolated
states released by the Sidecar after stop, EOS, limit or error.

## Function schema

```json
{
  "name": "long_text_qa",
  "arguments": {
    "question": "What does the pasted text say?"
  }
}
```

There is deliberately no `path`, `document_id`, attachment or source-text
argument. The model never copies the long source into a function call.

## Validation boundary

The external benchmark runner may read a fixture from disk to simulate the
user's first pasted chat message. The runtime function itself receives text
from `SessionTextBuffer`, never opens the fixture path.

The six-question Three Body smoke uses gold answers only after inference for
scoring. Runtime retrieval, worker prompts, parsing and reduction never read
answer labels or positive chunk IDs.

This is a first general Map-Reduce implementation, not a claim of universal
long-document QA quality. Multi-chunk synthesis, tables, PDFs/Office parsing,
cross-question cached chunk states and learned candidate reranking remain
future work.
