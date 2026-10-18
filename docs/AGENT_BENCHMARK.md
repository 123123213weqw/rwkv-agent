# RWKV Agent Unified Benchmark Metrics

This layer evaluates normalized benchmark cases and run results. It does not run
the model, download public datasets, or change the production service.

## Tracks

| Track | Primary metrics |
|---|---|
| `tool_protocol` | tool-needed accuracy, false positive/negative calls, function/name/argument/group/sequence exact match, strict protocol validity |
| `web_research` | answer EM/F1, exact-source/domain/evidence recall and precision, requests, rounds, latency and budgets |
| `citation_grounding` | citation presence, valid-ID precision, exact-source/domain precision and recall, claim citation coverage and unsupported-claim rate |
| `long_text` | answer EM/F1, gold evidence-chunk recall, state reuse, latency and context-length buckets |
| `memory` | answerability, evidence recall, state reuse; dataset adapters can add memory-specific judged claims |
| `end_to_end` | the applicable metrics above plus resource and safety measurements |

Metrics are grouped by track, dataset, language, and optional `context_bucket`.
There is deliberately no grand score: a faster system must not hide worse
grounding, and better retrieval must not hide broken Tool Call syntax.

## Input case schema

Each line in the case JSONL uses `rwkv-agent-benchmark-case.v1`:

```json
{
  "schema_version": "rwkv-agent-benchmark-case.v1",
  "id": "frames-1",
  "dataset": "frames",
  "split": "test",
  "track": "web_research",
  "language": "en",
  "prompt": "...",
  "gold": {
    "answers": ["..."],
    "answerable": true,
    "requires_citations": true,
    "source_uris": ["https://example.org/source"],
    "evidence_ids": ["gold-chunk-1"]
  },
  "limits": {
    "max_rounds": 2,
    "max_requests": 8,
    "max_latency_ms": 20000
  },
  "metadata": {"context_bucket": "32k-128k"}
}
```

For `tool_protocol`, `gold.should_call_tools` and `gold.tool_calls` are
mandatory. Every expected call contains `name`, exact `arguments`, and an
optional `parallel_group`.

## Result schema

Each result line uses `rwkv-agent-benchmark-result.v1`:

```json
{
  "schema_version": "rwkv-agent-benchmark-result.v1",
  "case_id": "frames-1",
  "status": "ok",
  "answer": "Answer [W1]",
  "abstained": false,
  "tool_calls": [],
  "evidence": [
    {"id": "W1", "gold_id": "gold-chunk-1", "uri": "https://example.org/source"}
  ],
  "claims": [
    {"text": "Answer", "citations": ["W1"], "supported": true}
  ],
  "protocol": {"tool_call_valid": true},
  "trace": {
    "requests": 4,
    "rounds": 2,
    "states_created": 5,
    "states_released": 5,
    "states_leaked": 0,
    "states_reused": 4
  },
  "resources": {
    "latency_ms": 14000,
    "ttft_ms": 800,
    "gpu_peak_mib": 16000,
    "cpu_state_peak_mib": 120,
    "input_tokens": 4000,
    "output_tokens": 30
  }
}
```

`claims[].supported` must come from a deterministic dataset annotation or a
separately versioned verifier. The metric code does not pretend that URL overlap
is semantic claim support. Missing annotations are excluded from that metric's
denominator.

## Run the evaluator

```bash
python benchmarks/run_agent_benchmark_metrics.py \
  --cases /path/to/cases.jsonl \
  --results /path/to/candidate.jsonl \
  --baseline /path/to/frozen-baseline.jsonl \
  --output /path/to/report.json \
  --rows-output /path/to/evaluations.jsonl
```

The evaluator requires identical case IDs for a frozen paired comparison,
records SHA-256 for every input, reports metric-specific denominators, and
calculates paired wins/ties/losses in the correct higher/lower direction.

## Dataset adapter boundary

BFCL, WebWalkerQA, FRAMES, LongBench v2, ALCE, WebCPM, BrowseComp, LongMemEval,
and GAIA need separate import adapters. Adapters may normalize public examples
into this schema, but must preserve the dataset revision, split, license,
original ID, and immutable input SHA-256 in `metadata`. Public datasets have not
been downloaded as part of this milestone.
