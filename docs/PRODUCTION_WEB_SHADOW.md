# Production Web Shadow Runbook

This runbook prepares a **non-visible** production observation path. It does
not route 10% of users to a new answer: every user continues to receive the
Legacy answer, while at most 10% of eligible `web_search` calls also run the
Enhanced retriever asynchronously.

## Safety contract

- Default off. No environment variable means no Shadow object or background work.
- Visible evidence and final answers always remain Legacy.
- Sampling happens before queue admission and before Enhanced retrieval.
- One worker and a bounded queue prevent Shadow traffic from exhausting the chat path.
- Queue saturation, candidate failure, empty Evidence, logging failure, and shutdown
  never replace or block visible Legacy Evidence.
- Production defaults to metrics-only telemetry: no query, URL, page body, Evidence
  body, or full trace is written.
- `RWKV_AGENT_WEB_SHADOW=0` is the kill switch. A process restart is required because
  the adapter is constructed at startup.

## Configuration

Add these values to the isolated Agent process environment:

```bash
RWKV_AGENT_WEB_SHADOW=1
RWKV_AGENT_WEB_SHADOW_SAMPLE_RATE=0.10
RWKV_AGENT_WEB_SHADOW_MAX_PENDING=2
RWKV_AGENT_WEB_SHADOW_LOG_MODE=metrics
RWKV_AGENT_WEB_SHADOW_LOG=var/web-shadow-metrics.jsonl
```

`RWKV_AGENT_WEB_SHADOW_SAMPLE_RATE` must be between `0` and `1` and
`RWKV_AGENT_WEB_SHADOW_MAX_PENDING` must be positive. Invalid values fail
startup instead of silently expanding Shadow traffic. `full` log mode remains
available for isolated benchmarks only and must not be used with user traffic.

## Preflight

Run without changing the public service:

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/test_web_shadow.py \
  tests/test_summarize_web_shadow_metrics.py

PYTHONPATH=src:. ruff check \
  src/rwkv_agent/tools/web.py \
  benchmarks/summarize_web_shadow_metrics.py \
  tests/test_web_shadow.py \
  tests/test_summarize_web_shadow_metrics.py
```

Before enabling Shadow, confirm that the normal Agent and its upstream
SearXNG/fallback providers are healthy. Shadow must not be used to mask an
unhealthy Legacy path.

## Observe

Summarize the metrics-only log:

```bash
PYTHONPATH=src:. python benchmarks/summarize_web_shadow_metrics.py \
  var/web-shadow-metrics.jsonl \
  --output var/web-shadow-summary.json
```

Review at least:

- record count versus eligible Web calls;
- `fallback_rate`;
- queue drops from request metadata/service logs;
- Legacy/Enhanced result and candidate counts;
- fetch success rate;
- Enhanced and total Shadow latency;
- Agent P50/P95, error rate, GPU/CPU, memory, and upstream request volume.

Metrics-only telemetry cannot measure answer correctness, exact-page recall,
or citation validity. Those remain offline release gates and must not be
inferred from candidate counts.

## Promotion and stop conditions

Keep the experiment invisible until all conditions hold for a representative
window:

1. Visible error rate, P95, and resource use do not regress beyond the agreed budget.
2. No unbounded queue growth, state leak, protocol leak, or HTTP 409 occurs.
3. Enhanced empty/fallback rate is no worse than Legacy.
4. The same frozen offline benchmark shows no quality regression.
5. A separate user-visible canary is explicitly approved.

Immediately disable Shadow if the visible path regresses, upstream traffic is
rate-limited, the log cannot be written safely, or private payload fields appear.

## Rollback

```bash
RWKV_AGENT_WEB_SHADOW=0
```

Restart only the explicitly approved Agent process, then verify that new
Shadow records stop and normal chat/search health remains unchanged. This
repository change does not perform that restart automatically.
