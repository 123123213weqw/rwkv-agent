# State-native parallel Agent MVP

Status: real-G1I-validated, localhost-only opt-in implementation. It is exposed
by the isolated 8118/8119/8120 Agent release but does not change the default
`/v1/agent/run` path.

## Execution model

One turn is pinned to one Sidecar. The Sidecar prefills the common root once,
copies that recurrent state into bounded GPU slab slots, advances branch
continuations as exact BxT waves, and greedily decodes active branches as Bx1.
Tool observations are appended to the same branch state. Branch tensors are
never merged; deduplicated Evidence is serialized into the retained root state
for the final continuation.

```text
root prefill once
  -> GPU fork B1..B4
  -> branch tool call batch
  -> concurrent Web I/O
  -> observation resume batch
  -> second tool call batch
  -> Evidence dedupe
  -> root resume and final answer
  -> release root and all branches
```

The first MVP deliberately uses two fixed rounds by default. It proves state
ownership and continuation; it does not claim that the current G1I prompt can
reliably make an adaptive stop/search decision. That requires a real-model
protocol benchmark before changing the loop.

## Internal Sidecar API

All operations require an `owner_id`; a state created for one Agent turn cannot
be read, forked, continued or released by another owner.

```http
POST /v1/states/prefill
POST /v1/states/{state_id}/fork
POST /v1/states/batch_continue
POST /v1/states/release
```

Example root:

```json
{
  "owner_id": "turn-session-uuid",
  "prompt": "System: ...\n\nUser: ...",
  "branch": "root"
}
```

Example batch resume:

```json
{
  "owner_id": "turn-session-uuid",
  "items": [
    {"state_id": "state-a", "input": "Tool: <tool_result>..."},
    {"state_id": "state-b", "input": "Tool: <tool_result>..."}
  ],
  "stop": ["</tool_call>"],
  "max_tokens": 96
}
```

Non-EOS stop tokens are committed to recurrent state before the next
continuation. This keeps a resumed branch equivalent to the text it actually
generated instead of resuming before the closing tool-call token.

## Opt-in Agent endpoint

```http
POST /v1/agent/run_stateful
Content-Type: application/json

{
  "session_id": "research-demo",
  "message": "What organization and author created RWKV?",
  "branch_width": 4,
  "max_rounds": 2
}
```

The four generic missions are primary discovery, official-source lookup,
independent corroboration and ambiguity/conflict checking. They are not
topic/domain routing rules. Each branch emits one strict `web_search(query)`
per round. Web calls run concurrently; observations resume the matching states.

The Rust CLI exposes the same bounded request as:

```text
rwkv research "What organization and author created RWKV?"
/research What organization and author created RWKV?
```

The command defaults to B4 and two rounds. The non-interactive command accepts
`--branches 1..4` and `--rounds 1..3`. `/web` remains the direct one-shot tool.

## Bounds

| Bound | Default or enforced value |
|---|---:|
| Persistent states per Sidecar | 8 |
| Persistent-state TTL | 120 seconds |
| Branch width | 1-4 |
| Search rounds | 1-3 |
| Default turn state usage | 1 root + 4 branches |
| Final Evidence | 12 unique URLs/texts |
| Sidecar affinity | one Sidecar for the whole turn |

Environment variables:

```text
G1I_PERSISTENT_STATE_CAPACITY=8
G1I_PERSISTENT_STATE_TTL_SECONDS=120
```

TTL cleanup, owner mismatch, duplicate state IDs, context overflow, capacity
overflow and stale handles fail closed. The Agent releases all states in a
`finally` block and exposes release status in its trace.

## Current validation and boundaries

Local validation uses an Albatross-compatible Torch fake model:

- source-to-child GPU-layout state copies are exact and isolated;
- forked variable continuations match independent serial recomputation;
- B2 resume reaches B2T1 active decode;
- owner isolation, TTL cleanup, bounded capacity and final release pass;
- a fake end-to-end turn performs root prefill, four forks, two search rounds,
  eight tool executions, root resume and five-state release.

Real G1I 7.2B HTTP validation on a Tesla V100 additionally proves:

- all four first continuations and all four second Observation resumes are
  Token/Text/Stop exact against independent full-prompt recomputation;
- B4 emits a strict `web_search(query)` in both bounded rounds;
- cross-owner access returns HTTP 403;
- Root and all branch states return both persistent and scheduler allocation
  counters to zero;
- observed State smoke time was 3.526 seconds for equivalence, 2.704 seconds
  for the English B4/two-round protocol and 3.702 seconds for the Chinese
  B4/two-round protocol on the isolated candidate Sidecar.
- the full Chinese Controller path completed eight strict branch Tool Calls,
  concurrent live Web execution, 12-item Evidence reduction, Root resume and
  five-State release in 19.928 seconds.

Not yet validated:

- quality improvement over the frozen one-shot Web benchmark;
- adaptive early stop, claim-level verification, CPU pinned offload or
  cross-Sidecar state migration;
- public-production deployment or default CLI routing. The only deployment is
  the existing localhost-only Agent reached through the local SSH tunnel.
