# Mechanisms

Visual explanation of how this runtime works, and where it breaks. Every figure
is annotated with measured numbers from the deployment described at the bottom.

- [1. State vs KV cache](#1-state-vs-kv-cache)
- [2. The pool and the tree](#2-the-pool-and-the-tree)
- [3. Snapshot, restore, packages](#3-snapshot-restore-packages)
- [4. The tool loop, and where it breaks](#4-the-tool-loop-and-where-it-breaks)
- [5. The routing gate, and how it fabricates](#5-the-routing-gate-and-how-it-fabricates)
- [6. Admission control](#6-admission-control)

---

## 1. State vs KV cache

The premise everything else follows from.

```
TRANSFORMER                          RWKV
─────────────────────────────────    ─────────────────────────────────
                                     prefill once
prompt                               ┌──────────┐
  │                                  │  state   │  34,078,724 B
  ▼                                  │ 32.5 MiB │  (fixed)
┌────────────────────────┐           └────┬─────┘
│ KV cache               │                │
│  4K ctx  →  256 MiB    │                │  continue(new tokens)
│ 16K ctx  →    1 GiB    │                │  mutate in place
│ 64K ctx  →    4 GiB    │                ▼
│  grows with context    │           ┌──────────┐
└────────────────────────┘           │  state   │  still 32.5 MiB
                                     └──────────┘
  memory = O(context)                  memory = O(1)
```

Measured cost of re-entering a conversation:

```
history   396 tok → continue 246 ms
history 12,364 tok → continue 242 ms        0.98× — flat
```

31 concurrent sessions:

```
transformer, GQA-8, 16K:  31 × 1 GiB  = 31 GiB   → does not fit on 24 GB
RWKV:                     31 × 32.5 MiB = 0.98 GiB
```

Not "faster". **It fits.**

Consequence: because a state does not grow, it can be copied, saved and restored
at a cost that does not depend on how much was read.

---

## 2. The pool and the tree

### The pool

All states live in one slab allocated at startup. Slots are handed out and
returned; nothing is allocated per request.

```
startup
  ┌─────────────────────────────────────────────────────────┐
  │  slab  1.52 GiB   48 slots × 32.5 MiB                   │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  free list [3,4,5,...47]        allocate() → pop a slot
                                 release()  → push it back

  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │ 0 │ 1 │ 2 │ 3 │ 4 │ ... │47 │
  └───┴───┴───┴───┴───┴─────┴───┘
    ▲   ▲
    │   └─ in use
    └─ gate root, always resident

  every allocate():
      slot = free.pop()
      generation[slot] += 1     ← stale handles now invalid
      zero(slot)                ← previous tenant's data erased
```

Verified: 6 rounds of take-3 / release-3 returned `allocated` to the same value
every round. Without reuse, round 2 would already fail against capacity 48.
A released handle returns `404 unknown state_id`, and a previous tenant's secret
was not visible to the next occupant of the same slot.

### The tree

States form a tree by construction: a fork copies a parent, and the child records
its origin.

```
              root  (gate root, or a prefilled document)
               │
     ┌─────────┼─────────┐
     │         │         │
   alpha     beta     gamma
     │
   ┌─┴─┐
   │   │
  a1  a2

  each node stores:  parent_state_id, branch, seen_tokens, last_used_at
```

`parent_state_id` was recorded from the start but nothing read it. Two decisions
now use it:

```
release(node)   → releases the whole subtree
                  (a forked worker branch is disposable by construction)

expiry          → a node is reclaimable only when its WHOLE subtree has been
                  idle for the full TTL; the MOST RECENTLY USED member decides,
                  so one active descendant keeps its ancestors alive

  before:  parent expires, children survive  → tree splits, orphans
  after:   the tree is reclaimed as a unit
```

`GET /v1/states/orphans` is the sentinel for this. It must stay empty. A non-empty
result means some path dropped a parent without its children.

---

## 3. Snapshot, restore, packages

```
POST /v1/states/{id}/snapshot
  ┌──────────┐   GPU → CPU copy   ┌────────────────┐   base64   ┌─────────┐
  │  state   │ ─────────────────► │ 34,210,448 B   │ ─────────► │ 43.5 MiB│
  │ 32.5 MiB │                    │ raw payload    │            │ JSON    │
  └──────────┘                    └────────────────┘            └─────────┘
                                          │
                    /snapshot_raw ────────┘  application/octet-stream
                                            32.6 MiB, checksum in a header

  measured:  base64  664 ms  |  raw  456 ms
```

Restore requires that the snapshot's **source is no longer live**:

```
POST /v1/states/restore    → 409 "snapshot source remains live"
```

So restore moves rather than copies. But one payload can be restored repeatedly:

```
                 ┌──────────────┐
   payload  ────►│ restore #1   │──► state-A   917 ms
   (held by      │ restore #2   │──► state-B   974 ms
    the client)  │ restore #N   │──► state-N
                 └──────────────┘

   A and B are independent, and with the same greedy suffix they emit
   token-identical output. That is what makes a payload file-like.
```

A package keeps the payload on disk so it survives the process:

```
POST /v1/states/packages {name, state_id}
  ┌───────────────────┐        ┌──────────────────────┐
  │ live state        │ ─────► │ packages/name.bin     │ 34.2 MB
  └───────────────────┘        │ packages/name.json    │ checksum, model_ref,
                               └──────────────────────┘ seen_tokens

POST /v1/states/packages/{name}/instantiate
  ┌──────────────────────┐
  │ packages/name.bin    │ ─────► fresh state   271 ms
  └──────────────────────┘        (vs 664 ms over HTTP with base64)
```

### When it is worth it

```
     464 tok : prefill   501 ms  vs restore 935 ms   → prefill wins
   1,839 tok : prefill 1,047 ms  vs restore 925 ms   → break-even
   5,513 tok : prefill 3,828 ms  vs restore 926 ms   →  4.1×
  11,013 tok : prefill 6,451 ms  vs restore 922 ms   →  7.0×
  18,339 tok : prefill 10,925 ms vs restore 920 ms   → 11.9×

  restore is constant. prefill grows. below ~1.8K, just recompute.
```

---

## 4. The tool loop, and where it breaks

### The intended loop

```
        ┌──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
  ┌───────────┐   continue(state, observation)   ┌─────┴──────┐
  │   state   │ ───────────────────────────────► │   model    │
  └───────────┘                                  └─────┬──────┘
        ▲                                              │ one envelope
        │                                              ▼
        │                                     ┌─────────────────┐
        │                                     │ <tool_call> or  │
        │                                     │ <answer>        │
        │                                     └────────┬────────┘
        │                                              │
        │            observation                       │
        └──────────────────────────────────────────────┤
                                                       │
                                          tool? ───────┴─────── answer?
                                            │                    │
                                            ▼                    ▼
                                     ┌────────────┐        return
                                     │  registry  │
                                     │  + sandbox │
                                     └────────────┘
```

One envelope per turn. The controller validates schema, budget and phase before
anything executes.

### Failure 1: the envelope opens with a think block

The model opens every turn with `<think>` even when the prompt forbids it. The
reconstruction step decided whether to synthesise an opening tag with
`generated.starts_with("<tool_call>")`:

```
  model emits:            <think>...</think><tool_call>{...}
  starts_with check:      false
  reconstruction adds:    <tool_call>
  result:                 <tool_call><think>...</think><tool_call>{...}
                                     ▲
                          inner body is no longer valid JSON
```

Fix: skip a leading think block before the check. Single-file reads went **0/8 →
17/20**.

### Failure 2: the write path is 0/6, structurally

The controller prefills half a call into the **input**:

```
  INPUT   ... Assistant: write_file(path="out.txt", content="
  OUTPUT  Hello, world.\n")

  the parser looks for a complete envelope in the OUTPUT
  the opening tag only exists in the INPUT
  nothing bridges them for a NEW file
```

`controller_tool_from_answer` can synthesise a call from an answer, but it
requires `mutation_ready(&plan)`, which requires having read the target file. A
new file has nothing to read. Every write dies as a protocol error.

### Failure 3: two files in one message is 0/6

```
  one file  → native format, correct
  two files → the model switches to the OpenAI array form

      <tool_calls>[{"id":"call_c9c8c4c4c4c4c4c4...   ← repeats to max_tokens
```

The `id` field degenerates into repetition until the 4096-token cap. The envelope
never closes. Measured 0/6 at temperature 0, so it is not sampling noise.

---

## 5. The routing gate, and how it fabricates

Before a request reaches the tool loop, a single-token classifier decides
whether tools are needed at all.

```
                    ┌──────────────────┐
   message ────────►│  gate            │
                    │  logit(tool)     │
                    │  − logit(chat)   │
                    └────────┬─────────┘
                             │  margin
              ┌──────────────┴──────────────┐
        margin > −1.0                 margin ≤ −1.0
              │                             │
              ▼                             ▼
        ┌───────────┐              ┌─────────────────┐
        │ tool_loop │              │ direct_chat     │
        │ has tools │              │ NO TOOLS        │
        └───────────┘              └─────────────────┘
```

The problem: the margin for one fixed intent depends on the filename.

```
"Read the file <NAME> in the current working directory ..."

"ja-2ae158.txt"  margin −1.137  ┐
"ja-e9caf1.txt"  margin −1.041  │  5 of 6 cross the threshold
"ja-0fa96a.txt"  margin −0.938  ┘

"notes-*.txt"    margin −0.371     0 of 6 cross

  spread across filenames for ONE intent: 0.86
  threshold sits inside that band → ~10% of file reads routed to chat
```

What happens when it picks chat:

```
  file on disk:  "MARK-AAAA1111"        (14 bytes)
  gate:          margin −1.0195  → chat
  model has no file contents:
  answer:        "The file contains a short, informal, multi-paragraph text
                  that reads like a personal, conversational note. It begins
                  with a greeting and a question about the reader's ..."
  status:        ok          tool_result: null      tool steps: 0
```

**The failure is asymmetric, and that is the real problem:**

```
  misroute → tool    visible error     "protocol error: ..."      recoverable
  misroute → chat    SILENT            plausible prose, ok        undetectable
```

Fix: an explicit-artifact check runs before the gate. A message naming a file, a
path, a URL, or a command verb goes to the tool path deterministically, and the
trace records `mode=explicit_signal`. Fabrication rate went **10% → 0** on the
same prompts.

---

## 6. Admission control

The state pool is a hard 48 slots. The task queue was unbounded, so a burst could
drive the pool to empty and stay there.

```
BEFORE
  request ──► task queue ──► wait for a slot ──► execute
                   ▲              │
                   │              ▼
                   │        ┌──────────┐
                   └────────│ pool     │  alloc=48 free=0
                            │ 48 slots │  /ready → state_capacity unavailable
                            └──────────┘  only kill -9 recovers

AFTER
  request ──► admission check ──► execute
                   │
                   ├─ slot free?  yes ──► register, execute
                   └─ no ────────────────► 429 "fleet at capacity: 32 of 32"
                                            (nothing parked, no slot held,
                                             ledger stays Pending so a retry
                                             with the same request_id works)

  measured, n=48:  16 × 429 + 32 × ok
                   /ready stayed "ready" throughout
                   service still answering afterwards
```

Default `admission_wait_seconds = 0` rejects immediately. Waiting is dangerous
here because a parked task still competes for the same 48 state slots.

---

## Measured on

```
model      rwkv-g1l-7b-temp-3797
revision   preview3260            ← G1L has never been published
backend    Albatross faster3a_2607
hardware   1 × RTX 4090D, 24 GB
state      34,078,724 B per slot = 32 MiB wkv (fp32) + 0.5 MiB shift
context    24,576 configured; 23,188 measured as the last perfect score
```

Numbers are reproducible with the scripts in `~/rwkv-4090d/alb-test/`. The
boundaries above should be re-measured on a public checkpoint before being
generalised: the published RWKV-7 G1k series is named `ctx25600`, which matches
where degradation begins here, but that is a correlation from one preview build.
