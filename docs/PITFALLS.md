# Pitfalls: operational lessons from the G1L fleet work

Every item below cost real debugging time on the 4090D deployment between the
G1k (20260930) and G1l (20261031) checkpoints. Each entry lists the symptom as
it first appeared, the root cause, the fix that shipped, and how to avoid
re-triggering it. Numbers come from the sessions in `COMPARISONS.md`.

---

## 1. Checkpoint releases without a config invite silent misassembly

**Symptom.** A freshly converted HF checkpoint generated fluent-looking but
wrong text. Logits were near-uniform; greedy decoding mostly echoed the prompt.
No error was raised anywhere.

**Root cause.** Preview checkpoints in `BlinkDL/temp-latest-training-models`
are bare `.pth` files: no `config.json`, no args, no tokenizer. Converters
guess geometry from defaults. `hidden_size=4096` is ambiguous — 32 heads × 128
or 64 × 64 both divide — and the wrong split reorders every per-head tensor
while shapes still line up, so `strict=False` loading succeeds silently.

**Fix that shipped.** Infer geometry from tensor shapes, never defaults:

```
blocks.0.att.r_k   [64, 64]   -> num_heads=64, head_dim=64   (unique solution)
blocks.0.att.w1    [4096,128] -> decay rank 128
ffn.key / emb      shapes     -> FFN size, vocab
count(blocks.N)               -> layers
```

**Prevention.** The official 20261031 releases ship a full `config.json` with
correct geometry plus a bundled `inference/` runtime. Prefer official releases;
for previews, run shape inference and verify the converted model's logits
against the bundled runtime (cosine ≥ 0.99) before serving.

---

## 2. Verify the probe before blaming the converter

**Symptom.** Two independent converters (rwkv7-hf, fla) produced "garbage"
output. Conclusion drawn: both converters misassemble weights.

**Root cause.** The comparison probe built models with `_from_config` and
**never loaded the weights** — random initialization produces exactly the
near-uniform logits that were attributed to the converters. Hours were spent
diffing converter code that was correct.

**Prevention.** Any coherence probe must `load_state_dict(..., strict=True)`
and print a checksum of at least one loaded tensor. If a probe's output looks
uniform across every implementation, suspect the probe.

---

## 3. fla and rwkv7-hf need mutually exclusive transformers versions

**Symptom.** Whichever package was installed last broke the other at import:
`Cache.__init__() got an unexpected keyword argument 'layer_classes'` (fla on
transformers 5.x) or missing `PretrainedConfig` (rwkv7-hf remote code on 4.x).

**Root cause.** fla 0.7 requires transformers 4.51; rwkv7-hf 0.8.1 requires
>=5.12. Both register `model_type=rwkv7` and both ship cache classes tied to
their transformers generation.

**Fix that shipped.** Separate interpreters per backend: a conda env (py3.11 +
torch 2.6 + transformers 4.51 + fla) for the fla sidecar; the main venv
(transformers 5.x) for albatross and rwkv7-hf tooling. Restart scripts select
the interpreter explicitly.

**Prevention.** Pin and isolate by backend. Never `pip install` one backend's
stack into the other's environment, even temporarily.

---

## 4. First-encounter Triton autotune starves concurrent requests

**Symptom.** Under 32-way concurrency, `batch_continue` calls failed in bulk at
~700 s. GPU sat near idle; logs showed no errors.

**Root cause.** Every new (batch × seq) shape triggers Triton autotuning on
first execution — minutes per shape. Queued requests wait behind tuning runs
until the client timeout fires. Prefill and decode autotune separately, so
warming one is not enough.

**Fix that shipped.** Startup warmup walks representative batch widths
(1/8/16/24/32 plus the configured max) through one 512-token prefill and three
decode steps each, before the sidecar reports ready.

**Prevention.** Any Triton-backed server under bounded-latency SLOs must warm
its shape matrix at startup. Keep the shape set small: the scheduler groups
rows, so five widths cover B1..B32.

---

## 5. The client timeout must exceed the server's request ceiling

**Symptom.** Large file generations aborted with `error sending request` at
~220 s while the sidecar kept computing to a 200 OK nobody read.

**Root cause.** The controller's HTTP client timed out at 190 s; the sidecar's
own request ceiling was 600 s. A full-budget 4096-token decode legitimately
takes several minutes on a loaded card.

**Fix that shipped.** Client timeout raised to 620 s — one notch above the
sidecar ceiling. The sidecar remains the binding limit; the client must never
cut a request the server would still be serving.

**Prevention.** Encode the ordering `client timeout > server request timeout >
task deadline` wherever the chain grows.

---

## 6. State pool arithmetic: every task costs at least two slots

**Symptom.** Fleet runs failed a chunk of tasks with
`persistent state capacity exceeded: 48+1>48`.

**Root cause.** Each task holds a root state plus a forked worker — two pool
slots — and protocol retries fork replacement workers. Admission bounds tasks,
not states, so tasks×2 (+1 for retries) must fit the pool. Stale states from a
previous fleet linger for the TTL (900 s) and eat capacity on back-to-back
runs. Filling the pool exactly also risks CUDA OOM once decode workspaces
allocate on top of resident states.

**Fix that shipped.** Admission below half the pool (16 tasks on a 48-slot
pool), a clean sidecar restart between measured runs, and a client that treats
429 as retryable with backoff.

**Prevention.** Size `pool ≥ 2 × max_active_tasks + retry headroom`, restart
the sidecar (or wait out the TTL) between benchmark runs, and never plan for
100% pool utilization.

---

## 7. 429 is a queue signal, not an error

**Symptom.** With bounded admission, over-capacity submissions failed fast;
throughput collapsed unless admission was raised — which then OOM'd (see #6).

**Root cause.** Admission control rejects immediately by default (a waiting
task holds ledger state and can deadlock the pool; see `29f6c46`). Clients
treated the 429 as fatal.

**Fix that shipped.** The fleet client retries 429 with a 15 s backoff (12
attempts). Rejected tasks re-enter as finishers release slots; the last fleet
run scored 32/32 with admission 16 on a 48-slot pool.

**Prevention.** Any submission loop in front of this controller must retry
429s. The optional wait mode (`RWKV_AGENT_ADMISSION_WAIT_SECONDS`) exists but
reject-and-retry composes better with the state pool.

---

## 8. Per-request INFO logging saturates the event loop at fleet scale

**Symptom.** Around 50+ concurrent sessions, health checks timed out and new
connections were refused although the GPU had headroom.

**Root cause.** uvicorn logged one INFO line per request; at fleet rates the
logging itself starved the event loop's accept path.

**Fix that shipped.** `log_level="warning"`, a 4096 accept backlog, and an
explicit keep-alive timeout in the sidecar's uvicorn config.

**Prevention.** Benchmark with production log levels. "The server died under
load" with idle hardware is a logging/IO story until proven otherwise.

---

## 9. torch.compile makes overhead-bound decode slower here

**Symptom.** Compiling the fla decode path (`reduce-overhead`) dropped
single-stream decode from 6.2 to 2.2 tok/s.

**Root cause.** CUDA graphs assume static shapes and static buffers; recurrent
state tensors are swapped every step, so the graph re-captures or falls back,
adding overhead instead of removing it.

**Prevention.** Do not apply whole-model `torch.compile` to recurrent decode
loops with per-step cache replacement. Kernel-level fusion (the bundled
runtime's decode path, or albatross) is the approach that works — it held
13.7 tok/s/row single-stream and ~170 tok/s aggregate at B32.

---

## 10. Batch sweet spots are per-phase, not per-model

Measured on G1L 7.2B @ 4090D, worth re-measuring per card:

```
decode (bandwidth-bound):  B1 14.7 tok/s → B32 126 tok/s aggregate, plateau
                           beyond 32 (split into two forwards)
prefill (compute-bound):   scales with width; 26× single-row at B32 with the
                           bundled chunk loop; fla Triton ~60× at any width
```

**Prevention.** Match admission to the decode plateau (32 here), and let
prefill ride whatever width the queue gives it. Treat "batch 32" as a property
of this card's memory bandwidth, not of RWKV-7 in general.

---

## 11. RAM disk belongs to weights and caches — never the ledger

**Symptom/setup.** `/home` sits on an HDD; model loads and Triton cache lookups
(random small reads) dominated cold starts.

**Fix that shipped.** Model directories and the Triton cache live under
`/dev/shm` (tmpfs, 63 GB); env files point `G1I_HF_MODEL_PATH` and
`TRITON_CACHE_DIR` there. Sidecar restarts fell from ~7 min to ~2 min (fla,
with warmup) and ~100 s (albatross).

**What stays on disk.** `sessions/` (the task ledger is the durability anchor —
volatile storage would break restart recovery) and `workspaces/` (deliverables).
tmpfs is volatile: repopulate after reboot (a 14 GB copy takes ~2 min from the
HDD) or accept the slower first start.

---

## 12. G1L answer closure is a checkpoint trait, not a protocol bug

**Symptom.** G1L (preview and official 20261031) writes files flawlessly but,
given an `<answer>` prefill, emits correct text and then keeps generating —
role-played turns, leaked training boilerplate — instead of stopping.

**Root cause.** Terminal-answer segments in the training data lack stop
conditioning (independent finding in the G1L prompt-format study). Confirmed
unchanged in the official release by direct probe.

**Fix that shipped.** Plain-answer mode (`RWKV_AGENT_PLAIN_ANSWER=1`):
unprefilled final turns, lenient plain-text acceptance armed only on
answer-forcing renders, and three deterministic closures (committed answer
after verification; written-file report on stall; synthesized exit on repeats).

**Prevention.** Do not remove the plain-answer machinery for G1L-class
checkpoints because single-task runs look fine without it — fleet-scale
closure failures are what it prevents.

---

## 13. Network-shaped failures that look like code bugs

Recurring environment traps during this work, worth recognizing on sight:

- **HF unreachable, mirrors fine.** School networks blocked huggingface.co
  intermittently. `hf-mirror.com` + `aria2c -x16` against the signed redirect
  URL sustained ~10 MB/s where single streams got 100 KB/s.
- **Tailscale path decay.** A direct 10 MB/s link silently became a relay at
  70 KB/s. `tailscale ping` from both ends restores/verifies the direct path
  before blaming the transfer.
- **sshd connection-reset storms.** Hundreds of fleet connections trip
  MaxStartups-style throttling; the box stays up, SSH just refuses. Wait it
  out or use the second host as a jumphost.
- **`pkill -f` self-matches.** A pattern like `pkill -f rwkv_agent` kills the
  invoking SSH session. Always bracket: `pkill -f "[r]wkv_agent"`.

---

## 14. Recipe: the measured 32-agent fleet run

The configuration that produced 32/32 in 73 s (26 tasks/min, G1l official on
albatross), assembled from the entries above:

```
sidecar:   clean restart before the run (no stale states), pool 48,
           batch 32, request timeout 1800 s, model on /dev/shm
controller: RWKV_AGENT_PLAIN_ANSWER=1, admission 16 (= pool/3),
           no wait mode — reject with 429
client:    submit all, retry 429 with 15 s backoff (12 attempts)
```

Two waves of 16 tasks, finishers release slots, retries absorb the seam.
Raising admission toward the pool limit trades the last few percent of wall
time for OOM risk — the measured choice is the safe one.
