# Preview4922 `run_command` same-State loop

This isolated benchmark executes the model's generated `run_command(command)`
calls in fresh fixture directories, returns each real command result as an
Observation, and continues from the same persistent RWKV State until the model
emits an answer or reaches the step limit. It did not change the Agent's public
tool protocol or any production service.

## Design

- six English/Chinese fixture tasks: read, compare, calculate, edit and verify,
  repair and test, and create and verify;
- two prompt styles: protocol instructions only, or the same instructions plus
  one complete Tool Call -> Tool Result -> Answer example;
- two greedy repeats per task and prompt style, for 24 formal runs;
- concurrency four, six-command limit, one persistent State per run;
- real command execution with an isolated working directory, eight-second
  timeout, bounded output, relative-path validation, and rejection of explicit
  network, package-management, process-control and absolute-path commands;
- task-specific filesystem/output verification, exact protocol parsing, State
  identity checking and unconditional State release.

## Formal bilingual result

| Prompt style | Passed | Protocol valid | Same State | Released | Repeat exact | Mean actions |
|---|---:|---:|---:|---:|---:|---:|
| Instructions only | 0/12 | 0/12 | 12/12 | 12/12 | 6/6 | 6.00 |
| One complete loop example | **10/12** | **10/12** | **12/12** | **12/12** | **5/6** | 2.67 |

All instruction-only runs successfully called tools but never switched to the
answer envelope; after satisfying the task they continued with redundant
commands until the step limit. A single complete closure example fixes this for
five of the six task families without adding another function.

The two remaining failures are both repeats of `fix_and_test`. In the formal
mixed C4 run, the model inspected both files, tried unavailable `pytest`, ran
the failing test through `unittest`, and then inspected both files again instead
of applying the repair. This is a real failure and is retained, not prompt-fitted
away.

## Focused concurrency finding

The unchanged `fix_and_test` task was then rerun four times sequentially and
four times as a homogeneous C4 batch. Both probes passed 4/4 with byte-identical
command/answer trajectories. Each run inspected the files, observed the failing
test, wrote the repair, reran the test and returned `PASS` in six actions.

This indicates that greedy argmax alone is not enough to guarantee an identical
agent trajectory under the current continuous-batching runtime: changing the
other requests sharing a mixed batch can change a near-tie token choice. Before
production integration, the runtime needs an explicit batch-invariance gate on
the same task set; prompt closure and State identity alone are insufficient.

## Decision

`run_command(command)` is sufficient to demonstrate the full execution loop:

```text
same State -> Tool Call -> controlled execution -> Observation -> same State
           -> next Tool Call or final Answer
```

The experiment validates the loop mechanism, task-side effects, Observation
feedback, State continuity and release. It does **not** yet justify exposing a
general shell in the Agent. The current executor is a benchmark guard based on
command validation, not a production OS sandbox; production work requires a
real isolation boundary, policy/audit layer, cancellation, and a frozen
batch-invariance/recovery benchmark.

Raw results remain under the ignored `bench/runs/` tree:

- formal bilingual C4 matrix SHA-256:
  `7beec2a85989db252e2d0d5a39b0f791166bfd1e573d429aae5e8517e5206bbe`;
- focused sequential SHA-256:
  `1388f2ee85c246d76db3b15feb655042c2850f70ae79298e5cbff5f39f749dc0`;
- focused homogeneous C4 SHA-256:
  `6b781434e15362e25fa983dbbc209326f2c88e59f9dac28fc52b12bf27fdf1e9`.

