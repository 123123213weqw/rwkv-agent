# Preview4922 native Tool Call prior probe

This isolated probe asks which small execution interface the greedy
`rwkv7-g1i-preview4922-13.3b` model already handles without demonstrations.
It does not execute any generated command and did not modify the production
Agent service.

## Design

- 12 Chinese/English tasks: four file reads, four exact writes and four command
  runs;
- seven interface arms;
- both ordinary text prefill and explicit prefix token ID `0`;
- two greedy repeats;
- 336 total completion rows;
- strict `<tool_call>` envelope, exact tool and argument-key checks, plus static
  path/content/command preservation checks.

## Results

| Interface | Passed | Repeat exact | Mean tokens | Mean model ms |
|---|---:|---:|---:|---:|
| native `read_file/write_file/run_command` trio | 48/48 | 100% | 33.50 | 2857 |
| `run_command(command)` | **48/48** | **100%** | 33.67 | 2721 |
| `execute_command(command)` | 48/48 | 100% | 33.58 | 2726 |
| `shell(command)` | 48/48 | 100% | 31.50 | 2620 |
| `bash(command)` | 48/48 | 100% | 30.92 | 2588 |
| `python(code)` | 40/48 | 100% | 51.96 | 3915 |
| invented `workspace(op,input)` | 34/48 | 100% | 34.00 | 2723 |

The flat command aliases all followed an explicitly supplied one-tool schema,
so their 48/48 scores alone do not identify a native name. The independent
token-ID-0 continuation probes do:

- a `read` name stem reconstructs `read_file(path)` and then `write_file`;
- a `write` stem reconstructs `write_file(path, content)` and then `read_file`;
- a `run_` stem reconstructs `run_command(command, optional timeout)`;
- an `execute_` stem reconstructs `execute_sql(database, query)`, not
  `execute_command`;
- an unconstrained tool name reconstructs `get_weather` in an OpenAI-style
  function schema.

`workspace(op,input)` is not a strong prior. It emitted extra `path/content`
keys, converted a required string input into an object, lost exact write
content or selected a different operation on 14/48 rows. `python(code)` was
valid JSON but did not preserve two exact-command/exact-content instructions
and used roughly 54% more output tokens than `run_command`.

## Decision

The smallest evidence-backed general action is:

```text
run_command(command)
```

It is both native-prior evidence and a 48/48 no-demonstration result. The
backend should apply a deterministic timeout, working directory and output
budget rather than asking G1I to generate those fields. A later Agent-loop
experiment should test real execution and Observation continuation before this
function is added to the user-visible protocol.

The complete raw result is intentionally retained under the ignored
`bench/runs/` tree. Its SHA-256 is
`b17ba3f96b93f531beedf72ae3a3fdbfd9c931318c88022fd2fd15f1362d0d7e`.
