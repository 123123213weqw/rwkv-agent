# RWKV Agent Core

`rwkv-agent-core` is the deterministic Rust control plane for the RWKV Agent.
It intentionally contains no HTTP client, CUDA code, search implementation,
database, terminal UI or operating-system command runner.

The core owns only:

- the strict `ToolCall | Answer` wire protocol;
- a typed tool registry and argument validation;
- a bounded `Model -> Tool -> Observation -> Model` loop;
- opaque recurrent-State identity checks and unconditional release;
- cancellation, elapsed-time and tool-step budgets;
- ordered lifecycle events.

Python remains the model and retrieval data plane. A later server crate can
implement the `StateModel` and `ToolExecutor` traits over the existing Sidecar
HTTP APIs without changing this loop.

The first implementation is isolated and mock-tested. It is not connected to
the public CLI request path or production service.
