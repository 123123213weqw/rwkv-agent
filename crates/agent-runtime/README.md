# RWKV Agent Runtime

This crate is the Rust control plane for RWKV Agent. It owns:

- semantic-gate routing and the strict bounded Tool Call loop;
- persistent recurrent State lifecycle, per-session serialization and a bounded
  chat-State LRU;
- durable session transcripts without profile extraction;
- B1-B4, one-to-three-round parallel-State Web research;
- answer/Evidence validation calls and explicit release on success and failure;
- the optional `run_command(command)` capability.

It deliberately does **not** load RWKV weights, CUDA, Web crawlers, knowledge
indexes or long-text models. Those stay behind the Sidecar and Python data-plane
HTTP APIs. This keeps the latency-sensitive orchestration strongly typed while
avoiding a rewrite of stable Python/Torch and retrieval code.

`run_command` is disabled by default. Enabling it requires Linux, a configured
workspace and Bubblewrap. The executor unshares the network and namespaces,
binds only the workspace writable, applies time/output limits and never falls
back to an unsandboxed shell.

The full mock integration suite is `tests/mock_full_path.rs`.
