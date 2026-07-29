# Deployment

## Supported Beta topology

```text
rwkv-agent CLI
  -> Controller 127.0.0.1:8120
  -> Sidecar    127.0.0.1:8118
  -> one CUDA device
  -> structured discovery / optional SearXNG / bounded fallback
```

Start it with:

```bash
rwkv-agent-service doctor
rwkv-agent-service start
rwkv-agent doctor
```

Runtime state and PID files are stored under
`${XDG_STATE_HOME:-$HOME/.local/state}/rwkv-agent` unless
`RWKV_AGENT_STATE_DIR` overrides it.

## Service guarantees

The lifecycle script:

- binds both services to loopback by default;
- refuses ports occupied outside its PID files;
- waits for HTTP health before returning success;
- stores logs, PID files, sessions and CUDA extension cache outside Git;
- verifies recorded process command markers before sending a stop signal;
- never contains a built-in server address or SSH credential.

## Remote host

Run the backend on the GPU host. Use SSH port forwarding, a VPN, or an
authenticated reverse proxy to reach it. Example:

```bash
ssh -N -L 8120:127.0.0.1:8120 user@gpu-host
```

## SearXNG

The example under `deploy/searxng/` binds to loopback and caps memory. Replace
its example secret before startup. Engine health and quality depend on network
egress; SearXNG is a discovery source, not the crawler or answer generator.

## Not production-complete

Before exposing the API to untrusted users, add:

- authentication and per-user authorization;
- TLS termination;
- request and token rate limits;
- bounded request queues and admission control;
- outbound allow/deny policy and DNS rebinding protection;
- centralized logs and metrics;
- at least one warm standby Sidecar;
- backup and retention policy for session storage.

These gaps are tracked in [Known issues](KNOWN_ISSUES.md).
