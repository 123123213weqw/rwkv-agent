# Troubleshooting

## Start with Doctor

```bash
rwkv-agent-service doctor
rwkv-agent-service status
rwkv-agent doctor
```

The service Doctor validates local paths and Python imports. The CLI Doctor
validates the live Controller, model Sidecar, required tools and state research.

## Model does not start

Inspect:

```bash
rwkv-agent-service logs
```

Common causes:

- checkpoint and runtime are from incompatible revisions;
- `G1I_RUNTIME_DIR` does not contain `rwkv7_fast_v3a.py`;
- wrong CUDA device or insufficient memory;
- CUDA extension cache was compiled with another PyTorch/CUDA combination;
- pipeline-parallel environment variables leaked from another experiment.

The public service explicitly removes `G1I_PP_DEVICES` and uses one CUDA device.

## Port occupied

The service refuses to overwrite a process it does not own. Change
`RWKV_AGENT_SIDECAR_PORT` or `RWKV_AGENT_CONTROLLER_PORT`, or stop the process
that already owns the port.

## Search returns no Evidence

Check:

1. SearXNG health, if configured;
2. network egress and DNS;
3. `RWKV_AGENT_WEB_API_PROVIDERS`;
4. Tavily/GitHub credentials, if used;
5. Controller logs for discovery and fetch warnings.

The system may abstain when Evidence is missing. This is expected and preferable
to a fabricated answer.

## Tool Call JSON error

Preview4922 13.3B produced three malformed outputs in the fixed 40-case BFCL
sample. Retry the request once. Persistent failures should be saved as a
minimal redacted trace and added to the Tool Call regression set.

## Slow research

State research intentionally fans out up to four branches and two rounds by
default. Use smaller values when latency matters:

```bash
rwkv-agent research --branches 2 --rounds 1 "question"
```

The current FRAMES P95 was about 33 seconds on one V100. Search result caching,
early stopping and streaming progress remain optimization work.

## Remote CLI cannot connect

Verify the backend locally on the GPU host, then verify your tunnel separately:

```bash
curl -fsS http://127.0.0.1:8120/health
ssh -N -L 8120:127.0.0.1:8120 user@gpu-host
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8120 rwkv-agent doctor
```
