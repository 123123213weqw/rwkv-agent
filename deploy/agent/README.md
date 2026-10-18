# Agent host deployment

The public Beta runs one G1I model Sidecar and one Agent Controller on the same
CUDA host. Both bind to loopback by default. The service script only manages
processes whose PID files live under `RWKV_AGENT_STATE_DIR`.

```bash
cp .env.example ~/.config/rwkv-agent/rwkv-agent.env
$EDITOR ~/.config/rwkv-agent/rwkv-agent.env

cli/scripts/rwkv-agent-service doctor
cli/scripts/rwkv-agent-service start
rwkv-agent doctor
rwkv-agent chat
```

For a remote GPU host, run the service on that host and create your own SSH
forward. Do not expose port 8120 directly: the Beta HTTP server has no public
authentication or rate limiting.

```bash
ssh -N -L 8120:127.0.0.1:8120 user@gpu-host
RWKV_AGENT_ENDPOINT=http://127.0.0.1:8120 rwkv-agent doctor
```

SearXNG is optional and has a separate example in `deploy/searxng/`.
