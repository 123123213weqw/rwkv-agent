# StatePool remote container smoke — 2026-08-26

Status: measured build and HTTP smoke evidence. This is not a GPU, KEDA,
PostgreSQL, MinIO or live RWKV restore result.

## Environment

- build host: `WZU_Server`;
- Docker Engine server: `28.5.2`;
- repository branch: `codex/statepool-cloud-plugin`;
- source included uncommitted Dockerfile fixes following commit `c0aa5ed`;
- Docker build network: host network, because the host's Docker bridge DNS
  could not resolve Debian/crates.io during the first attempt;
- Rust build occurred only inside the remote Docker build, in compliance with
  `AGENTS.md`.

## Commands

```bash
docker build --network=host -f deploy/statepool/Dockerfile \
  --target statepool-plugin -t rwkv-statepool-cloud-plugin:remote-test .
docker build --network=host -f deploy/statepool/Dockerfile \
  --target agent-controller -t rwkv-agent-controller:remote-test .

docker run -d --rm --name rwkv-statepool-smoke \
  -p 127.0.0.1:18130:8130 \
  --tmpfs /var/lib/statepool/states:uid=65532,gid=65532,mode=0700 \
  rwkv-statepool-cloud-plugin:remote-test
curl --fail http://127.0.0.1:18130/live
curl --fail http://127.0.0.1:18130/metrics
docker stop rwkv-statepool-smoke
docker run --rm rwkv-agent-controller:remote-test --help
```

## Observed output

```text
Finished `release` profile [optimized] target(s) in 48.71s
statepool-plugin image sha256:c47ade9447d61f5672450edcbb1d15e0c4351f41cc04c48e2565582bc8cea06c
agent-controller image sha256:0671e05a58418cc4a50ad32ffcbbc380dc50ca511475e3163f3f2aa7908092fb
statepool_leases_acquired_total 0
statepool_pending_requests 0
{"contract_version":"statepool-plugin.v1","plugin":"statepool-cloud","status":"alive"}
remote-container-smoke-ok
```

The image hashes above identify local remote-host build outputs; they are not
published registry digests and must not be used as pull references.
