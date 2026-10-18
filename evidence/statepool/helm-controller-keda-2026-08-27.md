# Helm Controller / Worker / KEDA wiring evidence — 2026-08-27

Scope: manifest rendering and safety validation only. This is not a live
Kubernetes or GPU scaling result.

Validated on `WZU_Server` with `alpine/helm:3.19.0` and Docker Compose v2 after
rsync of the current worktree:

```text
helm lint deploy/statepool/helm/statepool
helm template demo deploy/statepool/helm/statepool
helm template demo deploy/statepool/helm/statepool \
  --set worker.enabled=true --set autoscaling.enabled=true
helm template demo deploy/statepool/helm/statepool \
  --set controller.enabled=true
helm template demo deploy/statepool/helm/statepool \
  --set plugin.replicaCount=2 --set plugin.durable.enabled=true \
  --set plugin.durable.postgresSecret.name=statepool \
  --set plugin.durable.s3Secret.name=statepool \
  --set plugin.durable.s3.bucket=statepool
docker compose -f deploy/statepool/compose.yaml config --quiet
```

Results:

- Helm lint: 1 chart, 0 failures;
- default profile: 3 rendered resources;
- Worker + KEDA profile: 6 rendered resources;
- Controller profile: 6 rendered resources;
- two-replica PostgreSQL/S3 plugin profile: 2 rendered resources;
- Compose model: valid;
- rendered Worker advertises its Pod-IP endpoint;
- rendered ScaledObject includes `idleReplicaCount: 0`;
- rendered Controller enables the default-off fenced Cold lifecycle.

Negative safety gates were also rendered and all failed as required:

- Controller replicas > 1 before distributed Session admission;
- plugin replicas > 1 without PostgreSQL/S3 durable mode;
- autoscaled Worker termination grace shorter than 20 seconds.

The live KEDA 0→1→N→0 experiment remains E14 and is not satisfied by this
manifest evidence.
