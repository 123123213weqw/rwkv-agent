# Cloud Lite durable-adapter evidence — 2026-08-27

Host: `WZU_Server`. These are real service-container integration tests, not
in-memory mocks.

## PostgreSQL

- image used: `postgres:17.6-bookworm`
- image digest: `sha256:f3bd19c606e442c3d7bdfa8002e03fe260a1023351e0ea4598032022b68dd6e3`
- two independent `PostgresMetadataStore` connections
- verified one active writer, State version CAS, persisted current State,
  monotonic fencing and rejection of the stale holder
- raw output: `postgres-adapter-integration-2026-08-27.txt`

Reproduce by starting a disposable PostgreSQL container and running:

```bash
RWKV_STATEPOOL_TEST_POSTGRES_URL='<test-url>' \
  cargo test -p rwkv-statepool-cloud-plugin --test durable_adapters \
  postgres_enforces_cross_instance_lease_and_state_cas \
  -- --ignored --exact --nocapture
```

This is the exact PostgreSQL version selected by the Compose profile.

## S3/MinIO

- MinIO image: `RELEASE.2025-04-22T22-12-26Z`
- MinIO image digest observed on the test host:
  `sha256:a1ea29fa28355559ef137d71fc570e508a214ec84ff8083e39bc5428980b015e`
- mc image: `RELEASE.2025-04-16T18-13-26Z`
- mc image digest observed on the test host:
  `sha256:aead63c77f9db9107f1696fb08ecb0faeda23729cde94b0f663edf4fe09728e3`
- verified conditional create, read, idempotent retry, conflicting immutable
  write rejection and delete
- raw output: `minio-adapter-integration-2026-08-27.txt`

Reproduce after creating a disposable test bucket:

```bash
RWKV_STATEPOOL_TEST_S3_BUCKET='<bucket>' \
RWKV_STATEPOOL_TEST_S3_ENDPOINT='<endpoint>' \
RWKV_STATEPOOL_TEST_S3_ALLOW_HTTP=true \
  cargo test -p rwkv-statepool-cloud-plugin --test durable_adapters \
  s3_adapter_round_trip_is_immutable -- --ignored --exact --nocapture
```

Credentials are provided only through environment variables and are not
written into the repository evidence.

## Combined Cloud Lite restart proof

The opt-in Compose profile was started with PostgreSQL 17.6, MinIO and the
plugin image. A Cold State was committed, the plugin container was restarted,
and the same version/checksum/payload was restored under a new fencing token.
Raw output: `cloud-lite-compose-restart-2026-08-27.txt`.
