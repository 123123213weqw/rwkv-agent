use std::time::{SystemTime, UNIX_EPOCH};

use rwkv_statepool_cloud_plugin::metadata::{MetadataStore, PostgresMetadataStore};
use rwkv_statepool_cloud_plugin::state_store::{
    S3StateStore, S3StateStoreConfig, StateStore, sha256_checksum,
};
use rwkv_statepool_plugin_api::{
    ACQUIRE_LEASE_REQUEST_CONTRACT_VERSION, AcquireLeaseRequest, ModelRef,
    STATE_REFERENCE_CONTRACT_VERSION, StatePlacement, StateReference,
};

fn unique(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{prefix}-{}-{nanos}", std::process::id())
}

#[tokio::test]
#[ignore = "requires RWKV_STATEPOOL_TEST_POSTGRES_URL"]
async fn postgres_enforces_cross_instance_lease_and_state_cas() {
    let url = std::env::var("RWKV_STATEPOOL_TEST_POSTGRES_URL").expect("PostgreSQL test URL");
    let first = PostgresMetadataStore::connect(&url)
        .await
        .expect("first store");
    let second = PostgresMetadataStore::connect(&url)
        .await
        .expect("second store");
    let concurrent_session = unique("postgres-concurrent-session");
    let concurrent_request = |holder_id: &str| AcquireLeaseRequest {
        contract_version: ACQUIRE_LEASE_REQUEST_CONTRACT_VERSION.into(),
        session_id: concurrent_session.clone(),
        owner_id: "owner:concurrent-test".into(),
        holder_id: holder_id.into(),
        expected_state_version: 0,
        ttl_ms: 30_000,
    };
    let left_request = concurrent_request("concurrent-a");
    let right_request = concurrent_request("concurrent-b");
    let (left, right) = tokio::join!(
        first.acquire(&left_request, 900),
        second.acquire(&right_request, 900)
    );
    assert_eq!(usize::from(left.is_ok()) + usize::from(right.is_ok()), 1);
    let rejected = if let Err(error) = left {
        error
    } else {
        right.expect_err("second concurrent writer must lose")
    };
    assert_eq!(rejected.code(), "lease_held");
    let session_id = unique("postgres-session");
    let owner_id = "owner:durable-test".to_string();
    let acquire = |holder_id: &str, expected_state_version: u64| AcquireLeaseRequest {
        contract_version: ACQUIRE_LEASE_REQUEST_CONTRACT_VERSION.into(),
        session_id: session_id.clone(),
        owner_id: owner_id.clone(),
        holder_id: holder_id.into(),
        expected_state_version,
        ttl_ms: 30_000,
    };
    let lease = first
        .acquire(&acquire("worker-a", 0), 1_000)
        .await
        .expect("first lease");
    let conflict = second
        .acquire(&acquire("worker-b", 0), 1_001)
        .await
        .expect_err("one writer");
    assert_eq!(conflict.code(), "lease_held");
    let state_ref = StateReference {
        contract_version: STATE_REFERENCE_CONTRACT_VERSION.into(),
        state_id: unique("state"),
        session_id: session_id.clone(),
        owner_id: owner_id.clone(),
        version: 1,
        fencing_token: Some(lease.fencing_token),
        provider_mode: "rwkv_recurrent".into(),
        model_ref: ModelRef {
            model_id: "rwkv-test".into(),
            revision: "revision-test".into(),
            tokenizer: "tokenizer-test".into(),
            state_abi: "state-abi-test".into(),
        },
        placement: StatePlacement::Cold,
        worker_id: None,
        object_uri: Some("s3://state-bucket/test.state".into()),
        checksum: sha256_checksum(b"state"),
        size_bytes: 5,
        atomic: true,
        created_at_ms: 1_002,
        last_active_at_ms: 1_002,
        encryption: None,
    };
    first
        .commit_state(&lease, 0, state_ref.clone(), 1_002)
        .await
        .expect("commit v1");
    first.release(&lease, 1_003).await.expect("release");
    let next = second
        .acquire(&acquire("worker-b", 1), 1_004)
        .await
        .expect("next lease");
    assert!(next.fencing_token > lease.fencing_token);
    assert_eq!(
        second
            .current_state(&session_id, &owner_id)
            .await
            .expect("current state"),
        state_ref
    );
    assert_eq!(
        first
            .assert_lease(&lease, true, 1_005)
            .await
            .expect_err("stale holder")
            .code(),
        "stale_fencing_token"
    );
    second.release(&next, 1_006).await.expect("release next");
}

#[tokio::test]
#[ignore = "requires RWKV_STATEPOOL_TEST_S3_* variables and an existing bucket"]
async fn s3_adapter_round_trip_is_immutable() {
    let bucket = std::env::var("RWKV_STATEPOOL_TEST_S3_BUCKET").expect("S3 bucket");
    let store = S3StateStore::new(S3StateStoreConfig {
        bucket,
        region: std::env::var("RWKV_STATEPOOL_TEST_S3_REGION")
            .unwrap_or_else(|_| "us-east-1".into()),
        endpoint: std::env::var("RWKV_STATEPOOL_TEST_S3_ENDPOINT").ok(),
        access_key_id: std::env::var("RWKV_STATEPOOL_TEST_S3_ACCESS_KEY_ID").ok(),
        secret_access_key: std::env::var("RWKV_STATEPOOL_TEST_S3_SECRET_ACCESS_KEY").ok(),
        prefix: unique("adapter-test"),
        allow_http: std::env::var("RWKV_STATEPOOL_TEST_S3_ALLOW_HTTP").as_deref() == Ok("true"),
    })
    .expect("S3 store");
    let key = "session/v1.state";
    let object = store
        .put_immutable(key, b"durable-state")
        .await
        .expect("put");
    assert_eq!(store.get(&object.uri).await.expect("get"), b"durable-state");
    store
        .put_immutable(key, b"durable-state")
        .await
        .expect("idempotent put");
    assert!(store.put_immutable(key, b"other-state").await.is_err());
    store.delete(&object.uri).await.expect("delete");
}
