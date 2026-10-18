use std::sync::Arc;

use rwkv_state_runtime::{
    ContinueRequest, CreateRequest, InMemoryConformanceProvider, ModelRef, Placement,
    RestoreRequest, SnapshotRequest, StateContractError, StatefulInferenceProvider,
};

fn model() -> ModelRef {
    ModelRef {
        model_id: "rwkv-test".into(),
        revision: "abc123".into(),
        tokenizer: "tokenizer-v1".into(),
        state_abi: "state-v1".into(),
    }
}

async fn create(
    provider: &InMemoryConformanceProvider,
    owner: &str,
) -> rwkv_state_runtime::SessionHandle {
    provider
        .create(CreateRequest {
            owner_id: owner.into(),
            durable_session_ref: format!("durable:{owner}"),
            model_ref: model(),
        })
        .await
        .expect("create")
}

#[tokio::test]
async fn round_trip_and_double_release_are_exact() {
    let provider = InMemoryConformanceProvider::default();
    let handle = create(&provider, "owner-a").await;
    let output = provider
        .continue_session(ContinueRequest {
            owner_id: "owner-a".into(),
            session_handle: handle.clone(),
            input: "expected_action=ack:S-123".into(),
            token_budget: 128,
        })
        .await
        .expect("continue");
    assert!(output.text.contains("ack:S-123"));
    let checkpoint = provider
        .snapshot(SnapshotRequest {
            owner_id: "owner-a".into(),
            session_handle: handle.clone(),
            target_tier: Placement::Cpu,
        })
        .await
        .expect("snapshot");
    assert!(checkpoint.atomic);
    let first = provider
        .release("owner-a".into(), handle.clone())
        .await
        .expect("release");
    let second = provider
        .release("owner-a".into(), handle)
        .await
        .expect("idempotent release");
    assert!(first.released);
    assert!(!second.released);
    let restored = provider
        .restore(RestoreRequest {
            owner_id: "owner-a".into(),
            checkpoint_ref: checkpoint,
            expected_model_ref: model(),
        })
        .await
        .expect("restore");
    let description = provider
        .describe("owner-a".into(), restored.clone())
        .await
        .expect("describe");
    assert!(description.state_bytes.unwrap_or_default() > 0);
    assert!(
        provider
            .release("owner-a".into(), restored)
            .await
            .expect("release restored")
            .released
    );
    assert_eq!(provider.allocated().await, 0);
}

#[tokio::test]
async fn owner_stale_model_and_checksum_fail_closed() {
    let provider = InMemoryConformanceProvider::default();
    let handle = create(&provider, "owner-a").await;
    let owner_error = provider
        .describe("owner-b".into(), handle.clone())
        .await
        .expect_err("cross-owner describe must fail");
    assert_eq!(owner_error, StateContractError::OwnerMismatch);
    let checkpoint = provider
        .snapshot(SnapshotRequest {
            owner_id: "owner-a".into(),
            session_handle: handle.clone(),
            target_tier: Placement::Disk,
        })
        .await
        .expect("snapshot");
    let mut mismatch = model();
    mismatch.revision = "other".into();
    assert_eq!(
        provider
            .restore(RestoreRequest {
                owner_id: "owner-a".into(),
                checkpoint_ref: checkpoint.clone(),
                expected_model_ref: mismatch,
            })
            .await
            .expect_err("model mismatch"),
        StateContractError::ModelMismatch
    );
    provider
        .corrupt_checkpoint_for_test(&checkpoint.checkpoint_id)
        .await
        .expect("corrupt");
    assert_eq!(
        provider
            .restore(RestoreRequest {
                owner_id: "owner-a".into(),
                checkpoint_ref: checkpoint,
                expected_model_ref: model(),
            })
            .await
            .expect_err("checksum failure"),
        StateContractError::ChecksumFailure
    );
    provider
        .release("owner-a".into(), handle.clone())
        .await
        .expect("release");
    assert_eq!(
        provider
            .describe("owner-a".into(), handle)
            .await
            .expect_err("stale handle"),
        StateContractError::StaleHandle
    );
}

#[tokio::test]
async fn cancellation_path_releases_every_allocated_session() {
    let provider = Arc::new(InMemoryConformanceProvider::default());
    let handle = create(&provider, "owner-cancel").await;
    let (cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(false);
    let provider_task = Arc::clone(&provider);
    let handle_task = handle.clone();
    let task = tokio::spawn(async move {
        let outcome = tokio::select! {
            _ = cancel_rx.changed() => Err(StateContractError::Cancelled),
            _ = tokio::time::sleep(std::time::Duration::from_secs(30)) => Ok(()),
        };
        let released = provider_task
            .release("owner-cancel".into(), handle_task)
            .await?;
        assert!(released.released);
        outcome
    });
    cancel_tx.send(true).expect("cancel");
    assert_eq!(
        task.await.expect("join").expect_err("cancelled"),
        StateContractError::Cancelled
    );
    assert_eq!(provider.allocated().await, 0);
}
