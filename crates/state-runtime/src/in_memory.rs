use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ring::digest::{SHA256, digest};
use tokio::sync::RwLock;

use crate::{
    CONTRACT_VERSION, CheckpointRef, ContinueRequest, ContinueResult, CreateRequest, Placement,
    ProviderFuture, ProviderMode, ReleaseOutcome, RestoreRequest, SessionDescription,
    SessionHandle, SnapshotRequest, StateContractError, StatefulInferenceProvider,
};

#[derive(Clone)]
struct SessionRecord {
    handle: SessionHandle,
    durable_session_ref: String,
    payload: Vec<u8>,
    created_at_ms: u64,
    last_active_at_ms: u64,
    seen_tokens: u64,
}

#[derive(Clone)]
struct CheckpointRecord {
    reference: CheckpointRef,
    payload: Vec<u8>,
}

#[derive(Default)]
struct Store {
    sessions: HashMap<String, SessionRecord>,
    session_owners: HashMap<String, String>,
    checkpoints: HashMap<String, CheckpointRecord>,
    checkpoint_owners: HashMap<String, String>,
}

pub struct InMemoryConformanceProvider {
    store: RwLock<Store>,
    next_id: AtomicU64,
}

impl Default for InMemoryConformanceProvider {
    fn default() -> Self {
        Self {
            store: RwLock::new(Store::default()),
            next_id: AtomicU64::new(1),
        }
    }
}

impl InMemoryConformanceProvider {
    pub async fn allocated(&self) -> usize {
        self.store.read().await.sessions.len()
    }

    pub async fn corrupt_checkpoint_for_test(
        &self,
        checkpoint_id: &str,
    ) -> Result<(), StateContractError> {
        let mut store = self.store.write().await;
        let checkpoint = store
            .checkpoints
            .get_mut(checkpoint_id)
            .ok_or(StateContractError::StaleCheckpoint)?;
        checkpoint.payload.extend_from_slice(b"corrupt");
        Ok(())
    }

    fn identifier(&self, prefix: &str) -> String {
        let value = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{value:016x}")
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }

    fn checksum(payload: &[u8]) -> String {
        let value = digest(&SHA256, payload);
        let mut output = String::with_capacity(7 + 64);
        output.push_str("sha256:");
        for byte in value.as_ref() {
            use std::fmt::Write as _;
            let _ = write!(output, "{byte:02x}");
        }
        output
    }

    fn validate_owner(expected: &str, actual: &str) -> Result<(), StateContractError> {
        if actual.trim().is_empty() {
            return Err(StateContractError::InvalidRequest(
                "owner_id must not be empty".into(),
            ));
        }
        if expected != actual {
            return Err(StateContractError::OwnerMismatch);
        }
        Ok(())
    }

    fn validate_handle(handle: &SessionHandle) -> Result<(), StateContractError> {
        if handle.contract_version != CONTRACT_VERSION {
            return Err(StateContractError::ContractVersion);
        }
        handle.model_ref.validate()
    }

    async fn create_inner(
        &self,
        request: CreateRequest,
    ) -> Result<SessionHandle, StateContractError> {
        if request.owner_id.trim().is_empty() || request.durable_session_ref.trim().is_empty() {
            return Err(StateContractError::InvalidRequest(
                "owner_id and durable_session_ref must not be empty".into(),
            ));
        }
        request.model_ref.validate()?;
        let handle = SessionHandle {
            contract_version: CONTRACT_VERSION.into(),
            session_id: self.identifier("session"),
            owner_id: request.owner_id.clone(),
            provider_mode: ProviderMode::ContractTest,
            model_ref: request.model_ref,
        };
        let now = Self::now_ms();
        let record = SessionRecord {
            handle: handle.clone(),
            durable_session_ref: request.durable_session_ref,
            payload: Vec::new(),
            created_at_ms: now,
            last_active_at_ms: now,
            seen_tokens: 0,
        };
        let mut store = self.store.write().await;
        store
            .session_owners
            .insert(handle.session_id.clone(), handle.owner_id.clone());
        store.sessions.insert(handle.session_id.clone(), record);
        Ok(handle)
    }

    async fn continue_inner(
        &self,
        request: ContinueRequest,
    ) -> Result<ContinueResult, StateContractError> {
        Self::validate_handle(&request.session_handle)?;
        if request.input.is_empty() || request.token_budget == 0 {
            return Err(StateContractError::InvalidRequest(
                "input and token_budget must be non-empty".into(),
            ));
        }
        let started = Instant::now();
        let mut store = self.store.write().await;
        let expected_owner = store
            .session_owners
            .get(&request.session_handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        Self::validate_owner(expected_owner, &request.owner_id)?;
        let session = store
            .sessions
            .get_mut(&request.session_handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        session.payload.extend_from_slice(request.input.as_bytes());
        session.seen_tokens = session
            .seen_tokens
            .saturating_add(request.input.split_whitespace().count() as u64);
        session.last_active_at_ms = Self::now_ms();
        let text = request
            .input
            .chars()
            .take(request.token_budget as usize)
            .collect::<String>();
        Ok(ContinueResult {
            session_handle: session.handle.clone(),
            output_tokens: text.chars().count().try_into().unwrap_or(u32::MAX),
            text,
            seen_tokens: session.seen_tokens,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    async fn snapshot_inner(
        &self,
        request: SnapshotRequest,
    ) -> Result<CheckpointRef, StateContractError> {
        if !matches!(request.target_tier, Placement::Cpu | Placement::Disk) {
            return Err(StateContractError::UnsupportedPlacement(
                request.target_tier,
            ));
        }
        let mut store = self.store.write().await;
        let expected_owner = store
            .session_owners
            .get(&request.session_handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        Self::validate_owner(expected_owner, &request.owner_id)?;
        let session = store
            .sessions
            .get(&request.session_handle.session_id)
            .ok_or(StateContractError::StaleHandle)?
            .clone();
        let now = Self::now_ms();
        let reference = CheckpointRef {
            contract_version: CONTRACT_VERSION.into(),
            checkpoint_id: self.identifier("checkpoint"),
            owner_id: request.owner_id,
            provider_mode: ProviderMode::ContractTest,
            model_ref: session.handle.model_ref,
            placement: request.target_tier,
            checksum: Self::checksum(&session.payload),
            size_bytes: session.payload.len().try_into().unwrap_or(u64::MAX),
            created_at_ms: now,
            last_active_at_ms: session.last_active_at_ms,
            atomic: true,
        };
        reference.validate()?;
        store
            .checkpoint_owners
            .insert(reference.checkpoint_id.clone(), reference.owner_id.clone());
        store.checkpoints.insert(
            reference.checkpoint_id.clone(),
            CheckpointRecord {
                reference: reference.clone(),
                payload: session.payload,
            },
        );
        Ok(reference)
    }

    async fn restore_inner(
        &self,
        request: RestoreRequest,
    ) -> Result<SessionHandle, StateContractError> {
        request.checkpoint_ref.validate()?;
        let record = {
            let store = self.store.read().await;
            let expected_owner = store
                .checkpoint_owners
                .get(&request.checkpoint_ref.checkpoint_id)
                .ok_or(StateContractError::StaleCheckpoint)?;
            Self::validate_owner(expected_owner, &request.owner_id)?;
            store
                .checkpoints
                .get(&request.checkpoint_ref.checkpoint_id)
                .ok_or(StateContractError::StaleCheckpoint)?
                .clone()
        };
        if record.reference.provider_mode != ProviderMode::ContractTest {
            return Err(StateContractError::ProviderMismatch);
        }
        if record.reference.model_ref != request.expected_model_ref {
            return Err(StateContractError::ModelMismatch);
        }
        if Self::checksum(&record.payload) != request.checkpoint_ref.checksum
            || record.payload.len() as u64 != request.checkpoint_ref.size_bytes
        {
            return Err(StateContractError::ChecksumFailure);
        }
        let handle = self
            .create_inner(CreateRequest {
                owner_id: request.owner_id,
                durable_session_ref: format!("restored:{}", request.checkpoint_ref.checkpoint_id),
                model_ref: request.expected_model_ref,
            })
            .await?;
        let mut store = self.store.write().await;
        let session = store
            .sessions
            .get_mut(&handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        session.payload = record.payload;
        session.last_active_at_ms = Self::now_ms();
        Ok(handle)
    }

    async fn describe_inner(
        &self,
        owner_id: String,
        session_handle: SessionHandle,
    ) -> Result<SessionDescription, StateContractError> {
        let store = self.store.read().await;
        let expected_owner = store
            .session_owners
            .get(&session_handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        Self::validate_owner(expected_owner, &owner_id)?;
        let session = store
            .sessions
            .get(&session_handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        Ok(SessionDescription {
            session_handle: session.handle.clone(),
            durable_session_ref: session.durable_session_ref.clone(),
            placement: Placement::Gpu,
            created_at_ms: session.created_at_ms,
            last_active_at_ms: session.last_active_at_ms,
            state_bytes: Some(session.payload.len() as u64),
            seen_tokens: session.seen_tokens,
        })
    }

    async fn release_inner(
        &self,
        owner_id: String,
        session_handle: SessionHandle,
    ) -> Result<ReleaseOutcome, StateContractError> {
        let mut store = self.store.write().await;
        let expected_owner = store
            .session_owners
            .get(&session_handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        Self::validate_owner(expected_owner, &owner_id)?;
        Ok(ReleaseOutcome {
            released: store.sessions.remove(&session_handle.session_id).is_some(),
        })
    }
}

impl StatefulInferenceProvider for InMemoryConformanceProvider {
    fn create(&self, request: CreateRequest) -> ProviderFuture<'_, SessionHandle> {
        Box::pin(self.create_inner(request))
    }

    fn continue_session(&self, request: ContinueRequest) -> ProviderFuture<'_, ContinueResult> {
        Box::pin(self.continue_inner(request))
    }

    fn snapshot(&self, request: SnapshotRequest) -> ProviderFuture<'_, CheckpointRef> {
        Box::pin(self.snapshot_inner(request))
    }

    fn restore(&self, request: RestoreRequest) -> ProviderFuture<'_, SessionHandle> {
        Box::pin(self.restore_inner(request))
    }

    fn describe(
        &self,
        owner_id: String,
        session_handle: SessionHandle,
    ) -> ProviderFuture<'_, SessionDescription> {
        Box::pin(self.describe_inner(owner_id, session_handle))
    }

    fn release(
        &self,
        owner_id: String,
        session_handle: SessionHandle,
    ) -> ProviderFuture<'_, ReleaseOutcome> {
        Box::pin(self.release_inner(owner_id, session_handle))
    }
}
