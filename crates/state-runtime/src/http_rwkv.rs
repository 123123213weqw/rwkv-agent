use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::RwLock;

use crate::{
    CONTRACT_VERSION, CheckpointRef, ContinueRequest, ContinueResult, CreateRequest, ModelRef,
    Placement, ProviderFuture, ProviderMode, ReleaseOutcome, RestoreRequest, SessionDescription,
    SessionHandle, SnapshotRequest, StateContractError, StatefulInferenceProvider,
};

#[derive(Clone)]
struct SessionRecord {
    handle: SessionHandle,
    durable_session_ref: String,
    created_at_ms: u64,
    last_active_at_ms: u64,
    seen_tokens: u64,
}

#[derive(Default)]
struct Store {
    sessions: HashMap<String, SessionRecord>,
    owners: HashMap<String, String>,
}

pub struct RwkvHttpProvider {
    client: Client,
    endpoint: String,
    model_ref: ModelRef,
    bootstrap_prompt: String,
    store: RwLock<Store>,
    request_counter: AtomicU64,
}

#[derive(Deserialize)]
struct PrefillResponse {
    state: PrefillState,
}

#[derive(Deserialize)]
struct PrefillState {
    state_id: String,
    #[serde(default)]
    seen_tokens: u64,
}

#[derive(Deserialize)]
struct ContinueResponse {
    results: Vec<ContinueRow>,
}

#[derive(Deserialize)]
struct ContinueRow {
    state_id: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    token_ids: Vec<u32>,
    #[serde(default)]
    seen_tokens: u64,
}

impl RwkvHttpProvider {
    pub fn new(
        endpoint: impl Into<String>,
        model_ref: ModelRef,
        bootstrap_prompt: impl Into<String>,
    ) -> Result<Self, StateContractError> {
        model_ref.validate()?;
        let bootstrap_prompt = bootstrap_prompt.into();
        if bootstrap_prompt.trim().is_empty() {
            return Err(StateContractError::InvalidRequest(
                "bootstrap_prompt must not be empty".into(),
            ));
        }
        Ok(Self {
            client: Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            model_ref,
            bootstrap_prompt,
            store: RwLock::new(Store::default()),
            request_counter: AtomicU64::new(1),
        })
    }

    pub async fn allocated(&self) -> usize {
        self.store.read().await.sessions.len()
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }

    async fn response_json<T: for<'de> Deserialize<'de>>(
        response: reqwest::Response,
    ) -> Result<T, StateContractError> {
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(StateContractError::Provider(format!(
                "HTTP {status}: {}",
                body.chars().take(500).collect::<String>()
            )));
        }
        response
            .json::<T>()
            .await
            .map_err(|error| StateContractError::Provider(error.to_string()))
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
        if request.model_ref != self.model_ref {
            return Err(StateContractError::ModelMismatch);
        }
        let branch = format!(
            "rust-live-{:016x}",
            self.request_counter.fetch_add(1, Ordering::Relaxed)
        );
        let response = self
            .client
            .post(format!("{}/v1/states/prefill", self.endpoint))
            .json(&json!({
                "owner_id":request.owner_id,
                "prompt":self.bootstrap_prompt,
                "branch":branch,
            }))
            .send()
            .await
            .map_err(|error| StateContractError::Provider(error.to_string()))?;
        let response = Self::response_json::<PrefillResponse>(response).await?;
        let handle = SessionHandle {
            contract_version: CONTRACT_VERSION.into(),
            session_id: response.state.state_id.clone(),
            owner_id: request.owner_id.clone(),
            provider_mode: ProviderMode::RwkvRecurrent,
            model_ref: request.model_ref,
        };
        let now = Self::now_ms();
        let record = SessionRecord {
            handle: handle.clone(),
            durable_session_ref: request.durable_session_ref,
            created_at_ms: now,
            last_active_at_ms: now,
            seen_tokens: response.state.seen_tokens,
        };
        let mut store = self.store.write().await;
        store
            .owners
            .insert(handle.session_id.clone(), handle.owner_id.clone());
        store.sessions.insert(handle.session_id.clone(), record);
        Ok(handle)
    }

    async fn continue_inner(
        &self,
        request: ContinueRequest,
    ) -> Result<ContinueResult, StateContractError> {
        if request.input.trim().is_empty() || request.token_budget == 0 {
            return Err(StateContractError::InvalidRequest(
                "input and token_budget must be positive".into(),
            ));
        }
        {
            let store = self.store.read().await;
            let owner = store
                .owners
                .get(&request.session_handle.session_id)
                .ok_or(StateContractError::StaleHandle)?;
            if owner != &request.owner_id {
                return Err(StateContractError::OwnerMismatch);
            }
            if !store
                .sessions
                .contains_key(&request.session_handle.session_id)
            {
                return Err(StateContractError::StaleHandle);
            }
        }
        let started = Instant::now();
        let input = format!("\n\nUser: {}\n\nAssistant:", request.input.trim());
        let response = self
            .client
            .post(format!("{}/v1/states/batch_continue", self.endpoint))
            .json(&json!({
                "owner_id":request.owner_id,
                "items":[{"state_id":request.session_handle.session_id,"input":input}],
                "stop":["\n\nUser:","\nUser:","\nSystem:","</s>"],
                "max_tokens":request.token_budget,
            }))
            .send()
            .await
            .map_err(|error| StateContractError::Provider(error.to_string()))?;
        let mut response = Self::response_json::<ContinueResponse>(response).await?;
        if response.results.len() != 1 {
            return Err(StateContractError::Provider(
                "RWKV continuation returned an invalid row count".into(),
            ));
        }
        let row = response.results.remove(0);
        if row.state_id != request.session_handle.session_id {
            return Err(StateContractError::Provider(
                "RWKV continuation changed state identity".into(),
            ));
        }
        let mut store = self.store.write().await;
        let session = store
            .sessions
            .get_mut(&row.state_id)
            .ok_or(StateContractError::StaleHandle)?;
        session.last_active_at_ms = Self::now_ms();
        session.seen_tokens = row.seen_tokens;
        Ok(ContinueResult {
            session_handle: session.handle.clone(),
            text: row.text,
            output_tokens: row.token_ids.len().try_into().unwrap_or(u32::MAX),
            seen_tokens: row.seen_tokens,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    async fn describe_inner(
        &self,
        owner_id: String,
        handle: SessionHandle,
    ) -> Result<SessionDescription, StateContractError> {
        let store = self.store.read().await;
        let owner = store
            .owners
            .get(&handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        if owner != &owner_id {
            return Err(StateContractError::OwnerMismatch);
        }
        let session = store
            .sessions
            .get(&handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        Ok(SessionDescription {
            session_handle: session.handle.clone(),
            durable_session_ref: session.durable_session_ref.clone(),
            placement: Placement::Gpu,
            created_at_ms: session.created_at_ms,
            last_active_at_ms: session.last_active_at_ms,
            state_bytes: None,
            seen_tokens: session.seen_tokens,
        })
    }

    async fn release_inner(
        &self,
        owner_id: String,
        handle: SessionHandle,
    ) -> Result<ReleaseOutcome, StateContractError> {
        {
            let store = self.store.read().await;
            let owner = store
                .owners
                .get(&handle.session_id)
                .ok_or(StateContractError::StaleHandle)?;
            if owner != &owner_id {
                return Err(StateContractError::OwnerMismatch);
            }
            if !store.sessions.contains_key(&handle.session_id) {
                return Ok(ReleaseOutcome { released: false });
            }
        }
        let response = self
            .client
            .post(format!("{}/v1/states/release", self.endpoint))
            .json(&json!({"owner_id":owner_id,"state_ids":[handle.session_id]}))
            .send()
            .await
            .map_err(|error| StateContractError::Provider(error.to_string()))?;
        let value = Self::response_json::<Value>(response).await?;
        let released = value
            .get("released")
            .and_then(Value::as_u64)
            .map(|count| count > 0)
            .or_else(|| {
                value
                    .get("released")
                    .and_then(Value::as_array)
                    .map(|rows| !rows.is_empty())
            })
            .unwrap_or(false);
        if released {
            self.store.write().await.sessions.remove(&handle.session_id);
        }
        Ok(ReleaseOutcome { released })
    }
}

impl StatefulInferenceProvider for RwkvHttpProvider {
    fn create(&self, request: CreateRequest) -> ProviderFuture<'_, SessionHandle> {
        Box::pin(self.create_inner(request))
    }

    fn continue_session(&self, request: ContinueRequest) -> ProviderFuture<'_, ContinueResult> {
        Box::pin(self.continue_inner(request))
    }

    fn snapshot(&self, _request: SnapshotRequest) -> ProviderFuture<'_, CheckpointRef> {
        Box::pin(async {
            Err(StateContractError::Unsupported(
                "RWKV HTTP snapshot endpoint is not implemented".into(),
            ))
        })
    }

    fn restore(&self, _request: RestoreRequest) -> ProviderFuture<'_, SessionHandle> {
        Box::pin(async {
            Err(StateContractError::Unsupported(
                "RWKV HTTP restore endpoint is not implemented".into(),
            ))
        })
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
