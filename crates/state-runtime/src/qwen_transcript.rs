use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use reqwest::Client;
use serde::{Deserialize, Serialize};
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
    messages: Vec<ChatMessage>,
    created_at_ms: u64,
    last_active_at_ms: u64,
    seen_tokens: u64,
}

#[derive(Default)]
struct Store {
    sessions: HashMap<String, SessionRecord>,
    owners: HashMap<String, String>,
    durable_transcripts: HashMap<String, Vec<ChatMessage>>,
}

pub struct QwenTranscriptProvider {
    client: Client,
    endpoint: String,
    served_model: String,
    model_ref: ModelRef,
    store: RwLock<Store>,
    next_id: AtomicU64,
}

#[derive(Serialize)]
struct CompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    max_tokens: u32,
    temperature: f32,
    stream: bool,
}

#[derive(Clone, Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct CompletionChoice {
    message: ChatMessage,
}

#[derive(Default, Deserialize)]
struct CompletionUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u32,
}

#[derive(Deserialize)]
struct CompletionResponse {
    choices: Vec<CompletionChoice>,
    #[serde(default)]
    usage: CompletionUsage,
}

impl QwenTranscriptProvider {
    pub fn new(
        endpoint: impl Into<String>,
        served_model: impl Into<String>,
        model_ref: ModelRef,
    ) -> Result<Self, StateContractError> {
        model_ref.validate()?;
        Ok(Self {
            client: Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            served_model: served_model.into(),
            model_ref,
            store: RwLock::new(Store::default()),
            next_id: AtomicU64::new(1),
        })
    }

    pub async fn allocated(&self) -> usize {
        self.store.read().await.sessions.len()
    }

    pub async fn durable_bytes(&self) -> u64 {
        self.store
            .read()
            .await
            .durable_transcripts
            .values()
            .flat_map(|messages| messages.iter())
            .map(|message| (message.role.len() + message.content.len()) as u64)
            .sum()
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }

    fn validate_owner(expected: &str, actual: &str) -> Result<(), StateContractError> {
        if expected != actual {
            return Err(StateContractError::OwnerMismatch);
        }
        Ok(())
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
        let session_id = format!(
            "qwen-replay-{:016x}",
            self.next_id.fetch_add(1, Ordering::Relaxed)
        );
        let handle = SessionHandle {
            contract_version: CONTRACT_VERSION.into(),
            session_id: session_id.clone(),
            owner_id: request.owner_id.clone(),
            provider_mode: ProviderMode::QwenTranscriptReprefill,
            model_ref: request.model_ref,
        };
        let now = Self::now_ms();
        let mut store = self.store.write().await;
        let messages = store
            .durable_transcripts
            .get(&request.durable_session_ref)
            .cloned()
            .unwrap_or_default();
        store.owners.insert(session_id.clone(), request.owner_id);
        store.sessions.insert(
            session_id,
            SessionRecord {
                handle: handle.clone(),
                durable_session_ref: request.durable_session_ref,
                messages,
                created_at_ms: now,
                last_active_at_ms: now,
                seen_tokens: 0,
            },
        );
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
        let (mut messages, durable_ref) = {
            let store = self.store.read().await;
            let owner = store
                .owners
                .get(&request.session_handle.session_id)
                .ok_or(StateContractError::StaleHandle)?;
            Self::validate_owner(owner, &request.owner_id)?;
            let session = store
                .sessions
                .get(&request.session_handle.session_id)
                .ok_or(StateContractError::StaleHandle)?;
            (
                session.messages.clone(),
                session.durable_session_ref.clone(),
            )
        };
        messages.push(ChatMessage {
            role: "user".into(),
            content: request.input,
        });
        let started = Instant::now();
        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.endpoint))
            .json(&CompletionRequest {
                model: &self.served_model,
                messages: &messages,
                max_tokens: request.token_budget,
                temperature: 0.0,
                stream: false,
            })
            .send()
            .await
            .map_err(|error| StateContractError::Provider(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(StateContractError::Provider(format!(
                "HTTP {status}: {}",
                body.chars().take(500).collect::<String>()
            )));
        }
        let response = response
            .json::<CompletionResponse>()
            .await
            .map_err(|error| StateContractError::Provider(error.to_string()))?;
        let text = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| StateContractError::Provider("completion returned no choices".into()))?
            .message
            .content;
        messages.push(ChatMessage {
            role: "assistant".into(),
            content: text.clone(),
        });
        let mut store = self.store.write().await;
        let owner = store
            .owners
            .get(&request.session_handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        Self::validate_owner(owner, &request.owner_id)?;
        let session = store
            .sessions
            .get_mut(&request.session_handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        session.messages.clone_from(&messages);
        session.last_active_at_ms = Self::now_ms();
        session.seen_tokens = response
            .usage
            .prompt_tokens
            .saturating_add(u64::from(response.usage.completion_tokens));
        let seen_tokens = session.seen_tokens;
        let handle = session.handle.clone();
        store.durable_transcripts.insert(durable_ref, messages);
        Ok(ContinueResult {
            session_handle: handle,
            output_tokens: response.usage.completion_tokens,
            text,
            seen_tokens,
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
        Self::validate_owner(owner, &owner_id)?;
        let session = store
            .sessions
            .get(&handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        Ok(SessionDescription {
            session_handle: session.handle.clone(),
            durable_session_ref: session.durable_session_ref.clone(),
            placement: Placement::Dropped,
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
        let mut store = self.store.write().await;
        let owner = store
            .owners
            .get(&handle.session_id)
            .ok_or(StateContractError::StaleHandle)?;
        Self::validate_owner(owner, &owner_id)?;
        Ok(ReleaseOutcome {
            released: store.sessions.remove(&handle.session_id).is_some(),
        })
    }
}

impl StatefulInferenceProvider for QwenTranscriptProvider {
    fn create(&self, request: CreateRequest) -> ProviderFuture<'_, SessionHandle> {
        Box::pin(self.create_inner(request))
    }

    fn continue_session(&self, request: ContinueRequest) -> ProviderFuture<'_, ContinueResult> {
        Box::pin(self.continue_inner(request))
    }

    fn snapshot(&self, _request: SnapshotRequest) -> ProviderFuture<'_, CheckpointRef> {
        Box::pin(async {
            Err(StateContractError::Unsupported(
                "transcript replay is not a KV snapshot".into(),
            ))
        })
    }

    fn restore(&self, _request: RestoreRequest) -> ProviderFuture<'_, SessionHandle> {
        Box::pin(async {
            Err(StateContractError::Unsupported(
                "transcript replay is not a KV restore".into(),
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
