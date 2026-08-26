//! Out-of-process StatePool Cloud Plugin.
//!
//! The local profile implements protocol negotiation, a dynamic Worker
//! directory, explainable placement, single-writer Lease/fencing semantics,
//! immutable LocalFS/S3 State snapshots, optional PostgreSQL metadata and
//! bounded in-memory FinOps counters. Local backends remain the default.

pub mod metadata;
pub mod state_store;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use rwkv_statepool_plugin_api::{
    AcquireLeaseRequest, EXECUTION_PLAN_CONTRACT_VERSION, ExecutionPlan, HandshakeRequest,
    HandshakeResponse, Lease, Money, PLUGIN_CONTRACT_VERSION, PlanRequest, PrivacyClass,
    RESTORE_RESPONSE_CONTRACT_VERSION, ReleaseLeaseRequest, RenewLeaseRequest, RestoreStateRequest,
    RestoreStateResponse, STATE_REFERENCE_CONTRACT_VERSION, SnapshotStateRequest, StateReference,
    USAGE_RECORD_CONTRACT_VERSION, UsageRecord, WorkerCapability, WorkerLifecycle, WorkerZone,
};
use serde_json::{Value, json};
use tokio::sync::RwLock;

use metadata::{InMemoryMetadataStore, MetadataError, MetadataErrorKind, MetadataStore};
use state_store::{LocalFsStateStore, StateStore, sha256_checksum};

const CAPABILITIES: &[&str] = &[
    "placement",
    "worker_registry",
    "leases",
    "state_lifecycle",
    "drain",
    "finops",
];

#[derive(Clone, Debug)]
pub struct PluginConfig {
    pub plugin_version: String,
    pub worker_ttl: Duration,
    pub lease_max_ttl: Duration,
    pub max_state_bytes: u64,
    pub state_dir: PathBuf,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            plugin_version: env!("CARGO_PKG_VERSION").into(),
            worker_ttl: Duration::from_secs(30),
            lease_max_ttl: Duration::from_secs(120),
            max_state_bytes: 512 * 1024 * 1024,
            state_dir: PathBuf::from("var/statepool/states"),
        }
    }
}

#[derive(Default)]
struct Metrics {
    handshakes: AtomicU64,
    plan_requests: AtomicU64,
    local_plans: AtomicU64,
    remote_plans: AtomicU64,
    rejected_plans: AtomicU64,
    worker_registrations: AtomicU64,
    usage_records: AtomicU64,
    gpu_millis: AtomicU64,
    prefill_tokens_avoided: AtomicU64,
    state_bytes_read: AtomicU64,
    state_bytes_written: AtomicU64,
    leases_acquired: AtomicU64,
    lease_conflicts: AtomicU64,
    snapshots_committed: AtomicU64,
    restores_completed: AtomicU64,
    pending_requests: AtomicU64,
    estimated_decode_millis: AtomicU64,
    hot_state_hits: AtomicU64,
    warm_state_hits: AtomicU64,
    cold_state_hits: AtomicU64,
    transcript_reprefills: AtomicU64,
}

#[derive(Clone)]
pub struct PluginState {
    config: Arc<PluginConfig>,
    workers: Arc<RwLock<HashMap<String, WorkerCapability>>>,
    drain_deadlines: Arc<RwLock<HashMap<String, u64>>>,
    usage: Arc<RwLock<Vec<UsageRecord>>>,
    estimated_cost_micros: Arc<RwLock<HashMap<String, u64>>>,
    metrics: Arc<Metrics>,
    decisions: Arc<AtomicU64>,
    metadata: Arc<dyn MetadataStore>,
    state_store: Arc<dyn StateStore>,
    metadata_backend: Arc<str>,
    object_store_backend: Arc<str>,
}

impl PluginState {
    pub fn new(config: PluginConfig) -> Result<Self, String> {
        if config.plugin_version.trim().is_empty()
            || config.worker_ttl.is_zero()
            || config.lease_max_ttl < Duration::from_secs(1)
            || config.max_state_bytes == 0
        {
            return Err(
                "plugin version, Worker/Lease TTL and max State bytes must be positive".into(),
            );
        }
        let state_store =
            LocalFsStateStore::new(config.state_dir.clone()).map_err(|error| error.0)?;
        Self::with_backends(
            config,
            Arc::new(InMemoryMetadataStore::default()),
            Arc::new(state_store),
            "in_memory_lease_cas",
            "localfs",
        )
    }

    pub fn with_backends(
        config: PluginConfig,
        metadata: Arc<dyn MetadataStore>,
        state_store: Arc<dyn StateStore>,
        metadata_backend: impl Into<Arc<str>>,
        object_store_backend: impl Into<Arc<str>>,
    ) -> Result<Self, String> {
        if config.plugin_version.trim().is_empty()
            || config.worker_ttl.is_zero()
            || config.lease_max_ttl < Duration::from_secs(1)
            || config.max_state_bytes == 0
        {
            return Err(
                "plugin version, Worker/Lease TTL and max State bytes must be positive".into(),
            );
        }
        Ok(Self {
            config: Arc::new(config),
            workers: Arc::new(RwLock::new(HashMap::new())),
            drain_deadlines: Arc::new(RwLock::new(HashMap::new())),
            usage: Arc::new(RwLock::new(Vec::new())),
            estimated_cost_micros: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(Metrics::default()),
            decisions: Arc::new(AtomicU64::new(1)),
            metadata,
            state_store,
            metadata_backend: metadata_backend.into(),
            object_store_backend: object_store_backend.into(),
        })
    }

    fn decision_id(&self) -> String {
        format!(
            "decision-{:016x}",
            self.decisions.fetch_add(1, Ordering::Relaxed)
        )
    }
}

pub fn router(state: PluginState) -> Router {
    Router::new()
        .route("/live", get(live))
        .route("/plugin/v1/health", get(health))
        .route("/plugin/v1/handshake", post(handshake))
        .route("/plugin/v1/plan", post(plan))
        .route("/plugin/v1/usage", post(record_usage))
        .route("/plugin/v1/leases/acquire", post(acquire_lease))
        .route("/plugin/v1/leases/renew", post(renew_lease))
        .route("/plugin/v1/leases/release", post(release_lease))
        .route("/plugin/v1/states/snapshot", post(snapshot_state))
        .route("/plugin/v1/states/restore", post(restore_state))
        .route("/plugin/v1/workers", get(list_workers))
        .route("/plugin/v1/workers/register", post(register_worker))
        .route(
            "/plugin/v1/workers/{worker_id}/heartbeat",
            post(heartbeat_worker),
        )
        .route("/plugin/v1/workers/{worker_id}/drain", post(drain_worker))
        .route("/metrics", get(metrics))
        .with_state(state)
}

async fn live() -> Json<Value> {
    Json(json!({
        "status":"alive",
        "contract_version":PLUGIN_CONTRACT_VERSION,
        "plugin":"statepool-cloud"
    }))
}

async fn health(State(state): State<PluginState>) -> Json<Value> {
    let workers = state.workers.read().await;
    let ready = workers
        .values()
        .filter(|worker| worker.lifecycle == WorkerLifecycle::Ready && !stale(worker, &state))
        .count();
    Json(json!({
        "contract_version":PLUGIN_CONTRACT_VERSION,
        "status":if ready > 0 {"ready"} else {"degraded"},
        "checked_at_ms":now_ms(),
        "dependencies":{
            "worker_registry":if ready > 0 {"ready"} else {"degraded"},
            "metadata":state.metadata_backend.as_ref(),
            "object_store":state.object_store_backend.as_ref()
        }
    }))
}

async fn handshake(
    State(state): State<PluginState>,
    Json(request): Json<HandshakeRequest>,
) -> Result<Json<HandshakeResponse>, PluginError> {
    if request.contract_version != PLUGIN_CONTRACT_VERSION
        || request.host != "rwkv-agent"
        || request.host_version.trim().is_empty()
    {
        return Err(PluginError::new(
            StatusCode::CONFLICT,
            "contract_version",
            "plugin handshake contract, host or host version mismatch",
            false,
        ));
    }
    let missing = request
        .required_capabilities
        .iter()
        .filter(|capability| !CAPABILITIES.contains(&capability.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(PluginError::new(
            StatusCode::CONFLICT,
            "capability_missing",
            format!("missing capabilities: {}", missing.join(",")),
            false,
        ));
    }
    state.metrics.handshakes.fetch_add(1, Ordering::Relaxed);
    Ok(Json(HandshakeResponse {
        contract_version: PLUGIN_CONTRACT_VERSION.into(),
        plugin: "statepool-cloud".into(),
        plugin_version: state.config.plugin_version.clone(),
        capabilities: CAPABILITIES.iter().map(|value| (*value).into()).collect(),
    }))
}

async fn acquire_lease(
    State(state): State<PluginState>,
    Json(request): Json<AcquireLeaseRequest>,
) -> Result<Json<Lease>, PluginError> {
    request.validate().map_err(invalid_contract)?;
    ensure_lease_ttl(&state, request.ttl_ms)?;
    match state.metadata.acquire(&request, now_ms()).await {
        Ok(lease) => {
            state
                .metrics
                .leases_acquired
                .fetch_add(1, Ordering::Relaxed);
            Ok(Json(lease))
        }
        Err(error) => {
            state
                .metrics
                .lease_conflicts
                .fetch_add(1, Ordering::Relaxed);
            Err(metadata_error(error))
        }
    }
}

async fn renew_lease(
    State(state): State<PluginState>,
    Json(request): Json<RenewLeaseRequest>,
) -> Result<Json<Lease>, PluginError> {
    request.validate().map_err(invalid_contract)?;
    ensure_lease_ttl(&state, request.ttl_ms)?;
    state
        .metadata
        .renew(&request, now_ms())
        .await
        .map(Json)
        .map_err(metadata_error)
}

async fn release_lease(
    State(state): State<PluginState>,
    Json(request): Json<ReleaseLeaseRequest>,
) -> Result<StatusCode, PluginError> {
    request.validate().map_err(invalid_contract)?;
    state
        .metadata
        .release(&request.lease, now_ms())
        .await
        .map_err(metadata_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn snapshot_state(
    State(state): State<PluginState>,
    Json(request): Json<SnapshotStateRequest>,
) -> Result<Json<StateReference>, PluginError> {
    request.validate().map_err(invalid_contract)?;
    state
        .metadata
        .assert_lease(&request.lease, true, now_ms())
        .await
        .map_err(metadata_error)?;

    let payload = BASE64.decode(&request.payload_base64).map_err(|error| {
        PluginError::new(
            StatusCode::BAD_REQUEST,
            "invalid_state_payload",
            format!("State payload is not canonical base64: {error}"),
            false,
        )
    })?;
    let size_bytes = u64::try_from(payload.len()).unwrap_or(u64::MAX);
    if size_bytes == 0 || size_bytes > state.config.max_state_bytes {
        return Err(PluginError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "state_payload_too_large",
            format!(
                "State payload must be between 1 and {} bytes",
                state.config.max_state_bytes
            ),
            false,
        ));
    }
    let checksum = sha256_checksum(&payload);
    if request
        .expected_checksum
        .as_ref()
        .is_some_and(|expected| expected != &checksum)
    {
        return Err(PluginError::new(
            StatusCode::CONFLICT,
            "state_checksum_mismatch",
            "Uploaded State does not match expected checksum",
            false,
        ));
    }

    let version = request.expected_state_version.saturating_add(1);
    let identity_hash = sha256_checksum(
        format!("{}\0{}", request.lease.owner_id, request.lease.session_id).as_bytes(),
    );
    let state_id = format!(
        "state-v{version}-f{}-{}",
        request.lease.fencing_token,
        &checksum[7..23]
    );
    let object_key = format!(
        "{}/v{version}-f{}-{}.state",
        &identity_hash[7..39],
        request.lease.fencing_token,
        &checksum[7..23]
    );
    let stored = state
        .state_store
        .put_immutable(&object_key, &payload)
        .await
        .map_err(state_store_error)?;
    let committed_at = now_ms();
    let state_ref = StateReference {
        contract_version: STATE_REFERENCE_CONTRACT_VERSION.into(),
        state_id,
        session_id: request.lease.session_id.clone(),
        owner_id: request.lease.owner_id.clone(),
        version,
        fencing_token: Some(request.lease.fencing_token),
        provider_mode: request.provider_mode,
        model_ref: request.model_ref,
        placement: request.target_tier,
        worker_id: None,
        object_uri: Some(stored.uri.clone()),
        checksum: stored.checksum,
        size_bytes: stored.size_bytes,
        atomic: true,
        created_at_ms: committed_at,
        last_active_at_ms: committed_at,
        encryption: None,
    };
    if let Err(error) = state
        .metadata
        .commit_state(
            &request.lease,
            request.expected_state_version,
            state_ref.clone(),
            committed_at,
        )
        .await
    {
        let _ = state.state_store.delete(&stored.uri).await;
        return Err(metadata_error(error));
    }
    state
        .metrics
        .snapshots_committed
        .fetch_add(1, Ordering::Relaxed);
    state
        .metrics
        .state_bytes_written
        .fetch_add(size_bytes, Ordering::Relaxed);
    Ok(Json(state_ref))
}

async fn restore_state(
    State(state): State<PluginState>,
    Json(request): Json<RestoreStateRequest>,
) -> Result<Json<RestoreStateResponse>, PluginError> {
    request.validate().map_err(invalid_contract)?;
    state
        .metadata
        .assert_lease(&request.lease, true, now_ms())
        .await
        .map_err(metadata_error)?;
    let current = state
        .metadata
        .current_state(&request.state_ref.session_id, &request.state_ref.owner_id)
        .await
        .map_err(metadata_error)?;
    if current != request.state_ref
        || request.lease.session_id != current.session_id
        || request.lease.owner_id != current.owner_id
    {
        return Err(PluginError::new(
            StatusCode::CONFLICT,
            "state_version_conflict",
            "Restore reference is not the current committed State",
            false,
        ));
    }
    let uri = current.object_uri.as_deref().ok_or_else(|| {
        PluginError::new(
            StatusCode::CONFLICT,
            "state_not_persisted",
            "State reference has no persisted object URI",
            false,
        )
    })?;
    let payload = state
        .state_store
        .get(uri)
        .await
        .map_err(state_store_error)?;
    if sha256_checksum(&payload) != current.checksum
        || u64::try_from(payload.len()).unwrap_or(u64::MAX) != current.size_bytes
    {
        return Err(PluginError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "state_integrity_failure",
            "Persisted State checksum or size does not match metadata",
            false,
        ));
    }
    state
        .metrics
        .restores_completed
        .fetch_add(1, Ordering::Relaxed);
    state
        .metrics
        .state_bytes_read
        .fetch_add(current.size_bytes, Ordering::Relaxed);
    Ok(Json(RestoreStateResponse {
        contract_version: RESTORE_RESPONSE_CONTRACT_VERSION.into(),
        state_ref: current,
        payload_base64: BASE64.encode(payload),
    }))
}

fn ensure_lease_ttl(state: &PluginState, ttl_ms: u64) -> Result<(), PluginError> {
    if u128::from(ttl_ms) > state.config.lease_max_ttl.as_millis() {
        return Err(PluginError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!(
                "ttl_ms exceeds configured maximum of {}",
                state.config.lease_max_ttl.as_millis()
            ),
            false,
        ));
    }
    Ok(())
}

fn invalid_contract(error: rwkv_statepool_plugin_api::ContractError) -> PluginError {
    PluginError::new(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        error.to_string(),
        false,
    )
}

fn metadata_error(error: MetadataError) -> PluginError {
    let status = match error.kind {
        MetadataErrorKind::OwnerMismatch => StatusCode::FORBIDDEN,
        MetadataErrorKind::NotFound => StatusCode::NOT_FOUND,
        MetadataErrorKind::VersionConflict
        | MetadataErrorKind::LeaseHeld
        | MetadataErrorKind::LeaseExpired
        | MetadataErrorKind::StaleFence
        | MetadataErrorKind::StateConflict => StatusCode::CONFLICT,
        MetadataErrorKind::BackendFailure => StatusCode::SERVICE_UNAVAILABLE,
    };
    PluginError::new(status, error.code(), error.message, false)
}

fn state_store_error(error: state_store::StateStoreError) -> PluginError {
    PluginError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "state_store_failure",
        error.0,
        true,
    )
}

async fn register_worker(
    State(state): State<PluginState>,
    Json(worker): Json<WorkerCapability>,
) -> Result<Json<Value>, PluginError> {
    worker.validate().map_err(|error| {
        PluginError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            error.to_string(),
            false,
        )
    })?;
    let worker_id = worker.worker_id.clone();
    state
        .workers
        .write()
        .await
        .insert(worker_id.clone(), worker);
    // Explicit registration starts a new Worker incarnation. A stale drain
    // intent from a previous Pod with the same stable identity must not leak
    // into the replacement after the orchestrator has recreated it.
    state.drain_deadlines.write().await.remove(&worker_id);
    state
        .metrics
        .worker_registrations
        .fetch_add(1, Ordering::Relaxed);
    // Registration resolves the transient scale-from-zero demand signal.
    // Ongoing load remains represented by Worker heartbeat queue depth.
    state.metrics.pending_requests.store(0, Ordering::Relaxed);
    state
        .metrics
        .estimated_decode_millis
        .store(0, Ordering::Relaxed);
    Ok(Json(json!({"status":"ok","worker_id":worker_id})))
}

async fn heartbeat_worker(
    Path(worker_id): Path<String>,
    State(state): State<PluginState>,
    Json(mut worker): Json<WorkerCapability>,
) -> Result<Json<Value>, PluginError> {
    if worker_id != worker.worker_id {
        return Err(PluginError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "path and body Worker identities differ",
            false,
        ));
    }
    worker.validate().map_err(|error| {
        PluginError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            error.to_string(),
            false,
        )
    })?;
    let deadline = state.drain_deadlines.read().await.get(&worker_id).copied();
    if deadline.is_some() {
        // The control-plane drain decision is sticky. A heartbeat assembled
        // just before the drain request cannot accidentally reopen admission.
        worker.lifecycle = WorkerLifecycle::Draining;
    }
    let mut workers = state.workers.write().await;
    if !workers.contains_key(&worker_id) {
        return Err(PluginError::new(
            StatusCode::NOT_FOUND,
            "invalid_request",
            "unknown Worker",
            false,
        ));
    }
    let drain_status = deadline.map(|deadline| worker_drain_status(&worker, deadline, now_ms()));
    workers.insert(worker_id.clone(), worker.clone());
    Ok(Json(json!({
        "status":"ok",
        "worker_id":worker_id,
        "lifecycle":worker.lifecycle,
        "drain_status":drain_status
    })))
}

async fn list_workers(State(state): State<PluginState>) -> Json<Value> {
    let mut workers = state
        .workers
        .read()
        .await
        .values()
        .cloned()
        .collect::<Vec<_>>();
    workers.sort_by(|left, right| left.worker_id.cmp(&right.worker_id));
    Json(json!({"workers":workers,"count":workers.len()}))
}

async fn drain_worker(
    Path(worker_id): Path<String>,
    State(state): State<PluginState>,
    Json(request): Json<Value>,
) -> Result<Json<Value>, PluginError> {
    if request.get("contract_version").and_then(Value::as_str) != Some("statepool-drain-request.v1")
        || request
            .get("deadline_ms")
            .and_then(Value::as_u64)
            .is_none_or(|value| value == 0)
    {
        return Err(PluginError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "invalid drain request",
            false,
        ));
    }
    let deadline = request["deadline_ms"]
        .as_u64()
        .expect("validated drain deadline");
    if !state.workers.read().await.contains_key(&worker_id) {
        return Err(PluginError::new(
            StatusCode::NOT_FOUND,
            "invalid_request",
            "unknown Worker",
            false,
        ));
    }
    state
        .drain_deadlines
        .write()
        .await
        .insert(worker_id.clone(), deadline);
    let mut workers = state.workers.write().await;
    let worker = workers
        .get_mut(&worker_id)
        .expect("Worker cannot disappear from the in-memory registry");
    worker.lifecycle = WorkerLifecycle::Draining;
    Ok(Json(worker_drain_status(worker, deadline, now_ms())))
}

fn worker_drain_status(worker: &WorkerCapability, deadline_ms: u64, checked_at_ms: u64) -> Value {
    let unpersisted_states = worker
        .capacity
        .unpersisted_state_slots
        .unwrap_or(worker.capacity.state_slots);
    let active_requests = worker
        .capacity
        .running_requests
        .saturating_add(worker.capacity.queue_depth);
    let safe = active_requests == 0 && unpersisted_states == 0;
    json!({
        "contract_version":"statepool-drain-status.v1",
        "worker_id":worker.worker_id,
        "status":if safe {
            "safe_to_stop"
        } else if checked_at_ms >= deadline_ms {
            "deadline_exceeded"
        } else {
            "draining"
        },
        "active_requests":active_requests,
        "unpersisted_states":unpersisted_states
    })
}

async fn plan(
    State(state): State<PluginState>,
    Json(request): Json<PlanRequest>,
) -> Result<Json<ExecutionPlan>, PluginError> {
    request.validate().map_err(|error| {
        PluginError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            error.to_string(),
            false,
        )
    })?;
    state.metrics.plan_requests.fetch_add(1, Ordering::Relaxed);

    let workers = state.workers.read().await;
    let mut candidates = workers
        .values()
        .filter(|worker| !stale(worker, &state) && worker.supports(&request.model_ref))
        .filter(|worker| privacy_allows(request.privacy, &worker.zone))
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        if request.privacy == PrivacyClass::CloudAllowed {
            let _ = state.metrics.pending_requests.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |value| Some(value.saturating_add(1).min(10_000)),
            );
            let estimated_decode_ms = request.estimated_output_tokens.saturating_mul(50);
            let _ = state.metrics.estimated_decode_millis.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |value| Some(value.saturating_add(estimated_decode_ms).min(86_400_000)),
            );
        }
        state.metrics.rejected_plans.fetch_add(1, Ordering::Relaxed);
        return Ok(Json(ExecutionPlan {
            contract_version: EXECUTION_PLAN_CONTRACT_VERSION.into(),
            decision_id: state.decision_id(),
            request_id: request.request_id,
            mode: "reject".into(),
            worker_id: None,
            endpoint: None,
            state_action: if request.state_ref.is_some() {
                "transcript_reprefill".into()
            } else {
                "none".into()
            },
            reason_code: "no_compatible_worker".into(),
            lease_required: false,
            estimated_queue_ms: None,
            estimated_restore_ms: None,
            estimated_cost: None,
            fallback: "local".into(),
        }));
    }

    candidates.sort_by(|left, right| {
        score(left, &request)
            .total_cmp(&score(right, &request))
            .then_with(|| left.worker_id.cmp(&right.worker_id))
    });
    let worker = candidates[0];
    let _ = state.metrics.pending_requests.fetch_update(
        Ordering::Relaxed,
        Ordering::Relaxed,
        |value| Some(value.saturating_sub(1)),
    );
    let estimated_decode_ms = request.estimated_output_tokens.saturating_mul(50);
    let _ = state.metrics.estimated_decode_millis.fetch_update(
        Ordering::Relaxed,
        Ordering::Relaxed,
        |value| Some(value.saturating_sub(estimated_decode_ms)),
    );
    let (state_action, restore_ms, affinity) = state_action(worker, &request);
    match state_action.as_str() {
        "reuse_hot" => state.metrics.hot_state_hits.fetch_add(1, Ordering::Relaxed),
        "restore_warm" => state
            .metrics
            .warm_state_hits
            .fetch_add(1, Ordering::Relaxed),
        "restore_cold" => state
            .metrics
            .cold_state_hits
            .fetch_add(1, Ordering::Relaxed),
        "transcript_reprefill" => state
            .metrics
            .transcript_reprefills
            .fetch_add(1, Ordering::Relaxed),
        _ => 0,
    };
    let estimated_cost = estimate_cost(worker, &request);
    if request
        .max_cost
        .as_ref()
        .zip(estimated_cost.as_ref())
        .is_some_and(|(limit, cost)| limit.currency == cost.currency && cost.amount > limit.amount)
    {
        state.metrics.rejected_plans.fetch_add(1, Ordering::Relaxed);
        return Ok(Json(ExecutionPlan {
            contract_version: EXECUTION_PLAN_CONTRACT_VERSION.into(),
            decision_id: state.decision_id(),
            request_id: request.request_id,
            mode: "reject".into(),
            worker_id: None,
            endpoint: None,
            state_action: "none".into(),
            reason_code: "cost_limit".into(),
            lease_required: false,
            estimated_queue_ms: None,
            estimated_restore_ms: None,
            estimated_cost,
            fallback: "local".into(),
        }));
    }

    let mode = if worker.zone == WorkerZone::Local {
        state.metrics.local_plans.fetch_add(1, Ordering::Relaxed);
        "local"
    } else {
        state.metrics.remote_plans.fetch_add(1, Ordering::Relaxed);
        "remote"
    };
    let reason_code = if affinity {
        "state_affinity"
    } else if worker.zone == WorkerZone::Local {
        "local_within_slo"
    } else {
        "cloud_capacity"
    };
    Ok(Json(ExecutionPlan {
        contract_version: EXECUTION_PLAN_CONTRACT_VERSION.into(),
        decision_id: state.decision_id(),
        request_id: request.request_id,
        mode: mode.into(),
        worker_id: Some(worker.worker_id.clone()),
        endpoint: Some(worker.endpoint.clone()),
        state_action,
        reason_code: reason_code.into(),
        lease_required: request.state_ref.is_some(),
        estimated_queue_ms: Some(worker.capacity.queue_depth as f64 * 1000.0),
        estimated_restore_ms: restore_ms,
        estimated_cost,
        fallback: "local".into(),
    }))
}

async fn record_usage(
    State(state): State<PluginState>,
    Json(record): Json<UsageRecord>,
) -> Result<StatusCode, PluginError> {
    if record.contract_version != USAGE_RECORD_CONTRACT_VERSION
        || record.record_id.trim().is_empty()
        || record.request_id.trim().is_empty()
        || record.finished_at_ms < record.started_at_ms
    {
        return Err(PluginError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "invalid usage record",
            false,
        ));
    }
    state.metrics.usage_records.fetch_add(1, Ordering::Relaxed);
    state.metrics.gpu_millis.fetch_add(
        (record.metrics.gpu_seconds.max(0.0) * 1000.0) as u64,
        Ordering::Relaxed,
    );
    state
        .metrics
        .prefill_tokens_avoided
        .fetch_add(record.metrics.prefill_tokens_avoided, Ordering::Relaxed);
    state
        .metrics
        .state_bytes_read
        .fetch_add(record.metrics.state_bytes_read, Ordering::Relaxed);
    state
        .metrics
        .state_bytes_written
        .fetch_add(record.metrics.state_bytes_written, Ordering::Relaxed);
    if let Some(cost) = &record.metrics.estimated_cost
        && cost.currency.len() == 3
        && cost
            .currency
            .chars()
            .all(|value| value.is_ascii_uppercase())
        && cost.amount.is_finite()
        && cost.amount >= 0.0
    {
        let micros = (cost.amount * 1_000_000.0).min(u64::MAX as f64) as u64;
        let mut totals = state.estimated_cost_micros.write().await;
        let total = totals.entry(cost.currency.clone()).or_default();
        *total = total.saturating_add(micros);
    }
    let mut usage = state.usage.write().await;
    if usage.len() >= 10_000 {
        usage.remove(0);
    }
    usage.push(record);
    Ok(StatusCode::ACCEPTED)
}

async fn metrics(State(state): State<PluginState>) -> Response {
    let metrics = &state.metrics;
    let workers = state.workers.read().await;
    let ready_workers = workers
        .values()
        .filter(|worker| worker.lifecycle == WorkerLifecycle::Ready && !stale(worker, &state))
        .count();
    let mut body = format!(
        concat!(
            "# TYPE statepool_ready_workers gauge\n",
            "statepool_ready_workers {}\n",
            "# TYPE statepool_plan_requests_total counter\n",
            "statepool_plan_requests_total {}\n",
            "statepool_local_plans_total {}\n",
            "statepool_remote_plans_total {}\n",
            "statepool_rejected_plans_total {}\n",
            "statepool_worker_registrations_total {}\n",
            "statepool_usage_records_total {}\n",
            "statepool_gpu_seconds_total {}\n",
            "statepool_prefill_tokens_avoided_total {}\n",
            "statepool_state_bytes_read_total {}\n",
            "statepool_state_bytes_written_total {}\n",
            "statepool_leases_acquired_total {}\n",
            "statepool_lease_conflicts_total {}\n",
            "statepool_snapshots_committed_total {}\n",
            "statepool_restores_completed_total {}\n",
            "# TYPE statepool_pending_requests gauge\n",
            "statepool_pending_requests {}\n",
            "# TYPE statepool_estimated_decode_seconds gauge\n",
            "statepool_estimated_decode_seconds {}\n",
            "statepool_hot_state_hits_total {}\n",
            "statepool_warm_state_hits_total {}\n",
            "statepool_cold_state_hits_total {}\n",
            "statepool_transcript_reprefills_total {}\n"
        ),
        ready_workers,
        metrics.plan_requests.load(Ordering::Relaxed),
        metrics.local_plans.load(Ordering::Relaxed),
        metrics.remote_plans.load(Ordering::Relaxed),
        metrics.rejected_plans.load(Ordering::Relaxed),
        metrics.worker_registrations.load(Ordering::Relaxed),
        metrics.usage_records.load(Ordering::Relaxed),
        metrics.gpu_millis.load(Ordering::Relaxed) as f64 / 1000.0,
        metrics.prefill_tokens_avoided.load(Ordering::Relaxed),
        metrics.state_bytes_read.load(Ordering::Relaxed),
        metrics.state_bytes_written.load(Ordering::Relaxed),
        metrics.leases_acquired.load(Ordering::Relaxed),
        metrics.lease_conflicts.load(Ordering::Relaxed),
        metrics.snapshots_committed.load(Ordering::Relaxed),
        metrics.restores_completed.load(Ordering::Relaxed),
        metrics.pending_requests.load(Ordering::Relaxed),
        metrics.estimated_decode_millis.load(Ordering::Relaxed) as f64 / 1000.0,
        metrics.hot_state_hits.load(Ordering::Relaxed),
        metrics.warm_state_hits.load(Ordering::Relaxed),
        metrics.cold_state_hits.load(Ordering::Relaxed),
        metrics.transcript_reprefills.load(Ordering::Relaxed),
    );
    body.push_str("# TYPE statepool_estimated_cost_total counter\n");
    let costs = state.estimated_cost_micros.read().await;
    let mut currencies = costs.iter().collect::<Vec<_>>();
    currencies.sort_by_key(|(currency, _)| *currency);
    for (currency, micros) in currencies {
        body.push_str(&format!(
            "statepool_estimated_cost_total{{currency=\"{currency}\"}} {}\n",
            *micros as f64 / 1_000_000.0
        ));
    }
    let mut response = body.into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4"),
    );
    response
}

fn privacy_allows(privacy: PrivacyClass, zone: &WorkerZone) -> bool {
    match privacy {
        PrivacyClass::LocalOnly => *zone == WorkerZone::Local,
        PrivacyClass::Hybrid => matches!(zone, WorkerZone::Local | WorkerZone::Edge),
        PrivacyClass::CloudAllowed => true,
    }
}

fn stale(worker: &WorkerCapability, state: &PluginState) -> bool {
    now_ms().saturating_sub(worker.reported_at_ms)
        > state
            .config
            .worker_ttl
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
}

fn score(worker: &WorkerCapability, request: &PlanRequest) -> f64 {
    let affinity = request
        .state_ref
        .as_ref()
        .and_then(|state| state.worker_id.as_deref())
        == Some(worker.worker_id.as_str());
    let affinity_penalty = if affinity { 0.0 } else { 10_000.0 };
    let zone_penalty = if request.preferred_zone.as_ref() == Some(&worker.zone) {
        0.0
    } else {
        2_000.0
    };
    let load = worker.capacity.queue_depth as f64 * 1_000.0
        + worker.capacity.running_requests as f64 * 250.0;
    let price = worker
        .price
        .as_ref()
        .map(|price| price.per_gpu_hour * 10.0)
        .unwrap_or(0.0);
    affinity_penalty + zone_penalty + load + price
}

fn state_action(worker: &WorkerCapability, request: &PlanRequest) -> (String, Option<f64>, bool) {
    let Some(state) = &request.state_ref else {
        return ("none".into(), Some(0.0), false);
    };
    if !state.exact_restore_compatible(&request.model_ref) {
        return ("transcript_reprefill".into(), None, false);
    }
    match state.placement {
        rwkv_statepool_plugin_api::StatePlacement::Hot
            if state.worker_id.as_deref() == Some(worker.worker_id.as_str()) =>
        {
            ("reuse_hot".into(), Some(0.0), true)
        }
        rwkv_statepool_plugin_api::StatePlacement::Warm => {
            let milliseconds = state.size_bytes as f64 / 10_000_000.0;
            ("restore_warm".into(), Some(milliseconds), false)
        }
        rwkv_statepool_plugin_api::StatePlacement::Cold => {
            let milliseconds = 50.0 + state.size_bytes as f64 / 100_000.0;
            ("restore_cold".into(), Some(milliseconds), false)
        }
        _ => ("transcript_reprefill".into(), None, false),
    }
}

fn estimate_cost(worker: &WorkerCapability, request: &PlanRequest) -> Option<Money> {
    let price = worker.price.as_ref()?;
    // This is explicitly an estimate. The calibrated production policy will
    // replace 50 ms/token with measured per-Worker decode telemetry.
    let estimated_gpu_seconds = request.estimated_output_tokens as f64 * 0.05;
    Some(Money {
        currency: price.currency.clone(),
        amount: estimated_gpu_seconds / 3600.0 * price.per_gpu_hour,
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[derive(Debug)]
struct PluginError {
    status: StatusCode,
    code: &'static str,
    message: String,
    retryable: bool,
}

impl PluginError {
    fn new(
        status: StatusCode,
        code: &'static str,
        message: impl Into<String>,
        retryable: bool,
    ) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            retryable,
        }
    }
}

impl IntoResponse for PluginError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "contract_version":PLUGIN_CONTRACT_VERSION,
                "status":"error",
                "error":{
                    "code":self.code,
                    "message":self.message,
                    "retryable":self.retryable,
                    "execution_state":"not_started"
                }
            })),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use rwkv_statepool_plugin_api::{
        ACQUIRE_LEASE_REQUEST_CONTRACT_VERSION, AcquireLeaseRequest, Lease, ModelRef,
        PLAN_REQUEST_CONTRACT_VERSION, RELEASE_LEASE_REQUEST_CONTRACT_VERSION,
        RESTORE_REQUEST_CONTRACT_VERSION, SNAPSHOT_REQUEST_CONTRACT_VERSION,
        STATE_REFERENCE_CONTRACT_VERSION, StateReference, WORKER_CAPABILITY_CONTRACT_VERSION,
        WorkerCapacity, WorkerDevice, WorkerPrice,
    };
    use tower::ServiceExt;

    fn model() -> ModelRef {
        ModelRef {
            model_id: "rwkv7".into(),
            revision: "revision".into(),
            tokenizer: "world".into(),
            state_abi: "rwkv7-state-v1".into(),
        }
    }

    fn worker(zone: WorkerZone) -> WorkerCapability {
        WorkerCapability {
            contract_version: WORKER_CAPABILITY_CONTRACT_VERSION.into(),
            worker_id: format!("{zone}-worker"),
            zone,
            endpoint: "http://worker.test".into(),
            lifecycle: WorkerLifecycle::Ready,
            models: vec![model()],
            device: WorkerDevice {
                vendor: "nvidia".into(),
                model: "v100".into(),
                runtime: "cuda".into(),
                memory_bytes: 32 * 1024 * 1024 * 1024,
            },
            capacity: WorkerCapacity {
                state_slots: 32,
                free_state_slots: 32,
                max_batch: 32,
                queue_depth: 0,
                running_requests: 0,
                unpersisted_state_slots: Some(0),
            },
            price: Some(WorkerPrice {
                currency: "CNY".into(),
                per_gpu_hour: 2.8,
            }),
            labels: Default::default(),
            reported_at_ms: now_ms(),
        }
    }

    fn test_config(name: &str) -> PluginConfig {
        PluginConfig {
            state_dir: std::env::temp_dir().join(format!(
                "rwkv-statepool-{name}-{}-{}",
                std::process::id(),
                now_ms()
            )),
            ..PluginConfig::default()
        }
    }

    async fn json_request(app: Router, path: &str, body: Value) -> (StatusCode, Value) {
        let response = app
            .oneshot(
                Request::post(path)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value = if body.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&body).unwrap()
        };
        (status, value)
    }

    #[tokio::test]
    async fn handshake_rejects_capabilities_the_plugin_does_not_claim() {
        let app = router(PluginState::new(PluginConfig::default()).unwrap());
        let (status, body) = json_request(
            app,
            "/plugin/v1/handshake",
            json!({
                "contract_version":PLUGIN_CONTRACT_VERSION,
                "host":"rwkv-agent",
                "host_version":"test",
                "required_capabilities":["remote_state"]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["error"]["code"], "capability_missing");
    }

    #[tokio::test]
    async fn exact_model_and_privacy_filter_select_the_cloud_worker() {
        let state = PluginState::new(PluginConfig::default()).unwrap();
        state
            .workers
            .write()
            .await
            .insert("cloud-worker".into(), worker(WorkerZone::Cloud));
        let app = router(state);
        let (status, body) = json_request(
            app,
            "/plugin/v1/plan",
            json!({
                "contract_version":PLAN_REQUEST_CONTRACT_VERSION,
                "request_id":"request",
                "session_id":"session",
                "owner_id":"owner",
                "model_ref":model(),
                "privacy":"cloud_allowed",
                "latency_slo_ms":5000,
                "estimated_input_tokens":128,
                "estimated_output_tokens":64
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["mode"], "remote");
        assert_eq!(body["worker_id"], "cloud-worker");
    }

    #[tokio::test]
    async fn local_only_never_selects_a_cloud_worker() {
        let state = PluginState::new(PluginConfig::default()).unwrap();
        state
            .workers
            .write()
            .await
            .insert("cloud-worker".into(), worker(WorkerZone::Cloud));
        let app = router(state);
        let (status, body) = json_request(
            app,
            "/plugin/v1/plan",
            json!({
                "contract_version":PLAN_REQUEST_CONTRACT_VERSION,
                "request_id":"request",
                "session_id":"session",
                "owner_id":"owner",
                "model_ref":model(),
                "privacy":"local_only",
                "latency_slo_ms":5000,
                "estimated_input_tokens":128,
                "estimated_output_tokens":64
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["mode"], "reject");
        assert_eq!(body["fallback"], "local");
    }

    #[tokio::test]
    async fn draining_worker_stops_receiving_new_plans() {
        let state = PluginState::new(PluginConfig::default()).unwrap();
        state
            .workers
            .write()
            .await
            .insert("cloud-worker".into(), worker(WorkerZone::Cloud));
        let app = router(state);
        let (status, body) = json_request(
            app.clone(),
            "/plugin/v1/workers/cloud-worker/drain",
            json!({"contract_version":"statepool-drain-request.v1","deadline_ms":now_ms() + 10_000}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "safe_to_stop");

        let (_, plan) = json_request(
            app,
            "/plugin/v1/plan",
            json!({
                "contract_version":PLAN_REQUEST_CONTRACT_VERSION,
                "request_id":"request",
                "session_id":"session",
                "owner_id":"owner",
                "model_ref":model(),
                "privacy":"cloud_allowed",
                "latency_slo_ms":5000,
                "estimated_input_tokens":128,
                "estimated_output_tokens":64
            }),
        )
        .await;
        assert_eq!(plan["mode"], "reject");
    }

    #[tokio::test]
    async fn drain_requires_an_explicit_zero_dirty_state_heartbeat() {
        let state = PluginState::new(PluginConfig::default()).unwrap();
        let mut unknown = worker(WorkerZone::Cloud);
        unknown.capacity.unpersisted_state_slots = None;
        state
            .workers
            .write()
            .await
            .insert("cloud-worker".into(), unknown);
        let app = router(state);
        let (status, body) = json_request(
            app.clone(),
            "/plugin/v1/workers/cloud-worker/drain",
            json!({"contract_version":"statepool-drain-request.v1","deadline_ms":now_ms() + 10_000}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "draining");
        assert_eq!(body["unpersisted_states"], 32);

        let mut stale_ready = worker(WorkerZone::Cloud);
        stale_ready.capacity.unpersisted_state_slots = None;
        stale_ready.reported_at_ms = now_ms();
        let (status, heartbeat) = json_request(
            app,
            "/plugin/v1/workers/cloud-worker/heartbeat",
            serde_json::to_value(stale_ready).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(heartbeat["lifecycle"], "draining");
        assert_eq!(heartbeat["drain_status"]["status"], "draining");
    }

    #[tokio::test]
    async fn drain_deadline_expires_loudly_when_work_or_dirty_state_remains() {
        let state = PluginState::new(PluginConfig::default()).unwrap();
        let mut busy = worker(WorkerZone::Cloud);
        busy.capacity.running_requests = 1;
        busy.capacity.unpersisted_state_slots = Some(2);
        state
            .workers
            .write()
            .await
            .insert("cloud-worker".into(), busy);
        let app = router(state);
        let (status, body) = json_request(
            app,
            "/plugin/v1/workers/cloud-worker/drain",
            json!({"contract_version":"statepool-drain-request.v1","deadline_ms":1}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "deadline_exceeded");
        assert_eq!(body["active_requests"], 1);
        assert_eq!(body["unpersisted_states"], 2);
    }

    #[tokio::test]
    async fn cloud_plan_miss_exposes_and_registration_clears_scale_signal() {
        let state = PluginState::new(PluginConfig::default()).unwrap();
        let app = router(state.clone());
        let (status, body) = json_request(
            app.clone(),
            "/plugin/v1/plan",
            json!({
                "contract_version":PLAN_REQUEST_CONTRACT_VERSION,
                "request_id":"scale-request",
                "session_id":"session",
                "owner_id":"owner",
                "model_ref":model(),
                "privacy":"cloud_allowed",
                "latency_slo_ms":5000,
                "estimated_input_tokens":128,
                "estimated_output_tokens":64
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["mode"], "reject");
        assert_eq!(state.metrics.pending_requests.load(Ordering::Relaxed), 1);
        assert_eq!(
            state
                .metrics
                .estimated_decode_millis
                .load(Ordering::Relaxed),
            3_200
        );

        let (status, _) = json_request(
            app,
            "/plugin/v1/workers/register",
            serde_json::to_value(worker(WorkerZone::Cloud)).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(state.metrics.pending_requests.load(Ordering::Relaxed), 0);
        assert_eq!(
            state
                .metrics
                .estimated_decode_millis
                .load(Ordering::Relaxed),
            0
        );
    }

    #[tokio::test]
    async fn one_writer_and_monotonic_fencing_survive_lease_expiry() {
        let metadata = InMemoryMetadataStore::default();
        let request = AcquireLeaseRequest {
            contract_version: ACQUIRE_LEASE_REQUEST_CONTRACT_VERSION.into(),
            session_id: "session".into(),
            owner_id: "owner".into(),
            holder_id: "holder-a".into(),
            expected_state_version: 0,
            ttl_ms: 1_000,
        };
        let first = metadata.acquire(&request, 10_000).await.unwrap();
        let conflict = metadata.acquire(&request, 10_999).await.unwrap_err();
        assert_eq!(conflict.kind, MetadataErrorKind::LeaseHeld);

        let mut second_request = request;
        second_request.holder_id = "holder-b".into();
        let second = metadata.acquire(&second_request, 11_000).await.unwrap();
        assert!(second.fencing_token > first.fencing_token);
        let stale = metadata
            .assert_lease(&first, true, 11_001)
            .await
            .unwrap_err();
        assert_eq!(stale.kind, MetadataErrorKind::StaleFence);
    }

    #[tokio::test]
    async fn snapshot_restore_round_trip_uses_cas_checksum_and_fencing() {
        let config = test_config("round-trip");
        let state_dir = config.state_dir.clone();
        let app = router(PluginState::new(config).unwrap());
        let acquire_body = json!({
            "contract_version":ACQUIRE_LEASE_REQUEST_CONTRACT_VERSION,
            "session_id":"session",
            "owner_id":"owner",
            "holder_id":"worker-a/request-a",
            "expected_state_version":0,
            "ttl_ms":30000
        });
        let (status, lease_value) =
            json_request(app.clone(), "/plugin/v1/leases/acquire", acquire_body).await;
        assert_eq!(status, StatusCode::OK);
        let first_lease: Lease = serde_json::from_value(lease_value.clone()).unwrap();

        let payload = b"recurrent-state-v1";
        let checksum = sha256_checksum(payload);
        let (status, state_value) = json_request(
            app.clone(),
            "/plugin/v1/states/snapshot",
            json!({
                "contract_version":SNAPSHOT_REQUEST_CONTRACT_VERSION,
                "provider_mode":"rwkv_recurrent",
                "model_ref":model(),
                "target_tier":"cold",
                "lease":lease_value,
                "expected_state_version":0,
                "payload_base64":BASE64.encode(payload),
                "expected_checksum":checksum
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{state_value}");
        let state_ref: StateReference = serde_json::from_value(state_value.clone()).unwrap();
        assert_eq!(state_ref.contract_version, STATE_REFERENCE_CONTRACT_VERSION);
        assert_eq!(state_ref.version, 1);
        assert_eq!(state_ref.fencing_token, Some(first_lease.fencing_token));

        let (status, _) = json_request(
            app.clone(),
            "/plugin/v1/leases/release",
            json!({
                "contract_version":RELEASE_LEASE_REQUEST_CONTRACT_VERSION,
                "lease":first_lease
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NO_CONTENT);

        let (status, restore_lease_value) = json_request(
            app.clone(),
            "/plugin/v1/leases/acquire",
            json!({
                "contract_version":ACQUIRE_LEASE_REQUEST_CONTRACT_VERSION,
                "session_id":"session",
                "owner_id":"owner",
                "holder_id":"worker-b/request-b",
                "expected_state_version":1,
                "ttl_ms":30000
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let restore_lease: Lease = serde_json::from_value(restore_lease_value.clone()).unwrap();
        assert!(restore_lease.fencing_token > state_ref.fencing_token.unwrap());

        let (status, restored) = json_request(
            app.clone(),
            "/plugin/v1/states/restore",
            json!({
                "contract_version":RESTORE_REQUEST_CONTRACT_VERSION,
                "state_ref":state_value,
                "expected_model_ref":model(),
                "target_worker_id":"cloud-worker-b",
                "lease":restore_lease_value
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{restored}");
        assert_eq!(restored["payload_base64"], BASE64.encode(payload));

        let (_, stale) = json_request(
            app,
            "/plugin/v1/states/snapshot",
            json!({
                "contract_version":SNAPSHOT_REQUEST_CONTRACT_VERSION,
                "provider_mode":"rwkv_recurrent",
                "model_ref":model(),
                "target_tier":"cold",
                "lease":first_lease,
                "expected_state_version":0,
                "payload_base64":BASE64.encode(b"stale-writer")
            }),
        )
        .await;
        assert_eq!(stale["error"]["code"], "stale_fencing_token");
        let _ = tokio::fs::remove_dir_all(state_dir).await;
    }
}
