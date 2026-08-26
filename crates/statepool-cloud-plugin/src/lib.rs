//! Out-of-process StatePool Cloud Plugin.
//!
//! The first service slice deliberately implements only protocol negotiation,
//! a dynamic Worker directory, explainable placement and bounded in-memory
//! FinOps counters. Snapshot persistence and distributed leases are separate
//! gates and are never fabricated by this service.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rwkv_statepool_plugin_api::{
    EXECUTION_PLAN_CONTRACT_VERSION, ExecutionPlan, HandshakeRequest, HandshakeResponse, Money,
    PLUGIN_CONTRACT_VERSION, PlanRequest, PrivacyClass, USAGE_RECORD_CONTRACT_VERSION, UsageRecord,
    WorkerCapability, WorkerLifecycle, WorkerZone,
};
use serde_json::{Value, json};
use tokio::sync::RwLock;

const CAPABILITIES: &[&str] = &["placement", "worker_registry", "drain", "finops"];

#[derive(Clone, Debug)]
pub struct PluginConfig {
    pub plugin_version: String,
    pub worker_ttl: Duration,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            plugin_version: env!("CARGO_PKG_VERSION").into(),
            worker_ttl: Duration::from_secs(30),
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
}

#[derive(Clone)]
pub struct PluginState {
    config: Arc<PluginConfig>,
    workers: Arc<RwLock<HashMap<String, WorkerCapability>>>,
    usage: Arc<RwLock<Vec<UsageRecord>>>,
    metrics: Arc<Metrics>,
    decisions: Arc<AtomicU64>,
}

impl PluginState {
    pub fn new(config: PluginConfig) -> Result<Self, String> {
        if config.plugin_version.trim().is_empty() || config.worker_ttl.is_zero() {
            return Err("plugin version and Worker TTL must be positive".into());
        }
        Ok(Self {
            config: Arc::new(config),
            workers: Arc::new(RwLock::new(HashMap::new())),
            usage: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(Metrics::default()),
            decisions: Arc::new(AtomicU64::new(1)),
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
            "metadata":"in_memory",
            "object_store":"disabled"
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
    state
        .metrics
        .worker_registrations
        .fetch_add(1, Ordering::Relaxed);
    Ok(Json(json!({"status":"ok","worker_id":worker_id})))
}

async fn heartbeat_worker(
    Path(worker_id): Path<String>,
    State(state): State<PluginState>,
    Json(worker): Json<WorkerCapability>,
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
    let mut workers = state.workers.write().await;
    if !workers.contains_key(&worker_id) {
        return Err(PluginError::new(
            StatusCode::NOT_FOUND,
            "invalid_request",
            "unknown Worker",
            false,
        ));
    }
    workers.insert(worker_id.clone(), worker);
    Ok(Json(json!({"status":"ok","worker_id":worker_id})))
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
    let mut workers = state.workers.write().await;
    let worker = workers.get_mut(&worker_id).ok_or_else(|| {
        PluginError::new(
            StatusCode::NOT_FOUND,
            "invalid_request",
            "unknown Worker",
            false,
        )
    })?;
    worker.lifecycle = WorkerLifecycle::Draining;
    let safe = worker.capacity.running_requests == 0;
    Ok(Json(json!({
        "contract_version":"statepool-drain-status.v1",
        "worker_id":worker_id,
        "status":if safe {"safe_to_stop"} else {"draining"},
        "active_requests":worker.capacity.running_requests,
        "unpersisted_states":0
    })))
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
    let (state_action, restore_ms, affinity) = state_action(worker, &request);
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
    let body = format!(
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
            "statepool_state_bytes_written_total {}\n"
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
    );
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
        ModelRef, PLAN_REQUEST_CONTRACT_VERSION, WORKER_CAPABILITY_CONTRACT_VERSION,
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
            },
            price: Some(WorkerPrice {
                currency: "CNY".into(),
                per_gpu_hour: 2.8,
            }),
            labels: Default::default(),
            reported_at_ms: now_ms(),
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
            json!({"contract_version":"statepool-drain-request.v1","deadline_ms":1000}),
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
}
