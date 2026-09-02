use axum::body::{Body, Bytes};
use axum::extract::rejection::{JsonRejection, QueryRejection};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use rwkv_agent_runtime::{
    AgentService, DebugTraceFileKind, DebugTraceFilter, RequestIdentity, ResearchRequest,
    SERVICE_API_VERSION, ServiceErrorCode, ServiceErrorDetail, ServiceStreamEvent,
    TaskControlRequest, TaskRunRequest, ToolCallRequest,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

#[cfg(test)]
use rwkv_agent_runtime::TaskSpec;

const INDEX_HTML: &str = include_str!("../../../web/index.html");
const APP_CSS: &str = include_str!("../../../web/app.css");
const APP_JS: &str = include_str!("../../../web/dist/app.js");
const API_CLIENT_JS: &str = include_str!("../../../web/dist/api-client.js");
const SERVICE_SCHEMA_JSON: &str = include_str!("../../../contracts/agent-service-v1.schema.json");
const OPENAPI_JSON: &str = include_str!("../../../contracts/agent-service-v1.openapi.json");

#[derive(Clone)]
struct ServerState {
    service: AgentService,
    request_sequence: Arc<AtomicU64>,
}

impl ServerState {
    fn next_request_id(&self) -> String {
        let unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        format!(
            "request-{unix_ms:x}-{:x}",
            self.request_sequence.fetch_add(1, Ordering::Relaxed)
        )
    }
}

fn preview(value: &str, limit: usize) -> String {
    let clean = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if clean.chars().count() <= limit {
        return clean;
    }
    let mut output = clean
        .chars()
        .take(limit.saturating_sub(1))
        .collect::<String>();
    output.push('…');
    output
}

pub fn router(service: AgentService) -> Router {
    let state = ServerState {
        service,
        request_sequence: Arc::new(AtomicU64::new(1)),
    };
    let debug_api_enabled = state.service.debug_trace_api_enabled();
    let mut app = Router::new()
        .merge(ui_routes())
        .merge(health_routes())
        .merge(canonical_api_routes())
        .merge(compatibility_routes());
    if debug_api_enabled {
        app = app.merge(debug_routes());
    }
    app.with_state(state)
}

// Keep the stable frontend surface, compatibility aliases and local operator
// surface visibly separate. They may share handlers, but they must not become
// separate control planes or acquire different lifecycle semantics.
fn ui_routes() -> Router<ServerState> {
    Router::new()
        .route("/", get(index))
        .route("/tasks", get(index))
        .route("/status", get(index))
        .route("/assets/app.css", get(app_css))
        .route("/assets/app.js", get(app_js))
        .route("/assets/api-client.js", get(api_client_js))
}

fn health_routes() -> Router<ServerState> {
    Router::new()
        .route("/live", get(live))
        .route("/ready", get(ready))
}

fn canonical_api_routes() -> Router<ServerState> {
    Router::new()
        .route("/v1/openapi.json", get(openapi_document))
        .route("/v1/schema.json", get(service_schema_document))
        .route("/v1/tasks", get(tasks).post(run))
        .route("/v1/tasks/stream", axum::routing::post(run_stream))
        .route("/v1/tasks/{task_id}", get(task_record))
        .route(
            "/v1/tasks/{task_id}/resume",
            axum::routing::post(resume_task),
        )
        .route(
            "/v1/tasks/{task_id}/cancel",
            axum::routing::post(cancel_task),
        )
        .route("/v1/research", axum::routing::post(research))
        .route("/v1/tools/call", axum::routing::post(tool))
}

fn compatibility_routes() -> Router<ServerState> {
    Router::new()
        // Pre-v1 clients used `/health` as an always-200 readiness report.
        .route("/health", get(compat_health))
        // These aliases invoke the canonical handlers and lifecycle.
        .route("/v1/task-ledger", get(task_ledger))
        .route("/v1/task-ledger/{task_id}", get(task_record))
        .route(
            "/v1/task-ledger/{task_id}/resume",
            axum::routing::post(resume_task),
        )
        .route(
            "/v1/task-ledger/{task_id}/cancel",
            axum::routing::post(cancel_task),
        )
        .route("/v1/agent/run", axum::routing::post(run))
        .route("/v1/agent/run_stream", axum::routing::post(run_stream))
        .route("/v1/agent/run_stateful", axum::routing::post(research))
        .route("/v1/agent/gate", axum::routing::post(gate))
}

fn debug_routes() -> Router<ServerState> {
    Router::new()
        .route("/v1/debug/traces", get(debug_traces))
        .route("/v1/debug/traces/{trace_id}", get(debug_trace_manifest))
        .route(
            "/v1/debug/traces/{trace_id}/events",
            get(debug_trace_events),
        )
        .route(
            "/v1/debug/traces/{trace_id}/files/{kind}",
            get(debug_trace_file),
        )
}

async fn index() -> Response {
    secured_asset(Html(INDEX_HTML).into_response(), "text/html; charset=utf-8")
}

async fn app_css() -> Response {
    secured_asset(APP_CSS.into_response(), "text/css; charset=utf-8")
}

async fn app_js() -> Response {
    secured_asset(APP_JS.into_response(), "text/javascript; charset=utf-8")
}

async fn api_client_js() -> Response {
    secured_asset(
        API_CLIENT_JS.into_response(),
        "text/javascript; charset=utf-8",
    )
}

async fn openapi_document() -> Response {
    secured_asset(
        OPENAPI_JSON.into_response(),
        "application/vnd.oai.openapi+json;version=3.1",
    )
}

async fn service_schema_document() -> Response {
    secured_asset(
        SERVICE_SCHEMA_JSON.into_response(),
        "application/schema+json",
    )
}

fn secured_asset(mut response: Response, content_type: &'static str) -> Response {
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        ),
    );
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

#[derive(Deserialize)]
struct GateBody {
    message: String,
    #[serde(default)]
    threshold: Option<f64>,
    #[serde(default)]
    context: String,
    #[serde(default)]
    has_pasted_text: bool,
}

#[derive(Deserialize)]
struct TaskListQuery {
    api_version: String,
    request_id: String,
    owner_id: String,
}

#[derive(Deserialize)]
struct TaskAccessQuery {
    api_version: String,
    request_id: String,
    owner_id: String,
}

#[derive(Deserialize)]
struct DebugTraceListQuery {
    api_version: String,
    request_id: String,
    owner_id: String,
    #[serde(default, rename = "filter_request_id", alias = "trace_request_id")]
    request_filter: Option<String>,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    after_trace_id: Option<String>,
    #[serde(default = "default_debug_page_limit")]
    limit: usize,
}

#[derive(Deserialize)]
struct DebugTraceAccessQuery {
    api_version: String,
    request_id: String,
    owner_id: String,
}

#[derive(Deserialize)]
struct DebugTraceEventsQuery {
    api_version: String,
    request_id: String,
    owner_id: String,
    #[serde(default)]
    after_sequence: u64,
    #[serde(default = "default_debug_page_limit")]
    limit: usize,
}

fn default_debug_page_limit() -> usize {
    100
}

async fn debug_traces(
    State(state): State<ServerState>,
    query: Result<Query<DebugTraceListQuery>, QueryRejection>,
) -> (StatusCode, Json<Value>) {
    let query = match query {
        Ok(Query(query)) => query,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error.body_text(),
                &state.next_request_id(),
            );
        }
    };
    let request_id =
        match validate_control_identity(&query.api_version, &query.request_id, &query.owner_id) {
            Ok(request_id) => request_id,
            Err(error) => {
                return service_error(
                    StatusCode::BAD_REQUEST,
                    ServiceErrorCode::InvalidRequest,
                    error,
                    &state.next_request_id(),
                );
            }
        };
    match state
        .service
        .debug_traces_for_owner(
            &query.owner_id,
            DebugTraceFilter {
                request_id: query.request_filter,
                task_id: query.task_id,
                session_id: query.session_id,
                after_trace_id: query.after_trace_id,
                limit: query.limit,
            },
        )
        .await
    {
        Ok(page) => (
            StatusCode::OK,
            Json(with_meta(
                json!({"status":"ok","page":page}),
                &request_id,
                &query.owner_id,
            )),
        ),
        Err(error) => error_from_message(error, &request_id),
    }
}

async fn debug_trace_manifest(
    State(state): State<ServerState>,
    Path(trace_id): Path<String>,
    query: Result<Query<DebugTraceAccessQuery>, QueryRejection>,
) -> (StatusCode, Json<Value>) {
    let query = match query {
        Ok(Query(query)) => query,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error.body_text(),
                &state.next_request_id(),
            );
        }
    };
    let request_id =
        match validate_control_identity(&query.api_version, &query.request_id, &query.owner_id) {
            Ok(request_id) => request_id,
            Err(error) => {
                return service_error(
                    StatusCode::BAD_REQUEST,
                    ServiceErrorCode::InvalidRequest,
                    error,
                    &state.next_request_id(),
                );
            }
        };
    match state
        .service
        .debug_trace_manifest_for_owner(&trace_id, &query.owner_id)
        .await
    {
        Ok(manifest) => (
            StatusCode::OK,
            Json(with_meta(
                json!({"status":"ok","manifest":manifest}),
                &request_id,
                &query.owner_id,
            )),
        ),
        Err(error) => error_from_message(error, &request_id),
    }
}

async fn debug_trace_events(
    State(state): State<ServerState>,
    Path(trace_id): Path<String>,
    query: Result<Query<DebugTraceEventsQuery>, QueryRejection>,
) -> (StatusCode, Json<Value>) {
    let query = match query {
        Ok(Query(query)) => query,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error.body_text(),
                &state.next_request_id(),
            );
        }
    };
    let request_id =
        match validate_control_identity(&query.api_version, &query.request_id, &query.owner_id) {
            Ok(request_id) => request_id,
            Err(error) => {
                return service_error(
                    StatusCode::BAD_REQUEST,
                    ServiceErrorCode::InvalidRequest,
                    error,
                    &state.next_request_id(),
                );
            }
        };
    match state
        .service
        .debug_trace_events_for_owner(
            &trace_id,
            &query.owner_id,
            query.after_sequence,
            query.limit,
        )
        .await
    {
        Ok(events) => {
            let next_after_sequence = events.last().map(|event| event.sequence);
            (
                StatusCode::OK,
                Json(with_meta(
                    json!({"status":"ok","events":events,"next_after_sequence":next_after_sequence}),
                    &request_id,
                    &query.owner_id,
                )),
            )
        }
        Err(error) => error_from_message(error, &request_id),
    }
}

async fn debug_trace_file(
    State(state): State<ServerState>,
    Path((trace_id, kind)): Path<(String, String)>,
    headers: HeaderMap,
    query: Result<Query<DebugTraceAccessQuery>, QueryRejection>,
) -> Response {
    let query = match query {
        Ok(Query(query)) => query,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error.body_text(),
                &state.next_request_id(),
            )
            .into_response();
        }
    };
    let request_id =
        match validate_control_identity(&query.api_version, &query.request_id, &query.owner_id) {
            Ok(request_id) => request_id,
            Err(error) => {
                return service_error(
                    StatusCode::BAD_REQUEST,
                    ServiceErrorCode::InvalidRequest,
                    error,
                    &state.next_request_id(),
                )
                .into_response();
            }
        };
    let Some(kind) = DebugTraceFileKind::parse(&kind) else {
        return service_error(
            StatusCode::BAD_REQUEST,
            ServiceErrorCode::InvalidRequest,
            "unknown debug trace file kind",
            &request_id,
        )
        .into_response();
    };
    if headers.contains_key(header::RANGE) {
        return service_error(
            StatusCode::RANGE_NOT_SATISFIABLE,
            ServiceErrorCode::InvalidRequest,
            "debug trace raw files do not support Range requests",
            &request_id,
        )
        .into_response();
    }
    match state
        .service
        .debug_trace_file_for_owner(&trace_id, &query.owner_id, kind)
        .await
    {
        Ok(bytes) => {
            let content_type = match kind {
                DebugTraceFileKind::Checksums => "text/plain; charset=utf-8",
                DebugTraceFileKind::TaskRecord | DebugTraceFileKind::FinalResponse => {
                    "application/json; charset=utf-8"
                }
                _ => "application/x-ndjson; charset=utf-8",
            };
            let mut response = Body::from(bytes).into_response();
            response
                .headers_mut()
                .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
            response
                .headers_mut()
                .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
            response
        }
        Err(error) => error_from_message(error, &request_id).into_response(),
    }
}

async fn live(State(state): State<ServerState>) -> (StatusCode, Json<Value>) {
    (StatusCode::OK, Json(state.service.liveness()))
}

async fn ready(State(state): State<ServerState>) -> (StatusCode, Json<Value>) {
    let readiness = state.service.readiness().await;
    let status = if readiness.get("status").and_then(Value::as_str) == Some("ready") {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(readiness))
}

async fn compat_health(State(state): State<ServerState>) -> (StatusCode, Json<Value>) {
    (StatusCode::OK, Json(state.service.readiness().await))
}

async fn tasks(
    State(state): State<ServerState>,
    query: Result<Query<TaskListQuery>, QueryRejection>,
) -> (StatusCode, Json<Value>) {
    let query = match query {
        Ok(Query(query)) => query,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error.body_text(),
                &state.next_request_id(),
            );
        }
    };
    let request_id =
        match validate_control_identity(&query.api_version, &query.request_id, &query.owner_id) {
            Ok(request_id) => request_id,
            Err(error) => {
                return service_error(
                    StatusCode::BAD_REQUEST,
                    ServiceErrorCode::InvalidRequest,
                    error,
                    &state.next_request_id(),
                );
            }
        };
    match state.service.tasks_for_owner(&query.owner_id, 100).await {
        Ok(tasks) => (
            StatusCode::OK,
            Json(with_meta(
                task_wall_snapshot(&tasks),
                &request_id,
                &query.owner_id,
            )),
        ),
        Err(error) => error_from_message(error, &request_id),
    }
}

async fn task_ledger(State(state): State<ServerState>) -> (StatusCode, Json<Value>) {
    match state.service.tasks(100).await {
        Ok(tasks) => (StatusCode::OK, Json(task_wall_snapshot(&tasks))),
        Err(error) => error_from_message(error, &state.next_request_id()),
    }
}

fn task_wall_snapshot(tasks: &[rwkv_agent_runtime::TaskRecord]) -> Value {
    let status = |task: &rwkv_agent_runtime::TaskRecord| {
        serde_json::to_value(task.status)
            .ok()
            .and_then(|value| value.as_str().map(str::to_string))
            .unwrap_or_else(|| "unknown".into())
    };
    let running = tasks
        .iter()
        .filter(|task| status(task) == "running")
        .count();
    let complete = tasks
        .iter()
        .filter(|task| status(task) == "succeeded")
        .count();
    let failed = tasks
        .iter()
        .filter(|task| matches!(status(task).as_str(), "failed" | "interrupted"))
        .count();
    json!({
        "status":"ok",
        "durable":true,
        "counts":{"total":tasks.len(),"running":running,"complete":complete,"failed":failed},
        "tasks":tasks.iter().map(|task| {
            let task_status = status(task);
            let response = task.final_response.as_ref();
            json!({
                "id":task.task_id,
                "request_id":task.request_id,
                "owner_id":task.owner_id,
                "session_id":task.session_id,
                "message":preview(&task.task_spec.objective, 160),
                "kind":"agent",
                "status":if task_status == "succeeded" {"complete"} else if matches!(task_status.as_str(), "failed" | "interrupted" | "cancelled") {"error"} else {task_status.as_str()},
                "route":response.and_then(|value|value.pointer("/route/mode")).and_then(Value::as_str).unwrap_or(""),
                "state":task.current_stage.as_deref().unwrap_or(if task_status == "succeeded" {"released"} else {"checkpointed"}),
                "tool_count":response.and_then(|value|value.pointer("/trace/agent/tool_steps")).and_then(Value::as_array).map_or(0, Vec::len),
                "created_unix_ms":task.created_unix_ms,
                "elapsed_ms":task.updated_unix_ms.saturating_sub(task.created_unix_ms),
                "error":task.error,
                "revision":task.revision,
                "recovery_count":task.recovery_count,
            })
        }).collect::<Vec<_>>()
    })
}

async fn task_record(
    State(state): State<ServerState>,
    Path(task_id): Path<String>,
    query: Result<Query<TaskAccessQuery>, QueryRejection>,
) -> (StatusCode, Json<Value>) {
    let query = match query {
        Ok(Query(query)) => query,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error.body_text(),
                &state.next_request_id(),
            );
        }
    };
    let request_id =
        match validate_control_identity(&query.api_version, &query.request_id, &query.owner_id) {
            Ok(request_id) => request_id,
            Err(error) => {
                return service_error(
                    StatusCode::BAD_REQUEST,
                    ServiceErrorCode::InvalidRequest,
                    error,
                    &state.next_request_id(),
                );
            }
        };
    match state
        .service
        .task_for_owner(&task_id, &query.owner_id)
        .await
    {
        Ok(task) => (
            StatusCode::OK,
            Json(with_meta(
                json!({"status":"ok","task":safe_task_record(&task)}),
                &request_id,
                &query.owner_id,
            )),
        ),
        Err(error) => error_from_message(error, &request_id),
    }
}

fn safe_task_record(task: &rwkv_agent_runtime::TaskRecord) -> Value {
    json!({
        "ledger_schema_version":task.ledger_schema_version,
        "task_id":task.task_id,
        "request_id":task.request_id,
        "owner_id":task.owner_id,
        "session_id":task.session_id,
        "status":task.status,
        "current_stage":task.current_stage,
        "created_unix_ms":task.created_unix_ms,
        "updated_unix_ms":task.updated_unix_ms,
        "revision":task.revision,
        "recovery_count":task.recovery_count,
        "trace_id":task.trace_id,
        "debug_capture":task.debug_capture,
        "task_spec_summary":{
            "schema_version":task.task_spec.schema_version,
            "objective_preview":preview(&task.task_spec.objective, 160),
            "objective_chars":task.task_spec.objective.chars().count(),
            "acceptance_criteria_count":task.task_spec.acceptance_criteria.len(),
            "constraint_count":task.task_spec.constraints.len(),
            "verification_command_count":task.task_spec.verification_commands.len(),
            "stage_count":task.stages.len(),
        },
        "stages":task.stages.iter().map(|stage|json!({
            "id":stage.spec.id,
            "status":stage.status,
            "attempts":stage.attempts,
            "started_unix_ms":stage.started_unix_ms,
            "completed_unix_ms":stage.completed_unix_ms,
            "error":preview(&stage.error, 240),
        })).collect::<Vec<_>>(),
        "error":preview(&task.error, 240),
        "events":task.events.iter().map(|event|json!({
            "sequence":event.sequence,
            "unix_ms":event.unix_ms,
            "kind":event.kind,
            "stage_id":event.stage_id,
            "detail":preview(&event.detail, 240),
        })).collect::<Vec<_>>(),
        "final_summary":task.final_response.as_ref().map(|response|json!({
            "status":response.get("status").cloned().unwrap_or(Value::Null),
            "route":response.pointer("/route/mode").cloned().unwrap_or(Value::Null),
            "answer_chars":response.get("answer").and_then(Value::as_str).map_or(0, |value|value.chars().count()),
            "tool_steps":response.pointer("/trace/agent/tool_steps").and_then(Value::as_array).map_or(0, Vec::len),
        })),
    })
}

async fn resume_task(
    State(state): State<ServerState>,
    Path(task_id): Path<String>,
    body: Result<Json<TaskControlRequest>, JsonRejection>,
) -> (StatusCode, Json<Value>) {
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error.body_text(),
                &state.next_request_id(),
            );
        }
    };
    let request_id =
        match validate_control_identity(&body.api_version, &body.request_id, &body.owner_id) {
            Ok(request_id) => request_id,
            Err(error) => {
                return service_error(
                    StatusCode::BAD_REQUEST,
                    ServiceErrorCode::InvalidRequest,
                    error,
                    &state.next_request_id(),
                );
            }
        };
    response_with_identity(
        state
            .service
            .resume_task_for_owner(&task_id, &body.owner_id)
            .await,
        &request_id,
        &body.owner_id,
    )
}

async fn cancel_task(
    State(state): State<ServerState>,
    Path(task_id): Path<String>,
    body: Result<Json<TaskControlRequest>, JsonRejection>,
) -> (StatusCode, Json<Value>) {
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error.body_text(),
                &state.next_request_id(),
            );
        }
    };
    let request_id =
        match validate_control_identity(&body.api_version, &body.request_id, &body.owner_id) {
            Ok(request_id) => request_id,
            Err(error) => {
                return service_error(
                    StatusCode::BAD_REQUEST,
                    ServiceErrorCode::InvalidRequest,
                    error,
                    &state.next_request_id(),
                );
            }
        };
    match state
        .service
        .cancel_task_for_owner(&task_id, &body.owner_id)
        .await
    {
        Ok(task) => (
            StatusCode::OK,
            Json(with_meta(
                json!({"status":"ok","task":task}),
                &request_id,
                &body.owner_id,
            )),
        ),
        Err(error) => error_from_message(error, &request_id),
    }
}

async fn run(
    State(state): State<ServerState>,
    body: Result<Json<TaskRunRequest>, JsonRejection>,
) -> (StatusCode, Json<Value>) {
    let fallback_request_id = state.next_request_id();
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error.body_text(),
                &fallback_request_id,
            );
        }
    };
    let identity = match request_identity(&body, &fallback_request_id) {
        Ok(identity) => identity,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error,
                &fallback_request_id,
            );
        }
    };
    let task_spec = match body.resolve_task_spec() {
        Ok(task_spec) => task_spec,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error,
                &identity.request_id,
            );
        }
    };
    response_with_request_identity(
        state
            .service
            .run_task_with_identity(task_spec, &identity, body.task_id.as_deref())
            .await,
        &identity,
    )
}

async fn run_stream(
    State(state): State<ServerState>,
    body: Result<Json<TaskRunRequest>, JsonRejection>,
) -> Response {
    let fallback_request_id = state.next_request_id();
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error.body_text(),
                &fallback_request_id,
            )
            .into_response();
        }
    };
    let identity = match request_identity(&body, &fallback_request_id) {
        Ok(identity) => identity,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error,
                &fallback_request_id,
            )
            .into_response();
        }
    };
    let task_spec = match body.resolve_task_spec() {
        Ok(task_spec) => task_spec,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error,
                &identity.request_id,
            )
            .into_response();
        }
    };
    let (runtime_events, runtime_receiver) = mpsc::channel::<Value>(64);
    let (client_events, receiver) = mpsc::channel::<Value>(64);
    let runtime_identity = identity.clone();
    let disconnect_service = state.service.clone();
    let requested_task_id = body.task_id.clone();
    tokio::spawn(async move {
        let _ = state
            .service
            .run_task_stream_with_identity(
                task_spec,
                &runtime_identity,
                requested_task_id.as_deref(),
                runtime_events,
            )
            .await;
    });
    spawn_event_bridge(
        identity,
        body.task_id,
        runtime_receiver,
        client_events,
        Some(disconnect_service),
    );
    let stream = ReceiverStream::new(receiver)
        .map(|value| Ok::<Bytes, Infallible>(Bytes::from(format!("{value}\n"))));
    let mut response = Body::from_stream(stream).into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson; charset=utf-8"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

fn spawn_event_bridge(
    identity: RequestIdentity,
    initial_task_id: Option<String>,
    mut runtime_receiver: mpsc::Receiver<Value>,
    client_events: mpsc::Sender<Value>,
    disconnect_service: Option<AgentService>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut sequence = 0_u64;
        let mut task_id = initial_task_id;
        let mut terminal = false;
        let mut client_disconnected = false;
        loop {
            let next = if client_disconnected {
                runtime_receiver.recv().await
            } else {
                tokio::select! {
                    value = runtime_receiver.recv() => value,
                    () = client_events.closed() => {
                        client_disconnected = true;
                        if let (Some(task_id), Some(service)) =
                            (task_id.as_deref(), disconnect_service.as_ref())
                            && service
                                .cancel_task_for_owner(task_id, &identity.owner_id)
                                .await
                                .is_ok()
                        {
                            return;
                        }
                        // The client can disconnect before the durable task is
                        // created. Wait for the first runtime event to learn
                        // the generated task id, then cancel it before work
                        // proceeds past the next controller boundary.
                        continue;
                    }
                }
            };
            let Some(mut value) = next else {
                break;
            };
            let kind = value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("runtime_event")
                .to_string();
            if terminal {
                continue;
            }
            if task_id.is_none() {
                task_id = value
                    .get("task_id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| {
                        value
                            .pointer("/response/task_id")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    });
            }
            if client_disconnected {
                if let (Some(task_id), Some(service)) =
                    (task_id.as_deref(), disconnect_service.as_ref())
                {
                    let _ = service
                        .cancel_task_for_owner(task_id, &identity.owner_id)
                        .await;
                }
                return;
            }
            if kind == "final"
                && let Some(response) = value.get_mut("response")
            {
                *response = with_meta(response.take(), &identity.request_id, &identity.owner_id);
            }
            let mut data = value.as_object().cloned().unwrap_or_default();
            if let (Some(task_id), Some(service)) =
                (task_id.as_deref(), disconnect_service.as_ref())
                && let Ok(record) = service.task_for_owner(task_id, &identity.owner_id).await
            {
                if let Some(trace_id) = record.trace_id {
                    data.insert("trace_id".into(), json!(trace_id));
                }
                if let Some(capture) = record.debug_capture {
                    data.insert("debug_capture".into(), capture);
                }
            }
            sequence += 1;
            let event = ServiceStreamEvent::new(&identity, task_id.clone(), sequence, &kind, data);
            terminal = matches!(kind.as_str(), "final" | "error");
            let encoded = serde_json::to_value(event).unwrap_or_else(
                |error| json!({"type":"error","error":format!("encode stream event: {error}")}),
            );
            if client_events.send(encoded).await.is_err() {
                if let Some(task_id) = task_id.as_deref()
                    && let Some(service) = &disconnect_service
                {
                    let _ = service
                        .cancel_task_for_owner(task_id, &identity.owner_id)
                        .await;
                }
                return;
            }
        }
        if !terminal && !client_disconnected {
            sequence += 1;
            let detail = ServiceErrorDetail {
                code: ServiceErrorCode::Internal,
                message: "runtime stream ended without a terminal event".into(),
                retryable: false,
            };
            let event = ServiceStreamEvent::new(
                &identity,
                task_id,
                sequence,
                "error",
                Map::from_iter([
                    ("error".into(), Value::String(detail.message.clone())),
                    (
                        "error_detail".into(),
                        serde_json::to_value(detail).unwrap_or(Value::Null),
                    ),
                ]),
            );
            let _ = client_events
                .send(serde_json::to_value(event).unwrap_or(Value::Null))
                .await;
        }
    })
}

async fn research(
    State(state): State<ServerState>,
    body: Result<Json<ResearchRequest>, JsonRejection>,
) -> (StatusCode, Json<Value>) {
    let fallback_request_id = state.next_request_id();
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error.body_text(),
                &fallback_request_id,
            );
        }
    };
    let identity = match RequestIdentity::resolve(
        &body.api_version,
        body.request_id.as_deref(),
        body.owner_id.as_deref(),
        &body.session_id,
        || fallback_request_id.clone(),
    ) {
        Ok(identity) => identity,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error,
                &fallback_request_id,
            );
        }
    };
    response_with_request_identity(
        state
            .service
            .research_with_identity(&body.message, &identity, body.branch_width, body.max_rounds)
            .await,
        &identity,
    )
}

async fn gate(
    State(state): State<ServerState>,
    Json(body): Json<GateBody>,
) -> (StatusCode, Json<Value>) {
    let request_id = state.next_request_id();
    response_with_identity(
        state
            .service
            .gate(
                &body.message,
                body.threshold,
                &body.context,
                body.has_pasted_text,
            )
            .await,
        &request_id,
        "compat:gate",
    )
}

async fn tool(
    State(state): State<ServerState>,
    body: Result<Json<ToolCallRequest>, JsonRejection>,
) -> (StatusCode, Json<Value>) {
    let fallback_request_id = state.next_request_id();
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error.body_text(),
                &fallback_request_id,
            );
        }
    };
    let identity = match RequestIdentity::resolve(
        &body.api_version,
        body.request_id.as_deref(),
        body.owner_id.as_deref(),
        &body.session_id,
        || fallback_request_id.clone(),
    ) {
        Ok(identity) => identity,
        Err(error) => {
            return service_error(
                StatusCode::BAD_REQUEST,
                ServiceErrorCode::InvalidRequest,
                error,
                &fallback_request_id,
            );
        }
    };
    response_with_request_identity(
        state
            .service
            .call_tool_with_identity(
                &body.name,
                body.arguments,
                &identity,
                body.working_directory.as_deref(),
            )
            .await,
        &identity,
    )
}

fn response_with_request_identity(
    result: Result<Value, String>,
    identity: &RequestIdentity,
) -> (StatusCode, Json<Value>) {
    match result {
        Ok(value) => {
            let mut value = with_meta(value, &identity.request_id, &identity.owner_id);
            if let Some(object) = value.as_object_mut() {
                object.insert("session_id".into(), json!(identity.session_id));
            }
            (StatusCode::OK, Json(value))
        }
        Err(message) => {
            let (status, Json(mut value)) = error_from_message(message, &identity.request_id);
            if let Some(object) = value.as_object_mut() {
                object.insert("owner_id".into(), json!(identity.owner_id));
                object.insert("session_id".into(), json!(identity.session_id));
            }
            (status, Json(value))
        }
    }
}

fn response_with_identity(
    result: Result<Value, String>,
    request_id: &str,
    owner_id: &str,
) -> (StatusCode, Json<Value>) {
    match result {
        Ok(value) => (StatusCode::OK, Json(with_meta(value, request_id, owner_id))),
        Err(message) => {
            let (status, Json(mut value)) = error_from_message(message, request_id);
            if let Some(object) = value.as_object_mut() {
                object.insert("owner_id".into(), json!(owner_id));
            }
            (status, Json(value))
        }
    }
}

fn with_meta(mut value: Value, request_id: &str, owner_id: &str) -> Value {
    if let Some(object) = value.as_object_mut() {
        object.insert("api_version".into(), json!(SERVICE_API_VERSION));
        object.insert("request_id".into(), json!(request_id));
        object.insert("owner_id".into(), json!(owner_id));
    }
    value
}

fn request_identity(
    body: &TaskRunRequest,
    generated_request_id: &str,
) -> Result<RequestIdentity, String> {
    RequestIdentity::resolve(
        &body.api_version,
        body.request_id.as_deref(),
        body.owner_id.as_deref(),
        &body.session_id,
        || generated_request_id.to_string(),
    )
}

fn validate_control_identity(
    api_version: &str,
    request_id: &str,
    owner_id: &str,
) -> Result<String, String> {
    RequestIdentity::resolve(
        api_version,
        Some(request_id),
        Some(owner_id),
        "control",
        String::new,
    )
    .map(|identity| identity.request_id)
}

fn error_from_message(message: String, request_id: &str) -> (StatusCode, Json<Value>) {
    let lower = message.to_ascii_lowercase();
    let (status, code) = if lower.contains("not authorized")
        || lower.contains("read task `")
        || lower.contains("debug trace not found")
    {
        (StatusCode::NOT_FOUND, ServiceErrorCode::NotFound)
    } else if lower.contains("already exists")
        || lower.contains("already belongs")
        || lower.contains("not resumable")
        || lower.contains("already terminal")
        || lower.contains("cannot cancel")
        || lower.contains("side-effect reconciliation")
    {
        (StatusCode::CONFLICT, ServiceErrorCode::Conflict)
    } else if lower.contains("unsupported") || lower.contains("is disabled") {
        (StatusCode::NOT_IMPLEMENTED, ServiceErrorCode::Unsupported)
    } else if lower.contains("cancelled") {
        (StatusCode::CONFLICT, ServiceErrorCode::Cancelled)
    } else if lower.contains("deadline") || lower.contains("timed out") {
        (
            StatusCode::GATEWAY_TIMEOUT,
            ServiceErrorCode::DeadlineExceeded,
        )
    } else if lower.contains("connection")
        || lower.contains("connect")
        || lower.contains("sidecar")
        || lower.contains("data plane")
    {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            ServiceErrorCode::Unavailable,
        )
    } else if lower.contains("must ")
        || lower.contains("invalid")
        || lower.contains("empty")
        || lower.contains("requires ")
    {
        (StatusCode::BAD_REQUEST, ServiceErrorCode::InvalidRequest)
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            ServiceErrorCode::Internal,
        )
    };
    let trace_id = marker_value(&message, "debug_trace_id=");
    let trace_incomplete = marker_value(&message, "debug_trace_incomplete=");
    let capture_status = marker_value(&message, "debug_capture_status=");
    let capture_mode = marker_value(&message, "debug_capture_mode=");
    let (status, Json(mut value)) = service_error(status, code, message, request_id);
    if let Some(object) = value.as_object_mut() {
        if let Some(trace_id) = trace_id {
            object.insert("trace_id".into(), json!(trace_id));
        }
        if capture_status.is_some() || trace_incomplete.is_some() {
            let mut debug = Map::from_iter([(
                "status".into(),
                json!(capture_status.unwrap_or_else(|| "incomplete".into())),
            )]);
            if let Some(mode) = capture_mode {
                debug.insert("mode".into(), json!(mode));
            }
            if let Some(error) = trace_incomplete {
                debug.insert("error".into(), json!(error));
            }
            object.insert("debug_capture".into(), Value::Object(debug));
        }
    }
    (status, Json(value))
}

fn marker_value(message: &str, marker: &str) -> Option<String> {
    message
        .split(marker)
        .nth(1)
        .map(|value| value.split(';').next().unwrap_or_default().trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn service_error(
    status: StatusCode,
    code: ServiceErrorCode,
    message: impl Into<String>,
    request_id: &str,
) -> (StatusCode, Json<Value>) {
    let detail = ServiceErrorDetail {
        code,
        message: message.into(),
        retryable: code.retryable(),
    };
    (
        status,
        Json(json!({
            "api_version":SERVICE_API_VERSION,
            "request_id":request_id,
            "status":"error",
            "error":detail.message,
            "error_detail":detail,
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request;
    use rwkv_agent_runtime::RuntimeConfig;
    use tower::ServiceExt;

    async fn test_router() -> Router {
        let directory = tempfile::tempdir().unwrap().keep();
        let service = AgentService::new(RuntimeConfig {
            runtime_revision: "test-revision".into(),
            model_urls: vec!["http://127.0.0.1:9".into()],
            data_plane_url: "http://127.0.0.1:9".into(),
            session_dir: directory,
            ..RuntimeConfig::default()
        })
        .await
        .unwrap();
        router(service)
    }

    #[test]
    fn embedded_web_ui_has_no_external_runtime_dependency() {
        assert!(INDEX_HTML.contains("RWKV Agent"));
        assert!(INDEX_HTML.contains("/assets/app.css"));
        assert!(INDEX_HTML.contains("/assets/app.js"));
        assert!(!INDEX_HTML.contains("https://"));
        assert!(API_CLIENT_JS.contains("/v1/tasks/stream"));
        assert!(API_CLIENT_JS.contains("/v1/tasks?"));
        assert!(API_CLIENT_JS.contains("/ready"));
        assert!(API_CLIENT_JS.contains(SERVICE_API_VERSION));
        assert!(!APP_JS.contains("/v1/agent/run"));
        assert!(!APP_JS.contains("/v1/task-ledger"));
        assert!(!APP_JS.contains("/health"));
        assert!(INDEX_HTML.contains("/tasks"));
        assert!(INDEX_HTML.contains("/status"));
        assert!(APP_JS.contains("textContent"));
        assert!(APP_CSS.contains("prefers-reduced-motion"));
        let browser_surface =
            format!("{INDEX_HTML}{APP_CSS}{APP_JS}{API_CLIENT_JS}").to_lowercase();
        for forbidden in ["deepseek", "xharness", "xlang", "__dsh_boot__"] {
            assert!(!browser_surface.contains(forbidden));
        }
    }

    #[test]
    fn legacy_run_body_resolves_to_task_spec_v1() {
        let body = TaskRunRequest {
            api_version: SERVICE_API_VERSION.into(),
            request_id: None,
            owner_id: None,
            message: " fix calc.py ".into(),
            session_id: "s1".into(),
            working_directory: Some(" /repo ".into()),
            task_spec: None,
            task_id: None,
        };
        let task = body.resolve_task_spec().unwrap();
        assert_eq!(task.schema_version, 1);
        assert_eq!(task.objective, "fix calc.py");
        assert_eq!(task.working_directory.as_deref(), Some("/repo"));
    }

    #[test]
    fn structured_task_spec_can_replace_legacy_message() {
        let mut task = TaskSpec::legacy("Fix calc.py", Some("/repo".into()));
        task.acceptance_criteria = vec!["test passes".into()];
        task.verification_commands = vec!["python3 test_calc.py".into()];
        task.requires_mutation = Some(true);
        let body = TaskRunRequest {
            api_version: SERVICE_API_VERSION.into(),
            request_id: None,
            owner_id: None,
            message: String::new(),
            session_id: "s1".into(),
            working_directory: None,
            task_spec: Some(task.clone()),
            task_id: None,
        };
        assert_eq!(body.resolve_task_spec().unwrap(), task);
    }

    #[test]
    fn duplicate_legacy_and_structured_fields_must_agree() {
        let body = TaskRunRequest {
            api_version: SERVICE_API_VERSION.into(),
            request_id: None,
            owner_id: None,
            message: "legacy objective".into(),
            session_id: "s1".into(),
            working_directory: None,
            task_spec: Some(TaskSpec::new("structured objective")),
            task_id: None,
        };
        assert!(body.resolve_task_spec().is_err());
    }

    #[test]
    fn structured_errors_keep_compatibility_string_and_machine_code() {
        let (status, Json(value)) = service_error(
            StatusCode::SERVICE_UNAVAILABLE,
            ServiceErrorCode::Unavailable,
            "sidecar offline",
            "request-1",
        );
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            value.get("error").and_then(Value::as_str),
            Some("sidecar offline")
        );
        assert_eq!(
            value.pointer("/error_detail/code").and_then(Value::as_str),
            Some("unavailable")
        );
        assert_eq!(
            value.get("request_id").and_then(Value::as_str),
            Some("request-1")
        );
    }

    #[test]
    fn canonical_and_compatibility_routes_share_handlers() {
        let source = include_str!("lib.rs");
        assert!(source.contains("fn canonical_api_routes()"));
        assert!(source.contains("fn compatibility_routes()"));
        assert!(source.contains(".route(\"/v1/tasks\", get(tasks).post(run))"));
        assert!(source.contains(".route(\"/v1/agent/run\", axum::routing::post(run))"));
        assert!(source.contains(".route(\"/v1/tasks/stream\", axum::routing::post(run_stream))"));
        assert!(
            source.contains(".route(\"/v1/agent/run_stream\", axum::routing::post(run_stream))")
        );
    }

    #[test]
    fn openapi_tracks_every_canonical_frontend_route() {
        let document: Value = serde_json::from_str(OPENAPI_JSON).unwrap();
        assert_eq!(
            document.pointer("/info/version").and_then(Value::as_str),
            Some(SERVICE_API_VERSION)
        );
        for (method, path) in [
            ("get", "/live"),
            ("get", "/ready"),
            ("get", "/v1/openapi.json"),
            ("get", "/v1/schema.json"),
            ("get", "/v1/tasks"),
            ("post", "/v1/tasks"),
            ("post", "/v1/tasks/stream"),
            ("get", "/v1/tasks/{task_id}"),
            ("post", "/v1/tasks/{task_id}/resume"),
            ("post", "/v1/tasks/{task_id}/cancel"),
            ("post", "/v1/research"),
            ("post", "/v1/tools/call"),
        ] {
            assert!(
                document
                    .pointer(&format!(
                        "/paths/{}/{method}",
                        path.replace('~', "~0").replace('/', "~1")
                    ))
                    .is_some(),
                "OpenAPI is missing {method} {path}"
            );
        }
        for compatibility_path in [
            "/health",
            "/v1/agent/run",
            "/v1/agent/run_stream",
            "/v1/task-ledger",
        ] {
            let pointer = format!(
                "/paths/{}",
                compatibility_path.replace('~', "~0").replace('/', "~1")
            );
            assert!(
                document.pointer(&pointer).is_none(),
                "compatibility route leaked into the canonical OpenAPI surface"
            );
        }
    }

    #[tokio::test]
    async fn machine_readable_contract_documents_are_served_with_safe_types() {
        let app = test_router().await;
        let openapi = app
            .clone()
            .oneshot(
                Request::get("/v1/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(openapi.status(), StatusCode::OK);
        assert_eq!(
            openapi
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/vnd.oai.openapi+json;version=3.1")
        );
        let openapi: Value =
            serde_json::from_slice(&to_bytes(openapi.into_body(), 1024 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(openapi["openapi"], "3.1.0");

        let schema = app
            .oneshot(Request::get("/v1/schema.json").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(schema.status(), StatusCode::OK);
        assert_eq!(
            schema
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/schema+json")
        );
    }

    #[tokio::test]
    async fn liveness_is_provider_independent_and_invalid_json_is_structured() {
        let app = test_router().await;
        let live = app
            .clone()
            .oneshot(Request::get("/live").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(live.status(), StatusCode::OK);
        let live: Value =
            serde_json::from_slice(&to_bytes(live.into_body(), 64 * 1024).await.unwrap()).unwrap();
        assert_eq!(live.get("status").and_then(Value::as_str), Some("alive"));

        let invalid = app
            .oneshot(
                Request::post("/v1/tasks")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"api_version":"v2","session_id":"s1","message":"hello"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        let invalid: Value =
            serde_json::from_slice(&to_bytes(invalid.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(
            invalid
                .pointer("/error_detail/code")
                .and_then(Value::as_str),
            Some("invalid_request")
        );
        assert!(invalid.get("request_id").is_some());
    }

    #[tokio::test]
    async fn debug_routes_are_absent_in_release_default_off_mode() {
        let response = test_router()
            .await
            .oneshot(
                Request::get(concat!(
                    "/v1/debug/traces?api_version=rwkv-agent.service.v1",
                    "&request_id=debug-off&owner_id=owner"
                ))
                .body(Body::empty())
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn readiness_and_run_report_provider_unavailable_without_fallback() {
        let app = test_router().await;
        let ready = app
            .clone()
            .oneshot(Request::get("/ready").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(ready.status(), StatusCode::SERVICE_UNAVAILABLE);
        let ready: Value =
            serde_json::from_slice(&to_bytes(ready.into_body(), 64 * 1024).await.unwrap()).unwrap();
        assert_eq!(
            ready.get("status").and_then(Value::as_str),
            Some("unavailable")
        );
        assert_eq!(
            ready
                .pointer("/components/model_sidecar/status")
                .and_then(Value::as_str),
            Some("unavailable")
        );
        assert_eq!(
            ready
                .pointer("/components/data_plane/status")
                .and_then(Value::as_str),
            Some("unavailable")
        );
        assert_eq!(
            ready
                .pointer("/state_parallel_search/endpoint")
                .and_then(Value::as_str),
            Some("/v1/research")
        );

        let run = app
            .oneshot(
                Request::post("/v1/tasks")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({
                            "api_version":SERVICE_API_VERSION,
                            "request_id":"request-unavailable",
                            "owner_id":"owner-unavailable",
                            "session_id":"session-unavailable",
                            "task_id":"task-unavailable",
                            "task_spec":{"objective":"answer without fallback"}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(run.status(), StatusCode::SERVICE_UNAVAILABLE);
        let run: Value =
            serde_json::from_slice(&to_bytes(run.into_body(), 64 * 1024).await.unwrap()).unwrap();
        assert_eq!(
            run.pointer("/error_detail/code").and_then(Value::as_str),
            Some("unavailable")
        );
        assert_eq!(
            run.get("request_id").and_then(Value::as_str),
            Some("request-unavailable")
        );
        assert_eq!(
            run.get("owner_id").and_then(Value::as_str),
            Some("owner-unavailable")
        );
        assert_eq!(
            run.get("session_id").and_then(Value::as_str),
            Some("session-unavailable")
        );
    }

    #[tokio::test]
    async fn stream_bridge_orders_events_and_emits_exactly_one_terminal() {
        let identity = RequestIdentity {
            api_version: SERVICE_API_VERSION.into(),
            request_id: "request-1".into(),
            owner_id: "owner-1".into(),
            session_id: "session-1".into(),
        };
        let (runtime_events, runtime_receiver) = mpsc::channel(8);
        let (client_events, mut client_receiver) = mpsc::channel(8);
        let bridge = spawn_event_bridge(
            identity,
            Some("task-1".into()),
            runtime_receiver,
            client_events,
            None,
        );

        runtime_events
            .send(json!({"type":"phase","phase":"model"}))
            .await
            .unwrap();
        runtime_events
            .send(json!({"type":"final","response":{"status":"completed"}}))
            .await
            .unwrap();
        runtime_events
            .send(json!({"type":"error","error":"must be discarded"}))
            .await
            .unwrap();
        drop(runtime_events);
        bridge.await.unwrap();

        let mut events = Vec::new();
        while let Some(event) = client_receiver.recv().await {
            events.push(event);
        }
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].get("sequence").and_then(Value::as_u64), Some(1));
        assert_eq!(events[0].get("type").and_then(Value::as_str), Some("phase"));
        assert_eq!(events[1].get("sequence").and_then(Value::as_u64), Some(2));
        assert_eq!(events[1].get("type").and_then(Value::as_str), Some("final"));
        assert_eq!(
            events[1]
                .pointer("/response/request_id")
                .and_then(Value::as_str),
            Some("request-1")
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(
                    event.get("type").and_then(Value::as_str),
                    Some("final" | "error")
                ))
                .count(),
            1
        );
    }
}
