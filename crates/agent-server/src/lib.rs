use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rwkv_agent_runtime::{AgentService, TaskSpec};
use serde::Deserialize;
use serde_json::{Value, json};
use std::convert::Infallible;
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

const INDEX_HTML: &str = include_str!("../../../web/index.html");
const APP_CSS: &str = include_str!("../../../web/app.css");
const APP_JS: &str = include_str!("../../../web/app.js");

#[derive(Clone)]
struct ServerState {
    service: AgentService,
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
    let state = ServerState { service };
    Router::new()
        .route("/", get(index))
        .route("/tasks", get(index))
        .route("/assets/app.css", get(app_css))
        .route("/assets/app.js", get(app_js))
        .route("/health", get(health))
        .route("/v1/tasks", get(tasks))
        .route("/v1/task-ledger", get(task_ledger))
        .route("/v1/task-ledger/{task_id}", get(task_record))
        .route("/v1/task-ledger/{task_id}/resume", post(resume_task))
        .route("/v1/task-ledger/{task_id}/cancel", post(cancel_task))
        .route("/v1/agent/run", post(run))
        .route("/v1/agent/run_stream", post(run_stream))
        .route("/v1/agent/run_stateful", post(research))
        .route("/v1/agent/gate", post(gate))
        .route("/v1/tools/call", post(tool))
        .with_state(state)
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
struct RunBody {
    #[serde(default)]
    message: String,
    session_id: String,
    #[serde(default)]
    working_directory: Option<String>,
    #[serde(default)]
    task_spec: Option<TaskSpec>,
    #[serde(default)]
    task_id: Option<String>,
}

#[derive(Deserialize)]
struct ResearchBody {
    message: String,
    session_id: String,
    #[serde(default = "default_branches")]
    branch_width: usize,
    #[serde(default = "default_rounds")]
    max_rounds: usize,
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
struct ToolBody {
    name: String,
    #[serde(default)]
    arguments: Value,
    session_id: String,
    #[serde(default)]
    working_directory: Option<String>,
}

fn default_branches() -> usize {
    4
}

fn default_rounds() -> usize {
    2
}

async fn health(State(state): State<ServerState>) -> (StatusCode, Json<Value>) {
    response(state.service.health().await)
}

async fn tasks(State(state): State<ServerState>) -> (StatusCode, Json<Value>) {
    match state.service.tasks(100).await {
        Ok(tasks) => (StatusCode::OK, Json(task_wall_snapshot(&tasks))),
        Err(error) => response(Err(error)),
    }
}

async fn task_ledger(State(state): State<ServerState>) -> (StatusCode, Json<Value>) {
    match state.service.tasks(100).await {
        Ok(tasks) => {
            let running = tasks
                .iter()
                .filter(|task| matches!(task.status, rwkv_agent_runtime::TaskStatus::Running))
                .count();
            let complete = tasks
                .iter()
                .filter(|task| matches!(task.status, rwkv_agent_runtime::TaskStatus::Succeeded))
                .count();
            let failed = tasks
                .iter()
                .filter(|task| {
                    matches!(
                        task.status,
                        rwkv_agent_runtime::TaskStatus::Failed
                            | rwkv_agent_runtime::TaskStatus::Interrupted
                    )
                })
                .count();
            (
                StatusCode::OK,
                Json(
                    json!({"status":"ok","counts":{"total":tasks.len(),"running":running,"complete":complete,"failed":failed},"tasks":tasks}),
                ),
            )
        }
        Err(error) => response(Err(error)),
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
) -> (StatusCode, Json<Value>) {
    match state.service.task(&task_id).await {
        Ok(task) => (StatusCode::OK, Json(json!({"status":"ok","task":task}))),
        Err(error) => response(Err(error)),
    }
}

async fn resume_task(
    State(state): State<ServerState>,
    Path(task_id): Path<String>,
) -> (StatusCode, Json<Value>) {
    response(state.service.resume_task(&task_id).await)
}

async fn cancel_task(
    State(state): State<ServerState>,
    Path(task_id): Path<String>,
) -> (StatusCode, Json<Value>) {
    match state.service.cancel_task(&task_id).await {
        Ok(task) => (StatusCode::OK, Json(json!({"status":"ok","task":task}))),
        Err(error) => response(Err(error)),
    }
}

async fn run(
    State(state): State<ServerState>,
    Json(body): Json<RunBody>,
) -> (StatusCode, Json<Value>) {
    let task_spec = match resolve_task_spec(&body) {
        Ok(task_spec) => task_spec,
        Err(error) => return bad_request(error),
    };
    response(
        state
            .service
            .run_task_with_id(task_spec, &body.session_id, body.task_id.as_deref())
            .await,
    )
}

async fn run_stream(State(state): State<ServerState>, Json(body): Json<RunBody>) -> Response {
    let task_spec = match resolve_task_spec(&body) {
        Ok(task_spec) => task_spec,
        Err(error) => return bad_request(error).into_response(),
    };
    let (events, receiver) = mpsc::channel::<Value>(64);
    tokio::spawn(async move {
        let _ = state
            .service
            .run_task_stream_with_id(task_spec, &body.session_id, body.task_id.as_deref(), events)
            .await;
    });
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

async fn research(
    State(state): State<ServerState>,
    Json(body): Json<ResearchBody>,
) -> (StatusCode, Json<Value>) {
    response(
        state
            .service
            .research(
                &body.message,
                &body.session_id,
                body.branch_width,
                body.max_rounds,
            )
            .await,
    )
}

async fn gate(
    State(state): State<ServerState>,
    Json(body): Json<GateBody>,
) -> (StatusCode, Json<Value>) {
    response(
        state
            .service
            .gate(
                &body.message,
                body.threshold,
                &body.context,
                body.has_pasted_text,
            )
            .await,
    )
}

async fn tool(
    State(state): State<ServerState>,
    Json(body): Json<ToolBody>,
) -> (StatusCode, Json<Value>) {
    response(
        state
            .service
            .call_tool_with_workspace(
                &body.name,
                body.arguments,
                &body.session_id,
                body.working_directory.as_deref(),
            )
            .await,
    )
}

fn response(result: Result<Value, String>) -> (StatusCode, Json<Value>) {
    match result {
        Ok(value) => (StatusCode::OK, Json(value)),
        Err(message) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"status":"error","error":message})),
        ),
    }
}

fn bad_request(message: String) -> (StatusCode, Json<Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"status":"invalid","error":message})),
    )
}

fn resolve_task_spec(body: &RunBody) -> Result<TaskSpec, String> {
    let legacy_message = body.message.trim();
    let legacy_directory = body
        .working_directory
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let task = match &body.task_spec {
        Some(task) => {
            if !legacy_message.is_empty() && legacy_message != task.objective.trim() {
                return Err(
                    "message and task_spec.objective must match when both are provided".into(),
                );
            }
            let mut task = task.clone();
            match (task.working_directory.as_deref(), legacy_directory) {
                (Some(spec), Some(legacy)) if spec.trim() != legacy => {
                    return Err("working_directory and task_spec.working_directory must match when both are provided".into());
                }
                (None, Some(legacy)) => task.working_directory = Some(legacy.to_string()),
                _ => {}
            }
            task
        }
        None => TaskSpec::legacy(legacy_message, legacy_directory.map(str::to_string)),
    };
    task.normalize().map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_web_ui_has_no_external_runtime_dependency() {
        assert!(INDEX_HTML.contains("STATE AGENT"));
        assert!(INDEX_HTML.contains("/assets/app.css"));
        assert!(INDEX_HTML.contains("/assets/app.js"));
        assert!(!INDEX_HTML.contains("https://"));
        assert!(APP_JS.contains("/v1/agent/run"));
        assert!(APP_JS.contains("/v1/agent/run_stream"));
        assert!(APP_JS.contains("/v1/tasks"));
        assert!(INDEX_HTML.contains("/tasks"));
        assert!(APP_JS.contains("textContent"));
        assert!(APP_CSS.contains("prefers-reduced-motion"));
    }

    #[test]
    fn legacy_run_body_resolves_to_task_spec_v1() {
        let body = RunBody {
            message: " fix calc.py ".into(),
            session_id: "s1".into(),
            working_directory: Some(" /repo ".into()),
            task_spec: None,
            task_id: None,
        };
        let task = resolve_task_spec(&body).unwrap();
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
        let body = RunBody {
            message: String::new(),
            session_id: "s1".into(),
            working_directory: None,
            task_spec: Some(task.clone()),
            task_id: None,
        };
        assert_eq!(resolve_task_spec(&body).unwrap(), task);
    }

    #[test]
    fn duplicate_legacy_and_structured_fields_must_agree() {
        let body = RunBody {
            message: "legacy objective".into(),
            session_id: "s1".into(),
            working_directory: None,
            task_spec: Some(TaskSpec::new("structured objective")),
            task_id: None,
        };
        assert!(resolve_task_spec(&body).is_err());
    }
}
