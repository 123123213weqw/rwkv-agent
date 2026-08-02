use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use rwkv_agent_runtime::AgentService;
use serde::Deserialize;
use serde_json::{Value, json};

pub fn router(service: AgentService) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/agent/run", post(run))
        .route("/v1/agent/run_stateful", post(research))
        .route("/v1/agent/gate", post(gate))
        .route("/v1/tools/call", post(tool))
        .with_state(service)
}

#[derive(Deserialize)]
struct RunBody {
    message: String,
    session_id: String,
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
}

fn default_branches() -> usize {
    4
}

fn default_rounds() -> usize {
    2
}

async fn health(State(service): State<AgentService>) -> (StatusCode, Json<Value>) {
    response(service.health().await)
}

async fn run(
    State(service): State<AgentService>,
    Json(body): Json<RunBody>,
) -> (StatusCode, Json<Value>) {
    response(service.run(&body.message, &body.session_id).await)
}

async fn research(
    State(service): State<AgentService>,
    Json(body): Json<ResearchBody>,
) -> (StatusCode, Json<Value>) {
    response(
        service
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
    State(service): State<AgentService>,
    Json(body): Json<GateBody>,
) -> (StatusCode, Json<Value>) {
    response(
        service
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
    State(service): State<AgentService>,
    Json(body): Json<ToolBody>,
) -> (StatusCode, Json<Value>) {
    response(
        service
            .call_tool(&body.name, body.arguments, &body.session_id)
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
