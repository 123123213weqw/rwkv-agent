use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::body::Body;
use axum::extract::{Path, State};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use rwkv_agent_runtime::{
    AgentService, RuntimeConfig, StageStatus, TaskLedger, TaskSpec, TaskStageSpec, TaskStatus,
};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::sync::Mutex;

#[derive(Clone, Default)]
struct MockState {
    prefills: Arc<AtomicUsize>,
    releases: Arc<AtomicUsize>,
    prompts: Arc<Mutex<HashMap<String, String>>>,
}

async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{address}")
}

async fn sidecar() -> (String, MockState) {
    let state = MockState::default();
    let router = Router::new()
        .route(
            "/health",
            get(|| async { Json(json!({"status":"ready","persistent_state":{"allocated":0}})) }),
        )
        .route("/v1/gate/tool", post(gate))
        .route("/v1/states/prefill", post(prefill))
        .route("/v1/states/{state_id}/fork", post(fork))
        .route("/v1/states/batch_continue", post(continue_states))
        .route("/v1/states/stream_continue", post(stream_continue_state))
        .route("/v1/states/release", post(release))
        .with_state(state.clone());
    (spawn(router).await, state)
}

async fn data_plane() -> String {
    let router = Router::new()
        .route("/health", get(|| async { Json(json!({"status":"ready","tools":["web_search","knowledge_search","long_text_qa"]})) }))
        .route("/v1/session/text/status", post(|| async { Json(json!({"status":"empty","active":false})) }))
        .route("/v1/session/text", post(|Json(body): Json<Value>| async move {
            Json(json!({"status":"accepted","document":{"chars":body["text"].as_str().unwrap_or_default().chars().count()}}))
        }))
        .route("/v1/tools/call", post(|Json(body): Json<Value>| async move {
            Json(json!({"status":"ok","tool":body["name"],"evidence":[{"id":"W1","title":"Primary","content":"Supported fact.","uri":"https://example.test/fact"}]}))
        }))
        .route("/v1/answers/validate", post(|Json(body): Json<Value>| async move {
            Json(json!({"valid":true,"answer":body["answer"],"errors":[]}))
        }))
        .route("/v1/evidence/reduce", post(|Json(body): Json<Value>| async move {
            let evidence = body["tool_results"].as_array().into_iter().flatten().flat_map(|row| row["evidence"].as_array().into_iter().flatten().cloned()).collect::<Vec<_>>();
            Json(json!({"status":"ok","evidence":evidence.into_iter().take(8).collect::<Vec<_>>()}))
        }))
        .route("/v1/queries/coordinate", post(|Json(body): Json<Value>| async move {
            let generated = body["generated_query"].as_str().unwrap_or_default();
            Json(json!({"query":if generated.is_empty(){"fallback query"}else{generated},"strategy":"mock","accepted":true}))
        }));
    spawn(router).await
}

async fn gate(Json(body): Json<Value>) -> Json<Value> {
    let message = body["message"].as_str().unwrap_or_default();
    Json(
        json!({"use_tool":message.contains("search"),"label":if message.contains("search"){"tool"}else{"chat"},"margin":1.0}),
    )
}

async fn prefill(State(state): State<MockState>, Json(body): Json<Value>) -> Json<Value> {
    let index = state.prefills.fetch_add(1, Ordering::SeqCst) + 1;
    let state_id = format!("s{index}");
    state.prompts.lock().await.insert(
        state_id.clone(),
        body["prompt"].as_str().unwrap_or_default().to_string(),
    );
    Json(json!({"status":"ok","state":{"state_id":state_id,"branch":"root","seen_tokens":10}}))
}

async fn fork(
    State(state): State<MockState>,
    Path(parent): Path<String>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let prompt = state
        .prompts
        .lock()
        .await
        .get(&parent)
        .cloned()
        .unwrap_or_default();
    let rows = body["branches"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, branch)| {
            let state_id = format!("{parent}-b{}", index + 1);
            (state_id, branch.as_str().unwrap_or_default().to_string())
        })
        .collect::<Vec<_>>();
    let mut prompts = state.prompts.lock().await;
    for (state_id, _) in &rows {
        prompts.insert(state_id.clone(), prompt.clone());
    }
    Json(
        json!({"status":"ok","states":rows.into_iter().map(|(state_id, branch)|json!({"state_id":state_id,"branch":branch,"seen_tokens":10})).collect::<Vec<_>>()}),
    )
}

async fn continue_states(State(state): State<MockState>, Json(body): Json<Value>) -> Json<Value> {
    let prompts = state.prompts.lock().await;
    let rows = body["items"].as_array().unwrap().iter().map(|item| {
        let state_id = item["state_id"].as_str().unwrap();
        let input = item["input"].as_str().unwrap_or_default();
        let root_prompt = prompts.get(state_id).cloned().unwrap_or_default();
        let (text, stop) = if root_prompt.contains("search loop") {
            (
                r#"{"name":"web_search","arguments":{"query":"repeat"}}"#,
                "</tool_call>",
            )
        } else if input.contains("Final answer stage") {
            ("Supported fact [W1].", "</answer>")
        } else if input.ends_with("<tool_call>")
            || input.ends_with("<tool_call>{\"name\":\"")
        {
            (r#"{"name":"web_search","arguments":{"query":"primary fact"}}"#, "</tool_call>")
        } else if input.contains("<tool_result>") {
            ("Supported fact [W1].", "</answer>")
        } else if input.trim() == "Assistant:" && root_prompt.contains("bounded RWKV tool agent") {
            (r#"{"name":"web_search","arguments":{"query":"primary fact"}}"#, "</tool_call>")
        } else {
            ("mock chat answer", "\n\nUser:")
        };
        json!({"state_id":state_id,"branch":"mock","text":text,"stop_reason":stop,"token_ids":[1,2],"seen_tokens":20,"elapsed_ms":1.0})
    }).collect::<Vec<_>>();
    Json(json!({"status":"ok","results":rows}))
}

async fn stream_continue_state(Json(body): Json<Value>) -> Response {
    let state_id = body["items"][0]["state_id"].as_str().unwrap_or_default();
    let events = [
        json!({"type":"delta","state_id":state_id,"text":"mock","delta":"mock","replace":false,"output_tokens":1}),
        json!({"type":"delta","state_id":state_id,"text":"mock chat answer","delta":" chat answer","replace":false,"output_tokens":2}),
        json!({"type":"done","results":[{"state_id":state_id,"branch":"mock","text":"mock chat answer","stop_reason":"</answer>","token_ids":[1,2],"seen_tokens":20,"elapsed_ms":1.0}]}),
    ];
    let body = events
        .iter()
        .map(|event| format!("{event}\n"))
        .collect::<String>();
    Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(Body::from(body))
        .unwrap()
}

async fn release(State(state): State<MockState>, Json(body): Json<Value>) -> Json<Value> {
    let count = body["state_ids"].as_array().map_or(0, Vec::len);
    state.releases.fetch_add(count, Ordering::SeqCst);
    Json(json!({"status":"ok","released":count}))
}

async fn service(model_url: String, data_url: String, directory: &TempDir) -> AgentService {
    AgentService::new(RuntimeConfig {
        model_urls: vec![model_url],
        data_plane_url: data_url,
        session_dir: directory.path().to_path_buf(),
        ..RuntimeConfig::default()
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn direct_chat_reuses_state_and_shutdown_releases_it() {
    let (model_url, mock) = sidecar().await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = service(model_url, data_url, &directory).await;
    let first = service.run("hello", "same").await.unwrap();
    let second = service.run("again", "same").await.unwrap();
    assert_eq!(first["route"]["mode"], "direct");
    assert_eq!(second["trace"]["context"]["session_state"]["reused"], true);
    assert_eq!(mock.prefills.load(Ordering::SeqCst), 1);
    service.shutdown().await;
    assert_eq!(mock.releases.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn structured_task_spec_is_preserved_in_trace() {
    let (model_url, _mock) = sidecar().await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = service(model_url, data_url, &directory).await;
    let mut task = TaskSpec::new("hello");
    task.acceptance_criteria = vec!["answer visibly".into()];
    let result = service.run_task(task, "task-spec").await.unwrap();
    assert_eq!(result["trace"]["task_spec"]["schema_version"], 1);
    assert_eq!(
        result["trace"]["task_spec"]["acceptance_criteria"][0],
        "answer visibly"
    );
}

fn direct_stage(id: &str, objective: &str, depends_on: &[&str]) -> TaskStageSpec {
    TaskStageSpec {
        id: id.into(),
        objective: objective.into(),
        depends_on: depends_on.iter().map(|value| (*value).into()).collect(),
        acceptance_criteria: Vec::new(),
        constraints: Vec::new(),
        verification_commands: Vec::new(),
        requires_mutation: None,
    }
}

#[tokio::test]
async fn stage_controller_checkpoints_dag_and_persists_final_record() {
    let (model_url, _mock) = sidecar().await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = service(model_url, data_url, &directory).await;
    let mut task = TaskSpec::new("complete both stages");
    task.stages = vec![
        direct_stage("second", "answer second", &["first"]),
        direct_stage("first", "answer first", &[]),
    ];
    let response = service
        .run_task_with_id(task, "dag-session", Some("dag-task"))
        .await
        .unwrap();
    assert_eq!(response["task_id"], "dag-task");
    assert_eq!(response["trace"]["task_ledger"]["status"], "succeeded");
    let record = service.task("dag-task").await.unwrap();
    assert_eq!(record.status, TaskStatus::Succeeded);
    assert_eq!(record.stages[0].spec.id, "first");
    assert_eq!(record.stages[1].spec.id, "second");
    assert!(
        record
            .stages
            .iter()
            .all(|stage| stage.status == StageStatus::Succeeded && stage.attempts == 1)
    );
}

#[tokio::test]
async fn startup_recovery_resumes_only_the_interrupted_stage() {
    let (model_url, _mock) = sidecar().await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let ledger = TaskLedger::new(directory.path().join("task-ledger"))
        .await
        .unwrap();
    let mut task = TaskSpec::new("recover task");
    task.stages = vec![
        direct_stage("first", "answer first", &[]),
        direct_stage("second", "answer second", &["first"]),
    ];
    ledger
        .create("recover-session", task, Some("recover-task"))
        .await
        .unwrap();
    ledger.start_task("recover-task").await.unwrap();
    ledger.start_stage("recover-task", "first").await.unwrap();

    let service = service(model_url, data_url, &directory).await;
    let interrupted = service.task("recover-task").await.unwrap();
    assert_eq!(interrupted.status, TaskStatus::Interrupted);
    let response = service.resume_task("recover-task").await.unwrap();
    assert_eq!(response["trace"]["task_ledger"]["recovery_count"], 1);
    let completed = service.task("recover-task").await.unwrap();
    assert_eq!(completed.status, TaskStatus::Succeeded);
    assert_eq!(completed.stages[0].attempts, 2);
    assert_eq!(completed.stages[1].attempts, 1);
}

#[tokio::test]
async fn direct_chat_stream_forwards_decode_deltas_then_final_trace() {
    let (model_url, _mock) = sidecar().await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = service(model_url, data_url, &directory).await;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(16);
    service
        .run_with_workspace_stream("hello", "stream", None, sender)
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        events.push(event);
    }
    let deltas = events
        .iter()
        .filter(|event| event["type"] == "delta")
        .map(|event| event["text"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    assert_eq!(deltas, ["mock", "mock chat answer"]);
    let final_event = events
        .iter()
        .find(|event| event["type"] == "final")
        .unwrap();
    assert_eq!(final_event["response"]["answer"], "mock chat answer");
    assert_eq!(
        final_event["response"]["trace"]["answer_completion"]["stop"],
        "</answer>"
    );
}

#[tokio::test]
async fn tool_loop_executes_observes_validates_and_releases() {
    let (model_url, mock) = sidecar().await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = service(model_url, data_url, &directory).await;
    let result = service.run("search current fact", "tool").await.unwrap();
    assert_eq!(result["status"], "ok");
    assert_eq!(result["route"]["mode"], "tool_loop");
    assert_eq!(result["route"]["steps"], 1);
    assert_eq!(result["answer"], "Supported fact [W1].");
    assert_eq!(mock.releases.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn repeated_tool_failure_stops_before_budget_and_preserves_release_event() {
    let (model_url, mock) = sidecar().await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = service(model_url, data_url, &directory).await;

    let result = service.run("search loop", "tool-error").await.unwrap();

    assert_eq!(result["status"], "error");
    assert_eq!(result["error_code"], "no_progress");
    assert_eq!(result["route"]["steps"], 3);
    assert_eq!(
        result["trace"]["agent"]["tool_steps"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert!(
        result["trace"]["agent"]["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|event| event["type"] == "state_released" && event["success"] == true)
    );
    assert_eq!(mock.releases.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn research_forks_parallel_states_reduces_and_releases_all() {
    let (model_url, mock) = sidecar().await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = service(model_url, data_url, &directory).await;
    let result = service
        .research("research fact", "research", 2, 2)
        .await
        .unwrap();
    assert_eq!(result["status"], "ok");
    assert_eq!(result["route"]["branch_width"], 2);
    assert_eq!(result["trace"]["rounds"].as_array().unwrap().len(), 2);
    assert_eq!(mock.releases.load(Ordering::SeqCst), 3);
}
