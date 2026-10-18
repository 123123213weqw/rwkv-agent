use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::{Request, StatusCode, header};
use axum::routing::{get, post};
use axum::{Json, Router};
use rwkv_agent_runtime::{
    AgentService, DebugTraceConfig, DebugTraceMode, RuntimeConfig, SERVICE_API_VERSION,
};
use rwkv_agent_server::router;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use tower::ServiceExt;

#[derive(Clone, Default)]
struct MockState {
    allocated: Arc<AtomicUsize>,
    prefills: Arc<AtomicUsize>,
    releases: Arc<AtomicUsize>,
    continue_delay_ms: Arc<AtomicUsize>,
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
        .route("/health", get(sidecar_health))
        .route("/v1/gate/tool", post(gate))
        .route("/v1/states/prefill", post(prefill))
        .route("/v1/states/batch_continue", post(continue_states))
        .route("/v1/states/release", post(release))
        .with_state(state.clone());
    (spawn(router).await, state)
}

async fn sidecar_health(State(state): State<MockState>) -> Json<Value> {
    Json(json!({
        "status":"ready",
        "model":"mock-rwkv",
        "persistent_states":{
            "capacity":8,
            "allocated":state.allocated.load(Ordering::SeqCst),
            "free":8_usize.saturating_sub(state.allocated.load(Ordering::SeqCst))
        }
    }))
}

async fn gate(Json(body): Json<Value>) -> Json<Value> {
    let use_tool = body["message"]
        .as_str()
        .is_some_and(|message| message.contains("search"));
    Json(json!({
        "use_tool":use_tool,
        "label":if use_tool {"tool"} else {"chat"},
        "margin":1.0
    }))
}

async fn prefill(State(state): State<MockState>, Json(body): Json<Value>) -> Json<Value> {
    let index = state.prefills.fetch_add(1, Ordering::SeqCst) + 1;
    state.allocated.fetch_add(1, Ordering::SeqCst);
    let state_id = format!("state-{index}");
    state.prompts.lock().await.insert(
        state_id.clone(),
        body["prompt"].as_str().unwrap_or_default().to_string(),
    );
    Json(json!({
        "status":"ok",
        "state":{"state_id":state_id,"branch":"root","seen_tokens":10}
    }))
}

async fn continue_states(State(state): State<MockState>, Json(body): Json<Value>) -> Json<Value> {
    let delay_ms = state.continue_delay_ms.load(Ordering::SeqCst);
    if delay_ms > 0 {
        tokio::time::sleep(Duration::from_millis(delay_ms as u64)).await;
    }
    let prompts = state.prompts.lock().await;
    let results = body["items"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|item| {
            let state_id = item["state_id"].as_str().unwrap_or_default();
            let input = item["input"].as_str().unwrap_or_default();
            let root_prompt = prompts.get(state_id).map_or("", String::as_str);
            let (text, stop_reason) = if input.contains("Final answer stage")
                || input.contains("<tool_result>")
            {
                ("Supported fact [W1].", "</answer>")
            } else if input.ends_with("<tool_call>")
                || input.ends_with("<tool_call>{\"name\":\"")
                || (input.trim() == "Assistant:" && root_prompt.contains("bounded RWKV tool agent"))
            {
                (
                    r#"{"name":"web_search","arguments":{"query":"primary fact"}}"#,
                    "</tool_call>",
                )
            } else {
                ("mock answer", "</answer>")
            };
            json!({
                "state_id":state_id,
                "branch":"mock",
                "text":text,
                "stop_reason":stop_reason,
                "token_ids":[1,2],
                "seen_tokens":20,
                "elapsed_ms":1.0
            })
        })
        .collect::<Vec<_>>();
    Json(json!({"status":"ok","results":results}))
}

async fn release(State(state): State<MockState>, Json(body): Json<Value>) -> Json<Value> {
    let count = body["state_ids"].as_array().map_or(0, Vec::len);
    state.releases.fetch_add(count, Ordering::SeqCst);
    state.allocated.fetch_sub(count, Ordering::SeqCst);
    Json(json!({"status":"ok","released":count}))
}

async fn data_plane() -> String {
    let router = Router::new()
        .route(
            "/health",
            get(|| async {
                Json(json!({
                    "status":"ready",
                    "tools":["web_search","knowledge_search","long_text_qa"]
                }))
            }),
        )
        .route(
            "/v1/session/text/status",
            post(|| async { Json(json!({"status":"empty","active":false})) }),
        )
        .route(
            "/v1/tools/call",
            post(|Json(body): Json<Value>| async move {
                Json(json!({
                    "status":"ok",
                    "tool":body["name"],
                    "evidence":[{
                        "id":"W1",
                        "title":"Primary",
                        "content":"Supported fact.",
                        "uri":"https://example.test/fact"
                    }]
                }))
            }),
        )
        .route(
            "/v1/answers/validate",
            post(|Json(body): Json<Value>| async move {
                Json(json!({"valid":true,"answer":body["answer"],"errors":[]}))
            }),
        );
    spawn(router).await
}

async fn response_json(response: axum::response::Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap()).unwrap()
}

#[tokio::test]
async fn canonical_http_tasks_are_idempotent_owner_scoped_and_release_state() {
    let (model_url, mock) = sidecar().await;
    let data_plane_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = AgentService::new(RuntimeConfig {
        runtime_revision: "service-pipeline-test".into(),
        model_urls: vec![model_url],
        data_plane_url,
        session_dir: directory.path().to_path_buf(),
        debug_trace: DebugTraceConfig {
            mode: DebugTraceMode::Full,
            directory: directory.path().join("debug-traces"),
            api_enabled: true,
            ..DebugTraceConfig::default()
        },
        ..RuntimeConfig::default()
    })
    .await
    .unwrap();
    let app = router(service);
    let ready = app
        .clone()
        .oneshot(Request::get("/ready").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(ready.status(), StatusCode::OK);
    let ready = response_json(ready).await;
    assert_eq!(ready["components"]["state_capacity"]["status"], "ready");
    assert_eq!(
        ready["configuration"]["runtime_revision"],
        "service-pipeline-test"
    );
    let request_body = json!({
        "api_version":SERVICE_API_VERSION,
        "request_id":"request-e2e",
        "owner_id":"owner-e2e",
        "session_id":"session-e2e",
        "task_id":"task-e2e",
        "task_spec":{
            "schema_version":1,
            "objective":"search current fact",
            "acceptance_criteria":["return a cited fact"]
        }
    });

    let run = app
        .clone()
        .oneshot(
            Request::post("/v1/tasks")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(run.status(), StatusCode::OK);
    let run = response_json(run).await;
    assert_eq!(run["api_version"], SERVICE_API_VERSION);
    assert_eq!(run["request_id"], "request-e2e");
    assert_eq!(run["owner_id"], "owner-e2e");
    assert_eq!(run["task_id"], "task-e2e");
    assert_eq!(run["status"], "ok");
    assert_eq!(run["answer"], "Supported fact [W1].");
    assert_eq!(run["debug_capture"]["status"], "complete");
    let trace_id = run["trace_id"].as_str().unwrap().to_string();
    assert_eq!(mock.prefills.load(Ordering::SeqCst), 1);
    assert_eq!(mock.releases.load(Ordering::SeqCst), 1);
    assert_eq!(mock.allocated.load(Ordering::SeqCst), 0);

    let replay = app
        .clone()
        .oneshot(
            Request::post("/v1/tasks")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(replay.status(), StatusCode::OK);
    let replay = response_json(replay).await;
    assert_eq!(replay["task_id"], "task-e2e");
    assert_eq!(mock.prefills.load(Ordering::SeqCst), 1);

    let record = app
        .clone()
        .oneshot(
            Request::get(concat!(
                "/v1/tasks/task-e2e?api_version=rwkv-agent.service.v1",
                "&request_id=inspect-e2e&owner_id=owner-e2e"
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(record.status(), StatusCode::OK);
    let record = response_json(record).await;
    assert_eq!(record["task"]["status"], "succeeded");
    assert!(record["task"].get("task_spec").is_none());
    assert!(record["task"].get("final_response").is_none());
    assert_eq!(
        record["task"]["task_spec_summary"]["objective_chars"],
        "search current fact".chars().count()
    );
    assert_eq!(record["task"]["stages"][0]["status"], "succeeded");
    assert_eq!(record["task"]["stages"][0]["attempts"], 1);
    let events = record["task"]["events"].as_array().unwrap();
    assert!(events.windows(2).all(|pair| {
        pair[0]["sequence"].as_u64().unwrap() < pair[1]["sequence"].as_u64().unwrap()
    }));

    let traces = app
        .clone()
        .oneshot(
            Request::get(concat!(
                "/v1/debug/traces?api_version=rwkv-agent.service.v1",
                "&request_id=trace-list&owner_id=owner-e2e&task_id=task-e2e&limit=10"
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(traces.status(), StatusCode::OK);
    let traces = response_json(traces).await;
    assert_eq!(traces["page"]["traces"].as_array().unwrap().len(), 1);
    assert_eq!(traces["page"]["traces"][0]["trace_id"], trace_id);

    let trace_events = app
        .clone()
        .oneshot(
            Request::get(format!(
                "/v1/debug/traces/{trace_id}/events?api_version={SERVICE_API_VERSION}&request_id=trace-events&owner_id=owner-e2e&after_sequence=1&limit=2"
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(trace_events.status(), StatusCode::OK);
    let trace_events = response_json(trace_events).await;
    assert_eq!(trace_events["events"].as_array().unwrap().len(), 2);
    assert!(trace_events["events"][0]["sequence"].as_u64().unwrap() > 1);

    let model_file = app
        .clone()
        .oneshot(
            Request::get(format!(
                "/v1/debug/traces/{trace_id}/files/model?api_version={SERVICE_API_VERSION}&request_id=trace-file&owner_id=owner-e2e"
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(model_file.status(), StatusCode::OK);
    assert!(
        String::from_utf8(
            to_bytes(model_file.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .to_vec()
        )
        .unwrap()
        .contains("primary fact")
    );

    let ranged_file = app
        .clone()
        .oneshot(
            Request::get(format!(
                "/v1/debug/traces/{trace_id}/files/model?api_version={SERVICE_API_VERSION}&request_id=trace-range&owner_id=owner-e2e"
            ))
            .header(header::RANGE, "bytes=0-10")
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ranged_file.status(), StatusCode::RANGE_NOT_SATISFIABLE);

    let trace_denied = app
        .clone()
        .oneshot(
            Request::get(format!(
                "/v1/debug/traces/{trace_id}?api_version={SERVICE_API_VERSION}&request_id=trace-denied&owner_id=owner-other"
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(trace_denied.status(), StatusCode::NOT_FOUND);

    let direct_tool = app
        .clone()
        .oneshot(
            Request::post("/v1/tools/call")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "api_version":SERVICE_API_VERSION,
                        "request_id":"direct-tool-request",
                        "owner_id":"owner-e2e",
                        "session_id":"session-e2e",
                        "name":"web_search",
                        "arguments":{"query":"direct fact"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(direct_tool.status(), StatusCode::OK);
    let direct_tool = response_json(direct_tool).await;
    assert_eq!(direct_tool["debug_capture"]["status"], "complete");
    assert!(direct_tool["trace_id"].as_str().is_some());

    let denied = app
        .oneshot(
            Request::get(concat!(
                "/v1/tasks/task-e2e?api_version=rwkv-agent.service.v1",
                "&request_id=inspect-denied&owner_id=owner-other"
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::NOT_FOUND);
    let denied = response_json(denied).await;
    assert_eq!(denied["error_detail"]["code"], "not_found");
}

#[tokio::test]
async fn dropping_stream_cancels_the_task_and_releases_open_state() {
    let (model_url, mock) = sidecar().await;
    mock.continue_delay_ms.store(150, Ordering::SeqCst);
    let data_plane_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = AgentService::new(RuntimeConfig {
        runtime_revision: "disconnect-test".into(),
        model_urls: vec![model_url],
        data_plane_url,
        session_dir: directory.path().to_path_buf(),
        debug_trace: DebugTraceConfig {
            mode: DebugTraceMode::Full,
            directory: directory.path().join("disconnect-debug-traces"),
            api_enabled: true,
            ..DebugTraceConfig::default()
        },
        ..RuntimeConfig::default()
    })
    .await
    .unwrap();
    let app = router(service);
    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/tasks/stream")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "api_version":SERVICE_API_VERSION,
                        "request_id":"request-disconnect",
                        "owner_id":"owner-disconnect",
                        "session_id":"session-disconnect",
                        "task_id":"task-disconnect",
                        "task_spec":{"objective":"search current fact"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut stream = response.into_body().into_data_stream();
    let saw_tool_phase = tokio::time::timeout(Duration::from_secs(2), async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            for line in String::from_utf8_lossy(&chunk).lines() {
                let event: Value = serde_json::from_str(line).unwrap();
                if event.get("phase").and_then(Value::as_str) == Some("tool") {
                    return true;
                }
            }
        }
        false
    })
    .await
    .unwrap();
    assert!(saw_tool_phase);
    tokio::time::timeout(Duration::from_secs(2), async {
        while mock.allocated.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    drop(stream);

    tokio::time::timeout(Duration::from_secs(2), async {
        while mock.releases.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(mock.allocated.load(Ordering::SeqCst), 0);
    assert_eq!(mock.releases.load(Ordering::SeqCst), 1);

    let record = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let response = app
                .clone()
                .oneshot(
                    Request::get(concat!(
                        "/v1/tasks/task-disconnect?api_version=rwkv-agent.service.v1",
                        "&request_id=inspect-disconnect&owner_id=owner-disconnect"
                    ))
                    .body(Body::empty())
                    .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let record = response_json(response).await;
            if record["task"]["debug_capture"]["status"] == "complete" {
                break record;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(record["task"]["status"], "cancelled");
    assert_eq!(record["task"]["stages"][0]["status"], "cancelled");
    let trace_id = record["task"]["trace_id"].as_str().unwrap();
    let manifest = app
        .oneshot(
            Request::get(format!(
                "/v1/debug/traces/{trace_id}?api_version={SERVICE_API_VERSION}&request_id=inspect-disconnect-trace&owner_id=owner-disconnect"
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(manifest.status(), StatusCode::OK);
    assert_eq!(response_json(manifest).await["manifest"]["complete"], true);
}
