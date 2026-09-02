use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use rwkv_agent_runtime::{
    AgentService, CloudModelRef, CloudPluginConfig, CloudPluginFallback, CloudStatePlacement,
    DebugTraceConfig, DebugTraceFileKind, DebugTraceFilter, DebugTraceMode, PrivacyClass,
    RequestIdentity, RuntimeConfig, SERVICE_API_VERSION, StageStatus, TaskLedger, TaskSpec,
    TaskStageSpec, TaskStatus,
};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::sync::Mutex;

#[derive(Clone, Default)]
struct MockState {
    prefills: Arc<AtomicUsize>,
    snapshots: Arc<AtomicUsize>,
    restores: Arc<AtomicUsize>,
    continuations: Arc<AtomicUsize>,
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
        .route(
            "/health",
            get(|| async { Json(json!({"status":"ready","persistent_state":{"allocated":0}})) }),
        )
        .route("/v1/gate/tool", post(gate))
        .route("/v1/states/prefill", post(prefill))
        .route("/v1/states/{state_id}/snapshot", post(snapshot_state))
        .route("/v1/states/restore", post(restore_state))
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

#[derive(Clone)]
struct StatePoolMock {
    worker_endpoint: String,
    worker_zone: String,
    payloads: Arc<Mutex<HashMap<u64, String>>>,
    usage: Arc<Mutex<Vec<Value>>>,
    releases: Arc<AtomicUsize>,
    fail_snapshots: Arc<AtomicUsize>,
    fail_usage: Arc<AtomicUsize>,
}

async fn statepool_plugin(worker_endpoint: String) -> (String, StatePoolMock) {
    statepool_plugin_in_zone(worker_endpoint, "cloud").await
}

async fn statepool_plugin_in_zone(
    worker_endpoint: String,
    worker_zone: &str,
) -> (String, StatePoolMock) {
    async fn handshake(Json(request): Json<Value>) -> Json<Value> {
        let required = request["required_capabilities"].as_array().unwrap();
        assert!(required.contains(&json!("leases")));
        assert!(required.contains(&json!("state_lifecycle")));
        Json(json!({
            "contract_version":"statepool-plugin.v1",
            "plugin":"statepool-cloud",
            "plugin_version":"mock",
            "capabilities":["placement","leases","state_lifecycle","finops"]
        }))
    }

    async fn plan(State(state): State<StatePoolMock>, Json(request): Json<Value>) -> Json<Value> {
        Json(json!({
            "contract_version":"statepool-execution-plan.v1",
            "decision_id":format!("decision-{}", request["request_id"].as_str().unwrap()),
            "request_id":request["request_id"],
            "mode":if state.worker_zone == "local" {"local"} else {"remote"},
            "worker_id":"worker-mock",
            "worker_zone":state.worker_zone,
            "endpoint":state.worker_endpoint,
            "state_action":if request["state_ref"].is_null() {"none"} else {"restore"},
            "reason_code":"state_affinity",
            "lease_required":true,
            "estimated_queue_ms":0.0,
            "estimated_restore_ms":0.0,
            "estimated_cost":null,
            "fallback":"fail_closed"
        }))
    }

    async fn acquire(Json(request): Json<Value>) -> Json<Value> {
        let version = request["expected_state_version"].as_u64().unwrap();
        Json(json!({
            "contract_version":"statepool-lease.v1",
            "lease_id":format!("lease-{version}"),
            "session_id":request["session_id"],
            "owner_id":request["owner_id"],
            "holder_id":request["holder_id"],
            "fencing_token":version+1,
            "expected_state_version":version,
            "expires_at_ms":9_999_999_999_999_u64
        }))
    }

    async fn renew(Json(request): Json<Value>) -> Json<Value> {
        Json(request["lease"].clone())
    }

    async fn release_lease(
        State(state): State<StatePoolMock>,
        Json(_request): Json<Value>,
    ) -> axum::http::StatusCode {
        state.releases.fetch_add(1, Ordering::SeqCst);
        axum::http::StatusCode::NO_CONTENT
    }

    async fn snapshot(
        State(state): State<StatePoolMock>,
        Json(request): Json<Value>,
    ) -> (axum::http::StatusCode, Json<Value>) {
        if state
            .fail_snapshots
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |value| {
                value.checked_sub(1)
            })
            .is_ok()
        {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error":"injected snapshot failure"})),
            );
        }
        let version = request["expected_state_version"].as_u64().unwrap() + 1;
        state.payloads.lock().await.insert(
            version,
            request["payload_base64"].as_str().unwrap().to_string(),
        );
        (
            axum::http::StatusCode::OK,
            Json(json!({
                "contract_version":"statepool-state-reference.v1",
                "state_id":format!("durable-{version}"),
                "session_id":request["lease"]["session_id"],
                "owner_id":request["lease"]["owner_id"],
                "version":version,
                "fencing_token":request["lease"]["fencing_token"],
                "provider_mode":"rwkv_recurrent",
                "model_ref":request["model_ref"],
                "placement":request["target_tier"],
                "worker_id":null,
                "object_uri":format!("s3://mock/durable-{version}"),
                "checksum":request["expected_checksum"],
                "size_bytes":BASE64_STANDARD.decode(request["payload_base64"].as_str().unwrap()).unwrap().len(),
                "atomic":true,
                "created_at_ms":1,
                "last_active_at_ms":1,
                "encryption":null
            })),
        )
    }

    async fn restore(
        State(state): State<StatePoolMock>,
        Json(request): Json<Value>,
    ) -> Json<Value> {
        let version = request["state_ref"]["version"].as_u64().unwrap();
        let payload = state.payloads.lock().await.get(&version).cloned().unwrap();
        Json(json!({
            "contract_version":"statepool-restore-response.v1",
            "state_ref":request["state_ref"],
            "payload_base64":payload
        }))
    }

    async fn record_usage(
        State(state): State<StatePoolMock>,
        Json(record): Json<Value>,
    ) -> axum::http::StatusCode {
        assert_eq!(record["contract_version"], "statepool-usage-record.v1");
        assert_eq!(record["worker_id"], "worker-mock");
        assert_eq!(record["zone"], state.worker_zone);
        assert_eq!(record["outcome"], "succeeded");
        if state
            .fail_usage
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |value| {
                value.checked_sub(1)
            })
            .is_ok()
        {
            return axum::http::StatusCode::SERVICE_UNAVAILABLE;
        }
        state.usage.lock().await.push(record);
        axum::http::StatusCode::ACCEPTED
    }

    let state = StatePoolMock {
        worker_endpoint,
        worker_zone: worker_zone.into(),
        payloads: Arc::new(Mutex::new(HashMap::new())),
        usage: Arc::new(Mutex::new(Vec::new())),
        releases: Arc::new(AtomicUsize::new(0)),
        fail_snapshots: Arc::new(AtomicUsize::new(0)),
        fail_usage: Arc::new(AtomicUsize::new(0)),
    };
    let router = Router::new()
        .route("/plugin/v1/handshake", post(handshake))
        .route("/plugin/v1/plan", post(plan))
        .route("/plugin/v1/leases/acquire", post(acquire))
        .route("/plugin/v1/leases/renew", post(renew))
        .route("/plugin/v1/leases/release", post(release_lease))
        .route("/plugin/v1/states/snapshot", post(snapshot))
        .route("/plugin/v1/states/restore", post(restore))
        .route("/plugin/v1/usage", post(record_usage))
        .with_state(state.clone());
    (spawn(router).await, state)
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

fn sha256_checksum(payload: &[u8]) -> String {
    let digest = ring::digest::digest(&ring::digest::SHA256, payload);
    format!(
        "sha256:{}",
        digest
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

async fn snapshot_state(
    State(state): State<MockState>,
    Path(state_id): Path<String>,
    Json(body): Json<Value>,
) -> Json<Value> {
    state.snapshots.fetch_add(1, Ordering::SeqCst);
    let payload = format!("snapshot:{state_id}").into_bytes();
    Json(json!({
        "status":"ok",
        "checkpoint":{
            "checkpoint_id":format!("checkpoint-{state_id}"),
            "model_ref":body["model_ref"],
            "provider_mode":"rwkv_recurrent",
            "placement":"cpu",
            "checksum":sha256_checksum(&payload),
            "size_bytes":payload.len(),
            "atomic":true,
            "seen_tokens":20
        },
        "payload_base64":BASE64_STANDARD.encode(payload)
    }))
}

async fn restore_state(State(state): State<MockState>, Json(body): Json<Value>) -> Json<Value> {
    let payload = BASE64_STANDARD
        .decode(body["payload_base64"].as_str().unwrap())
        .unwrap();
    assert_eq!(body["checksum"], sha256_checksum(&payload));
    let index = state.restores.fetch_add(1, Ordering::SeqCst) + 1;
    let state_id = format!("restored-{index}");
    state
        .prompts
        .lock()
        .await
        .insert(state_id.clone(), "restored-chat".into());
    Json(json!({
        "status":"ok",
        "state":{
            "state_id":state_id,
            "owner_id":body["owner_id"],
            "branch":"restored",
            "seen_tokens":20
        }
    }))
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
    state.continuations.fetch_add(1, Ordering::SeqCst);
    let delay_ms = state.continue_delay_ms.load(Ordering::SeqCst);
    if delay_ms > 0 {
        tokio::time::sleep(Duration::from_millis(delay_ms as u64)).await;
    }
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
    service.shutdown().await.unwrap();
    assert_eq!(mock.releases.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn direct_chat_persists_releases_restores_and_advances_fenced_state() {
    let (model_url, model) = sidecar().await;
    let (plugin_url, plugin) = statepool_plugin(model_url.clone()).await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let model_ref = CloudModelRef {
        model_id: "rwkv7".into(),
        revision: "revision".into(),
        tokenizer: "tokenizer".into(),
        state_abi: "rwkv7-state-v1".into(),
    };
    let service = AgentService::new(RuntimeConfig {
        model_urls: vec![model_url],
        data_plane_url: data_url,
        session_dir: directory.path().to_path_buf(),
        cloud_plugin: CloudPluginConfig {
            enabled: true,
            endpoint: plugin_url,
            fallback: CloudPluginFallback::FailClosed,
            default_privacy: PrivacyClass::CloudAllowed,
            model_ref: Some(model_ref),
            state_lifecycle: true,
            state_target_tier: CloudStatePlacement::Cold,
            ..CloudPluginConfig::default()
        },
        ..RuntimeConfig::default()
    })
    .await
    .unwrap();

    let first = service.run("hello", "durable").await.unwrap();
    assert_eq!(first["trace"]["finops"]["status"], "accepted");
    assert_eq!(
        first["trace"]["context"]["session_state"]["residency"],
        "durable"
    );
    assert_eq!(
        first["trace"]["context"]["session_state"]["state_version"],
        1
    );
    assert_eq!(model.prefills.load(Ordering::SeqCst), 1);
    assert_eq!(model.snapshots.load(Ordering::SeqCst), 1);
    assert_eq!(model.releases.load(Ordering::SeqCst), 1);

    let second = service.run("again", "durable").await.unwrap();
    assert_eq!(second["trace"]["context"]["session_state"]["reused"], true);
    assert_eq!(second["trace"]["placement"]["state_action"], "restore");
    assert_eq!(
        second["trace"]["context"]["session_state"]["residency"],
        "durable"
    );
    assert_eq!(
        second["trace"]["context"]["session_state"]["state_version"],
        2
    );
    assert_eq!(model.prefills.load(Ordering::SeqCst), 1);
    assert_eq!(model.restores.load(Ordering::SeqCst), 1);
    assert_eq!(model.snapshots.load(Ordering::SeqCst), 2);
    assert_eq!(model.releases.load(Ordering::SeqCst), 2);
    assert_eq!(plugin.releases.load(Ordering::SeqCst), 2);
    let usage = plugin.usage.lock().await.clone();
    assert_eq!(usage.len(), 2);
    assert_eq!(usage[0]["operation"], "create");
    assert!(usage[0]["state_tier_before"].is_null());
    assert_eq!(usage[0]["state_tier_after"], "cold");
    assert_eq!(usage[0]["metrics"]["state_bytes_read"], 0);
    assert!(usage[0]["metrics"]["state_bytes_written"].as_u64().unwrap() > 0);
    assert_eq!(usage[1]["operation"], "continue");
    assert_eq!(usage[1]["state_tier_before"], "cold");
    assert_eq!(usage[1]["state_tier_after"], "cold");
    assert!(
        usage[1]["metrics"]["prefill_tokens_avoided"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(usage[1]["metrics"]["state_bytes_read"].as_u64().unwrap() > 0);
    assert!(usage[1]["metrics"]["state_bytes_written"].as_u64().unwrap() > 0);
    assert!(usage[1]["metrics"]["restore_ms"].as_f64().unwrap() >= 0.0);
    assert!(usage[1]["metrics"]["snapshot_ms"].as_f64().unwrap() >= 0.0);

    let readiness = service.readiness().await;
    assert_eq!(readiness["context"]["session_state"]["allocated"], 0);
    assert_eq!(readiness["context"]["session_state"]["durable"], 1);
    service.shutdown().await.unwrap();
    assert_eq!(model.releases.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn plugin_selected_local_worker_endpoint_is_used_for_create_and_restore() {
    let (default_url, default_model) = sidecar().await;
    let (selected_url, selected_model) = sidecar().await;
    let (plugin_url, plugin) = statepool_plugin_in_zone(selected_url, "local").await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = AgentService::new(RuntimeConfig {
        model_urls: vec![default_url],
        data_plane_url: data_url,
        session_dir: directory.path().to_path_buf(),
        cloud_plugin: CloudPluginConfig {
            enabled: true,
            endpoint: plugin_url,
            fallback: CloudPluginFallback::FailClosed,
            default_privacy: PrivacyClass::CloudAllowed,
            model_ref: Some(CloudModelRef {
                model_id: "rwkv7".into(),
                revision: "revision".into(),
                tokenizer: "tokenizer".into(),
                state_abi: "rwkv7-state-v1".into(),
            }),
            state_lifecycle: true,
            state_target_tier: CloudStatePlacement::Cold,
            ..CloudPluginConfig::default()
        },
        ..RuntimeConfig::default()
    })
    .await
    .unwrap();

    let first = service.run("hello", "local-placement").await.unwrap();
    let second = service.run("again", "local-placement").await.unwrap();
    assert_eq!(first["trace"]["placement"]["mode"], "local");
    assert_eq!(second["trace"]["placement"]["mode"], "local");
    assert_eq!(second["trace"]["placement"]["state_action"], "restore");
    assert_eq!(default_model.prefills.load(Ordering::SeqCst), 0);
    assert_eq!(default_model.restores.load(Ordering::SeqCst), 0);
    assert_eq!(selected_model.prefills.load(Ordering::SeqCst), 1);
    assert_eq!(selected_model.restores.load(Ordering::SeqCst), 1);
    assert_eq!(selected_model.snapshots.load(Ordering::SeqCst), 2);
    let usage = plugin.usage.lock().await.clone();
    assert_eq!(usage.len(), 2);
    assert!(usage.iter().all(|record| record["zone"] == "local"));
    service.shutdown().await.unwrap();
}

#[tokio::test]
async fn uncertain_lifecycle_commit_blocks_automatic_double_execution() {
    let (model_url, model) = sidecar().await;
    let (plugin_url, plugin) = statepool_plugin(model_url.clone()).await;
    plugin.fail_snapshots.store(1, Ordering::SeqCst);
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = AgentService::new(RuntimeConfig {
        model_urls: vec![model_url],
        data_plane_url: data_url,
        session_dir: directory.path().to_path_buf(),
        cloud_plugin: CloudPluginConfig {
            enabled: true,
            endpoint: plugin_url,
            fallback: CloudPluginFallback::FailClosed,
            default_privacy: PrivacyClass::CloudAllowed,
            model_ref: Some(CloudModelRef {
                model_id: "rwkv7".into(),
                revision: "revision".into(),
                tokenizer: "tokenizer".into(),
                state_abi: "rwkv7-state-v1".into(),
            }),
            state_lifecycle: true,
            ..CloudPluginConfig::default()
        },
        ..RuntimeConfig::default()
    })
    .await
    .unwrap();

    let first = service.run("hello", "uncertain").await.unwrap();
    assert_eq!(
        first["trace"]["context"]["session_state"]["residency"],
        "blocked_hot"
    );
    assert!(
        first["trace"]["context"]["session_state"]["persistence_error"]
            .as_str()
            .unwrap()
            .contains("snapshot")
    );
    assert_eq!(model.continuations.load(Ordering::SeqCst), 1);

    let error = service.run("again", "uncertain").await.unwrap_err();
    assert!(error.contains("requires reconciliation"));
    assert_eq!(model.continuations.load(Ordering::SeqCst), 1);
    assert_eq!(model.prefills.load(Ordering::SeqCst), 1);
    assert_eq!(model.restores.load(Ordering::SeqCst), 0);
    assert!(service.shutdown().await.unwrap_err().contains("unresolved"));
}

#[tokio::test]
async fn finops_reporting_failure_never_turns_successful_inference_into_failure() {
    let (model_url, model) = sidecar().await;
    let (plugin_url, plugin) = statepool_plugin(model_url.clone()).await;
    plugin.fail_usage.store(1, Ordering::SeqCst);
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = AgentService::new(RuntimeConfig {
        model_urls: vec![model_url],
        data_plane_url: data_url,
        session_dir: directory.path().to_path_buf(),
        cloud_plugin: CloudPluginConfig {
            enabled: true,
            endpoint: plugin_url,
            fallback: CloudPluginFallback::FailClosed,
            default_privacy: PrivacyClass::CloudAllowed,
            model_ref: Some(CloudModelRef {
                model_id: "rwkv7".into(),
                revision: "revision".into(),
                tokenizer: "tokenizer".into(),
                state_abi: "rwkv7-state-v1".into(),
            }),
            state_lifecycle: true,
            ..CloudPluginConfig::default()
        },
        ..RuntimeConfig::default()
    })
    .await
    .unwrap();

    let response = service.run("hello", "finops-failure").await.unwrap();
    assert_eq!(response["status"], "ok");
    assert_eq!(response["trace"]["finops"]["status"], "report_failed");
    assert!(
        response["trace"]["finops"]["error"]
            .as_str()
            .unwrap()
            .contains("503")
    );
    assert_eq!(model.continuations.load(Ordering::SeqCst), 1);
    assert_eq!(plugin.usage.lock().await.len(), 0);
    service.shutdown().await.unwrap();
}

#[tokio::test]
async fn direct_chat_deadline_releases_the_open_state() {
    let (model_url, mock) = sidecar().await;
    mock.continue_delay_ms.store(100, Ordering::SeqCst);
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = AgentService::new(RuntimeConfig {
        model_urls: vec![model_url],
        data_plane_url: data_url,
        session_dir: directory.path().to_path_buf(),
        max_run_elapsed: Duration::from_millis(20),
        debug_trace: DebugTraceConfig {
            mode: DebugTraceMode::Full,
            directory: directory.path().join("deadline-debug-traces"),
            api_enabled: true,
            ..DebugTraceConfig::default()
        },
        ..RuntimeConfig::default()
    })
    .await
    .unwrap();
    let error = service.run("hello", "direct-timeout").await.unwrap_err();
    assert!(error.contains("deadline exceeded"));
    assert_eq!(mock.releases.load(Ordering::SeqCst), 1);
    let trace_id = error
        .split("debug_trace_id=")
        .nth(1)
        .unwrap()
        .split(';')
        .next()
        .unwrap();
    let events = service
        .debug_trace_events_for_owner(trace_id, "session:direct-timeout", 0, 100)
        .await
        .unwrap();
    assert_eq!(events.last().unwrap().event_type, "trace_finished");
    assert!(events.iter().any(|event| {
        event.event_type == "state_released"
            && event.payload.get("success").and_then(Value::as_bool) == Some(true)
    }));
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
    let service = AgentService::new(RuntimeConfig {
        model_urls: vec![model_url],
        data_plane_url: data_url,
        session_dir: directory.path().join("sessions"),
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
    let identity = RequestIdentity::resolve(
        SERVICE_API_VERSION,
        Some("stream-request"),
        Some("stream-owner"),
        "stream",
        String::new,
    )
    .unwrap();
    let (sender, mut receiver) = tokio::sync::mpsc::channel(16);
    let response = service
        .run_task_stream_with_identity(
            TaskSpec::new("hello"),
            &identity,
            Some("stream-task"),
            sender,
        )
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        events.push(event);
    }
    let created = events
        .iter()
        .find(|event| event["type"] == "task_created")
        .unwrap();
    assert_eq!(created["trace_id"], response["trace_id"]);
    assert_eq!(created["debug_capture"]["status"], "active");
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
    assert_eq!(response["debug_capture"]["status"], "complete");
    let stream_file = service
        .debug_trace_file_for_owner(
            response["trace_id"].as_str().unwrap(),
            "stream-owner",
            DebugTraceFileKind::Stream,
        )
        .await
        .unwrap();
    let stream_file = String::from_utf8(stream_file).unwrap();
    assert!(stream_file.contains("\"event_type\":\"delta\""));
    assert!(stream_file.contains("\"event_type\":\"final\""));
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
    assert!(result.get("trace_id").is_none());
    assert!(result.get("debug_capture").is_none());
    assert_eq!(mock.releases.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn full_debug_task_round_trips_provider_tool_state_and_task_record() {
    let (model_url, mock) = sidecar().await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let debug_directory = directory.path().join("debug-traces");
    let service = AgentService::new(RuntimeConfig {
        model_urls: vec![model_url],
        data_plane_url: data_url,
        session_dir: directory.path().join("sessions"),
        debug_trace: DebugTraceConfig {
            mode: DebugTraceMode::Full,
            directory: debug_directory,
            api_enabled: true,
            ..DebugTraceConfig::default()
        },
        ..RuntimeConfig::default()
    })
    .await
    .unwrap();
    let identity = RequestIdentity::resolve(
        SERVICE_API_VERSION,
        Some("debug-request"),
        Some("debug-owner"),
        "debug-session",
        String::new,
    )
    .unwrap();
    let response = service
        .run_task_with_identity(
            TaskSpec::new("search current fact"),
            &identity,
            Some("debug-task"),
        )
        .await
        .unwrap();
    assert_eq!(response["debug_capture"]["status"], "complete");
    let trace_id = response["trace_id"].as_str().unwrap();
    assert_eq!(mock.releases.load(Ordering::SeqCst), 1);
    let manifest = service
        .debug_trace_manifest_for_owner(trace_id, "debug-owner")
        .await
        .unwrap();
    assert!(manifest.complete);
    assert_eq!(manifest.task_id.as_deref(), Some("debug-task"));
    let events = service
        .debug_trace_events_for_owner(trace_id, "debug-owner", 0, 1_000)
        .await
        .unwrap();
    assert_eq!(events.first().unwrap().event_type, "trace_started");
    assert_eq!(events.last().unwrap().event_type, "trace_finished");
    assert!(
        events
            .iter()
            .any(|event| event.event_type == "model_completed")
    );
    assert!(
        events
            .iter()
            .any(|event| event.event_type == "tool_completed")
    );
    assert!(
        events
            .iter()
            .any(|event| event.event_type == "state_released")
    );
    let model = service
        .debug_trace_file_for_owner(trace_id, "debug-owner", DebugTraceFileKind::Model)
        .await
        .unwrap();
    let model = String::from_utf8(model).unwrap();
    assert!(model.contains("primary fact"));
    let completed = model
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|event| event["event_type"] == "model_completed")
        .unwrap();
    assert!(
        completed["payload"]["provider_request"]["input"]
            .as_str()
            .is_some_and(|input| !input.is_empty())
    );
    assert_eq!(
        completed["payload"]["provider_request"]["stops"],
        json!(["</tool_call>", "</answer>"])
    );
    assert!(
        completed["payload"]["provider_response"]["raw_output"]
            .as_str()
            .is_some_and(|output| output.contains("primary fact"))
    );
    let task = service
        .debug_trace_file_for_owner(trace_id, "debug-owner", DebugTraceFileKind::TaskRecord)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&task).unwrap()["status"],
        "succeeded"
    );
    let page = service
        .debug_traces_for_owner(
            "debug-owner",
            DebugTraceFilter {
                task_id: Some("debug-task".into()),
                limit: 10,
                ..DebugTraceFilter::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(page.traces.len(), 1);
    assert!(
        service
            .debug_trace_manifest_for_owner(trace_id, "other-owner")
            .await
            .is_err()
    );
}

#[tokio::test]
async fn debug_store_failure_is_reported_without_blocking_state_release() {
    let (model_url, mock) = sidecar().await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let debug_path = directory.path().join("debug-is-a-file");
    tokio::fs::write(&debug_path, b"not a directory")
        .await
        .unwrap();
    let service = AgentService::new(RuntimeConfig {
        model_urls: vec![model_url],
        data_plane_url: data_url,
        session_dir: directory.path().join("sessions"),
        debug_trace: DebugTraceConfig {
            mode: DebugTraceMode::Full,
            directory: debug_path,
            ..DebugTraceConfig::default()
        },
        ..RuntimeConfig::default()
    })
    .await
    .unwrap();
    let response = service
        .run("search current fact", "debug-write-failure")
        .await
        .unwrap();
    assert_eq!(response["status"], "ok");
    assert_eq!(response["debug_capture"]["status"], "incomplete");
    assert!(response["trace_id"].as_str().is_some());
    assert_eq!(mock.releases.load(Ordering::SeqCst), 1);
    assert_eq!(
        service.readiness().await["components"]["debug_trace"]["writeable"],
        false
    );
}

#[tokio::test]
async fn provider_failure_still_finalizes_a_queryable_debug_trace() {
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = AgentService::new(RuntimeConfig {
        model_urls: vec!["http://127.0.0.1:9".into()],
        data_plane_url: data_url,
        session_dir: directory.path().join("sessions"),
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
    let error = service
        .run("search current fact", "provider-failure")
        .await
        .unwrap_err();
    assert!(error.contains("sidecar unavailable"));
    let trace_id = error
        .split("debug_trace_id=")
        .nth(1)
        .unwrap()
        .split(';')
        .next()
        .unwrap();
    let manifest = service
        .debug_trace_manifest_for_owner(trace_id, "session:provider-failure")
        .await
        .unwrap();
    assert!(manifest.complete);
    let events = service
        .debug_trace_events_for_owner(trace_id, "session:provider-failure", 0, 100)
        .await
        .unwrap();
    assert_eq!(events.first().unwrap().event_type, "trace_started");
    assert_eq!(events.last().unwrap().event_type, "trace_finished");
    assert_eq!(
        service
            .task(&manifest.task_id.unwrap())
            .await
            .unwrap()
            .status,
        TaskStatus::Failed
    );
}

#[tokio::test]
async fn repeated_tool_failure_stops_before_budget_and_preserves_release_event() {
    let (model_url, mock) = sidecar().await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = AgentService::new(RuntimeConfig {
        model_urls: vec![model_url],
        data_plane_url: data_url,
        session_dir: directory.path().join("sessions"),
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
    assert_eq!(result["debug_capture"]["status"], "complete");
    let events = service
        .debug_trace_events_for_owner(
            result["trace_id"].as_str().unwrap(),
            "session:tool-error",
            0,
            1_000,
        )
        .await
        .unwrap();
    assert!(
        events
            .iter()
            .filter(|event| event.event_type == "tool_completed")
            .count()
            >= 3
    );
    assert!(events.iter().any(|event| event.event_type == "run_failed"));
}

#[tokio::test]
async fn research_forks_parallel_states_reduces_and_releases_all() {
    let (model_url, mock) = sidecar().await;
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = AgentService::new(RuntimeConfig {
        model_urls: vec![model_url],
        data_plane_url: data_url,
        session_dir: directory.path().join("sessions"),
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
    let identity = RequestIdentity::resolve(
        SERVICE_API_VERSION,
        Some("research-request"),
        Some("research-owner"),
        "research",
        String::new,
    )
    .unwrap();
    let result = service
        .research_with_identity("research fact", &identity, 2, 2)
        .await
        .unwrap();
    assert_eq!(result["status"], "ok");
    assert_eq!(result["route"]["branch_width"], 2);
    assert_eq!(result["trace"]["rounds"].as_array().unwrap().len(), 2);
    assert_eq!(mock.releases.load(Ordering::SeqCst), 3);
    assert_eq!(result["debug_capture"]["status"], "complete");
    let events = service
        .debug_trace_events_for_owner(
            result["trace_id"].as_str().unwrap(),
            "research-owner",
            0,
            1_000,
        )
        .await
        .unwrap();
    assert!(
        events
            .iter()
            .any(|event| event.event_type == "research_batch_continue_completed")
    );
    assert!(
        events
            .iter()
            .any(|event| event.event_type == "state_release_completed")
    );
}

#[tokio::test]
async fn research_deadline_releases_root_and_every_forked_state() {
    let (model_url, mock) = sidecar().await;
    mock.continue_delay_ms.store(100, Ordering::SeqCst);
    let data_url = data_plane().await;
    let directory = tempfile::tempdir().unwrap();
    let service = AgentService::new(RuntimeConfig {
        model_urls: vec![model_url],
        data_plane_url: data_url,
        session_dir: directory.path().to_path_buf(),
        max_run_elapsed: Duration::from_millis(20),
        ..RuntimeConfig::default()
    })
    .await
    .unwrap();
    let error = service
        .research("research fact", "research-timeout", 2, 2)
        .await
        .unwrap_err();
    assert!(error.contains("deadline exceeded"));
    assert_eq!(mock.releases.load(Ordering::SeqCst), 3);
}
