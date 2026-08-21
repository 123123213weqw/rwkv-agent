use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::routing::post;
use rwkv_state_runtime::{
    ModelRef, PlacementPolicy, ProviderMode, RwkvHttpProvider, Split, TraceGeneration,
    WorkloadConfig, build_trace, run_long_lived_workload,
};
use serde_json::{Value, json};
use tokio::sync::Mutex;

#[derive(Clone, Default)]
struct MockState {
    next_id: Arc<Mutex<u64>>,
    owners: Arc<Mutex<HashMap<String, String>>>,
}

async fn prefill(State(state): State<MockState>, Json(request): Json<Value>) -> Json<Value> {
    let owner = request["owner_id"].as_str().unwrap_or_default().to_string();
    let mut next = state.next_id.lock().await;
    *next += 1;
    let id = format!("rwkv-state-{next}");
    state.owners.lock().await.insert(id.clone(), owner);
    Json(json!({"state":{"state_id":id,"seen_tokens":8}}))
}

async fn continue_state(
    State(_state): State<MockState>,
    Json(request): Json<Value>,
) -> Json<Value> {
    let item = &request["items"][0];
    let input = item["input"].as_str().unwrap_or_default();
    let action = input
        .split_whitespace()
        .find(|value| value.starts_with("ack:S-"))
        .unwrap_or("ack:missing")
        .trim_end_matches('.');
    Json(json!({
        "results":[{
            "state_id":item["state_id"],
            "text":action,
            "token_ids":[1,2,3],
            "seen_tokens":16
        }]
    }))
}

async fn release(State(state): State<MockState>, Json(request): Json<Value>) -> Json<Value> {
    let mut owners = state.owners.lock().await;
    let released = request["state_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|id| owners.remove(*id).is_some())
        .count();
    Json(json!({"released":released}))
}

fn model() -> ModelRef {
    ModelRef {
        model_id: "rwkv-test".into(),
        revision: "revision-test".into(),
        tokenizer: "tokenizer-test".into(),
        state_abi: "rwkv-state-v1".into(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rwkv_http_keeps_owner_affine_states_and_releases_all() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    let state = MockState::default();
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            axum::Router::new()
                .route("/v1/states/prefill", post(prefill))
                .route("/v1/states/batch_continue", post(continue_state))
                .route("/v1/states/release", post(release))
                .with_state(state),
        )
        .await
        .expect("serve");
    });
    let provider = Arc::new(
        RwkvHttpProvider::new(
            format!("http://{address}"),
            model(),
            "System: Follow the latest user request exactly.",
        )
        .expect("provider"),
    );
    let events = build_trace(
        Split::Dev,
        10,
        &TraceGeneration {
            seed: 7,
            rounds_per_agent: 3,
            min_wait_ms: 1_000,
            max_wait_ms: 2_000,
            min_token_budget: 64,
            max_token_budget: 64,
            event_types: vec!["fake_tool_done".into()],
        },
    )
    .expect("trace");
    let result = run_long_lived_workload(
        Arc::clone(&provider),
        events,
        WorkloadConfig {
            workers: 4,
            queue_capacity_per_worker: 2,
            policy: PlacementPolicy::KeepHot,
            model_ref: model(),
            provider_mode: ProviderMode::RwkvRecurrent,
            provider_resolves_durable_state: false,
        },
    )
    .await
    .expect("workload");
    assert_eq!(result.correctness.events, 30);
    assert_eq!(result.correctness.useful_events, 30);
    assert_eq!(result.lifecycle.states_created, 10);
    assert_eq!(result.lifecycle.states_released, 10);
    assert_eq!(result.lifecycle.state_leak_count, 0);
    assert_eq!(provider.allocated().await, 0);
    server.abort();
}
