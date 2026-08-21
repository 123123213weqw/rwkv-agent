use std::sync::Arc;

use axum::Json;
use axum::routing::post;
use rwkv_state_runtime::{
    ModelRef, PlacementPolicy, ProviderMode, QwenTranscriptProvider, Split, TraceGeneration,
    WorkloadConfig, build_trace, run_long_lived_workload,
};
use serde_json::{Value, json};

async fn completion(Json(request): Json<Value>) -> Json<Value> {
    let content = request["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_str())
        .unwrap_or_default();
    let action = content
        .split_whitespace()
        .find(|value| value.starts_with("ack:S-"))
        .unwrap_or("ack:missing")
        .trim_end_matches('.');
    Json(json!({
        "choices":[{"message":{"role":"assistant","content":action}}],
        "usage":{"prompt_tokens":32,"completion_tokens":4}
    }))
}

async fn failed_completion() -> (axum::http::StatusCode, Json<Value>) {
    (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error":"injected"})),
    )
}

fn model() -> ModelRef {
    ModelRef {
        model_id: "qwen-test".into(),
        revision: "revision-test".into(),
        tokenizer: "tokenizer-test".into(),
        state_abi: "qwen-transcript-reprefill-v1".into(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transcript_provider_rehydrates_without_fake_kv_snapshot() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            axum::Router::new().route("/v1/chat/completions", post(completion)),
        )
        .await
        .expect("serve");
    });
    let provider = Arc::new(
        QwenTranscriptProvider::new(format!("http://{address}"), "qwen-test", model())
            .expect("provider"),
    );
    let events = build_trace(
        Split::Dev,
        10,
        &TraceGeneration {
            seed: 42,
            rounds_per_agent: 3,
            min_wait_ms: 1_000,
            max_wait_ms: 30_000,
            min_token_budget: 64,
            max_token_budget: 128,
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
            policy: PlacementPolicy::DropReprefill,
            model_ref: model(),
            provider_mode: ProviderMode::QwenTranscriptReprefill,
            provider_resolves_durable_state: true,
        },
    )
    .await
    .expect("workload");
    assert_eq!(result.correctness.events, 30);
    assert_eq!(result.correctness.useful_events, 30);
    assert_eq!(result.lifecycle.state_leak_count, 0);
    assert_eq!(provider.allocated().await, 0);
    assert!(provider.durable_bytes().await > 0);
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_failure_still_closes_every_session() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            axum::Router::new().route("/v1/chat/completions", post(failed_completion)),
        )
        .await
        .expect("serve");
    });
    let provider = Arc::new(
        QwenTranscriptProvider::new(format!("http://{address}"), "qwen-test", model())
            .expect("provider"),
    );
    let events = build_trace(
        Split::Dev,
        4,
        &TraceGeneration {
            seed: 99,
            rounds_per_agent: 2,
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
            workers: 2,
            queue_capacity_per_worker: 1,
            policy: PlacementPolicy::DropReprefill,
            model_ref: model(),
            provider_mode: ProviderMode::QwenTranscriptReprefill,
            provider_resolves_durable_state: true,
        },
    )
    .await
    .expect("bounded failure result");
    assert_eq!(result.correctness.failed_events, 8);
    assert_eq!(result.lifecycle.state_leak_count, 0);
    assert_eq!(provider.allocated().await, 0);
    server.abort();
}
