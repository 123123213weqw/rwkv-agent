use std::sync::Arc;

use rwkv_state_runtime::{
    InMemoryConformanceProvider, ModelRef, PlacementPolicy, ProviderMode, Split, TraceGeneration,
    WorkloadConfig, build_trace, duplicate_report, run_long_lived_workload,
};

fn generation(seed: u64, ood: bool) -> TraceGeneration {
    TraceGeneration {
        seed,
        rounds_per_agent: 3,
        min_wait_ms: if ood { 31_000 } else { 1_000 },
        max_wait_ms: if ood { 60_000 } else { 30_000 },
        min_token_budget: 64,
        max_token_budget: if ood { 256 } else { 200 },
        event_types: if ood {
            vec!["human_reply".into(), "timer_fired".into()]
        } else {
            vec![
                "github_comment".into(),
                "ci_finished".into(),
                "server_alert".into(),
                "fake_tool_done".into(),
            ]
        },
    }
}

fn config(policy: PlacementPolicy, workers: usize, queue: usize) -> WorkloadConfig {
    WorkloadConfig {
        workers,
        queue_capacity_per_worker: queue,
        policy,
        model_ref: ModelRef {
            model_id: "contract-test".into(),
            revision: "v1".into(),
            tokenizer: "byte-echo".into(),
            state_abi: "bytes-v1".into(),
        },
        provider_mode: ProviderMode::ContractTest,
        provider_resolves_durable_state: false,
    }
}

#[test]
fn traces_are_deterministic_split_and_deduplicated() {
    let dev_a = build_trace(Split::Dev, 100, &generation(11, false)).expect("dev");
    let dev_b = build_trace(Split::Dev, 100, &generation(11, false)).expect("dev");
    let ood = build_trace(Split::OodTransfer, 100, &generation(12, true)).expect("ood");
    assert_eq!(dev_a, dev_b);
    assert_eq!(dev_a.len(), 300);
    assert!(duplicate_report(&[&dev_a, &ood]).clean);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_thousand_agents_use_bounded_tokio_workers_and_release() {
    let events = build_trace(Split::Dev, 1_000, &generation(21, false)).expect("trace");
    for policy in [
        PlacementPolicy::KeepHot,
        PlacementPolicy::DropReprefill,
        PlacementPolicy::MoveCpu,
    ] {
        let provider = Arc::new(InMemoryConformanceProvider::default());
        let result =
            run_long_lived_workload(Arc::clone(&provider), events.clone(), config(policy, 32, 8))
                .await
                .expect("workload");
        assert_eq!(result.correctness.events, 3_000);
        assert_eq!(result.correctness.useful_events, 3_000);
        assert_eq!(result.correctness.cross_talk, 0);
        assert_eq!(result.lifecycle.state_leak_count, 0);
        assert!(result.metrics.events_per_second > 0.0);
        assert!(result.metrics.queue_backpressure_count > 0);
        assert!(result.errors.is_empty(), "{:?}", result.errors);
        assert_eq!(provider.allocated().await, 0);
    }
}
