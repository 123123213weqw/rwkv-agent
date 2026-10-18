use std::collections::BTreeSet;
use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Arc;

use rwkv_state_runtime::{
    AntiOverfitManifest, InMemoryConformanceProvider, MANIFEST_SCHEMA_VERSION, ModelRef,
    PlacementPolicy, ProviderMode, Split, TraceGeneration, TraceManifest, WorkloadConfig,
    build_trace, run_long_lived_workload, sha256_file, write_trace_jsonl,
};

#[derive(Debug)]
struct Args {
    agents: usize,
    rounds: u32,
    workers: usize,
    queue: usize,
    seed: u64,
    split: Split,
    policy: PlacementPolicy,
    trace_output: Option<PathBuf>,
    manifest_output: Option<PathBuf>,
    result_output: Option<PathBuf>,
}

fn value(arguments: &[String], name: &str) -> Option<String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn number<T: std::str::FromStr>(arguments: &[String], name: &str, default: T) -> T {
    value(arguments, name)
        .and_then(|item| item.parse().ok())
        .unwrap_or(default)
}

fn parse_args() -> Result<Args, String> {
    let arguments = env::args().collect::<Vec<_>>();
    if arguments.iter().any(|value| value == "--help") {
        println!(
            "rwkv-state-bench [--agents N] [--rounds N] [--workers N] \
             [--queue N] [--seed N] [--split dev|sealed_blind|ood_transfer] \
             [--policy keep_hot|drop_reprefill|move_cpu] \
             [--trace-output PATH] [--manifest-output PATH] [--result-output PATH]"
        );
        std::process::exit(0);
    }
    let split = match value(&arguments, "--split").as_deref().unwrap_or("dev") {
        "dev" => Split::Dev,
        "public_regression" => Split::PublicRegression,
        "sealed_blind" => Split::SealedBlind,
        "ood_transfer" => Split::OodTransfer,
        other => return Err(format!("unsupported --split: {other}")),
    };
    let policy = match value(&arguments, "--policy")
        .as_deref()
        .unwrap_or("keep_hot")
    {
        "keep_hot" => PlacementPolicy::KeepHot,
        "drop_reprefill" => PlacementPolicy::DropReprefill,
        "move_cpu" => PlacementPolicy::MoveCpu,
        other => return Err(format!("unsupported --policy: {other}")),
    };
    Ok(Args {
        agents: number(&arguments, "--agents", 100),
        rounds: number(&arguments, "--rounds", 3),
        workers: number(&arguments, "--workers", 16),
        queue: number(&arguments, "--queue", 64),
        seed: number(&arguments, "--seed", 0x5eed_u64),
        split,
        policy,
        trace_output: value(&arguments, "--trace-output").map(PathBuf::from),
        manifest_output: value(&arguments, "--manifest-output").map(PathBuf::from),
        result_output: value(&arguments, "--result-output").map(PathBuf::from),
    })
}

fn generation(args: &Args) -> TraceGeneration {
    let event_types = match args.split {
        Split::OodTransfer => vec![
            "human_reply".into(),
            "timer_fired".into(),
            "dependency_ready".into(),
        ],
        _ => vec![
            "github_comment".into(),
            "ci_finished".into(),
            "server_alert".into(),
            "fake_tool_done".into(),
        ],
    };
    TraceGeneration {
        seed: args.seed,
        rounds_per_agent: args.rounds,
        min_wait_ms: if matches!(args.split, Split::OodTransfer) {
            31_000
        } else {
            1_000
        },
        max_wait_ms: if matches!(args.split, Split::OodTransfer) {
            60_000
        } else {
            30_000
        },
        min_token_budget: 64,
        max_token_budget: if matches!(args.split, Split::OodTransfer) {
            256
        } else {
            200
        },
        event_types,
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(std::io::Error::other)?;
    let generation = generation(&args);
    let events = build_trace(args.split, args.agents, &generation)?;
    let trace_path = args
        .trace_output
        .clone()
        .unwrap_or_else(|| env::temp_dir().join("rwkv-state-bench-trace.jsonl"));
    write_trace_jsonl(&trace_path, &events)?;
    let generator_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/trace.rs");
    let manifest = TraceManifest {
        schema_version: MANIFEST_SCHEMA_VERSION.into(),
        split: args.split,
        trace_sha256: sha256_file(&trace_path)?,
        event_count: events.len(),
        agent_count: events
            .iter()
            .map(|event| event.agent_id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        generator_sha256: sha256_file(&generator_path)?,
        generation,
        anti_overfit: AntiOverfitManifest {
            gold_visible_during_development: args.split.visible_during_development(),
            case_specific_routing_allowed: false,
            failure_trace_training_allowed: false,
            formal_runs_allowed: if matches!(args.split, Split::SealedBlind) {
                1
            } else {
                0
            },
        },
    };
    manifest.validate()?;
    if let Some(path) = &args.manifest_output {
        serde_json::to_writer_pretty(BufWriter::new(File::create(path)?), &manifest)?;
    }

    let provider = Arc::new(InMemoryConformanceProvider::default());
    let result = run_long_lived_workload(
        Arc::clone(&provider),
        events,
        WorkloadConfig {
            workers: args.workers,
            queue_capacity_per_worker: args.queue,
            policy: args.policy,
            model_ref: ModelRef {
                model_id: "contract-test".into(),
                revision: "v1".into(),
                tokenizer: "byte-echo".into(),
                state_abi: "bytes-v1".into(),
            },
            provider_mode: ProviderMode::ContractTest,
            provider_resolves_durable_state: false,
        },
    )
    .await?;
    if provider.allocated().await != 0 {
        return Err("conformance provider leaked sessions".into());
    }
    if let Some(path) = &args.result_output {
        serde_json::to_writer_pretty(BufWriter::new(File::create(path)?), &result)?;
    }
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "manifest": manifest,
            "result": result,
        }))?
    );
    Ok(())
}
