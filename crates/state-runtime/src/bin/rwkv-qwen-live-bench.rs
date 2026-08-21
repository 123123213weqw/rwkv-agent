use std::collections::BTreeSet;
use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rwkv_state_runtime::{
    ModelRef, PlacementPolicy, ProviderMode, QwenTranscriptProvider, WorkloadConfig,
    load_trace_jsonl, run_long_lived_workload, sha256_file,
};
use serde::Serialize;

#[derive(Clone, Debug)]
struct Args {
    endpoint: String,
    served_model: String,
    revision: String,
    trace: PathBuf,
    agents: usize,
    workers: usize,
    queue: usize,
    gpu_index: u32,
    output: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
struct GpuSample {
    timestamp_ms: u64,
    used_memory_mib: u64,
    utilization_pct: f64,
    temperature_c: f64,
    power_w: f64,
}

#[derive(Debug, Serialize)]
struct GpuSummary {
    samples: usize,
    used_memory_mib_min: u64,
    used_memory_mib_peak: u64,
    utilization_pct_mean: f64,
    utilization_pct_peak: f64,
    temperature_c_peak: f64,
    power_w_peak: f64,
}

fn value(arguments: &[String], name: &str) -> Option<String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn parse_args() -> Result<Args, String> {
    let arguments = env::args().collect::<Vec<_>>();
    Ok(Args {
        endpoint: value(&arguments, "--endpoint").ok_or("--endpoint is required")?,
        served_model: value(&arguments, "--served-model").ok_or("--served-model is required")?,
        revision: value(&arguments, "--revision").ok_or("--revision is required")?,
        trace: PathBuf::from(value(&arguments, "--trace").ok_or("--trace is required")?),
        agents: value(&arguments, "--agents")
            .as_deref()
            .unwrap_or("100")
            .parse()
            .map_err(|_| "--agents must be an integer")?,
        workers: value(&arguments, "--workers")
            .as_deref()
            .unwrap_or("16")
            .parse()
            .map_err(|_| "--workers must be an integer")?,
        queue: value(&arguments, "--queue")
            .as_deref()
            .unwrap_or("8")
            .parse()
            .map_err(|_| "--queue must be an integer")?,
        gpu_index: value(&arguments, "--gpu-index")
            .as_deref()
            .unwrap_or("0")
            .parse()
            .map_err(|_| "--gpu-index must be an integer")?,
        output: PathBuf::from(value(&arguments, "--output").ok_or("--output is required")?),
    })
}

fn sample_gpu(index: u32) -> Option<GpuSample> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            &index.to_string(),
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&output.stdout);
    let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 4 {
        return None;
    }
    Some(GpuSample {
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX),
        used_memory_mib: fields[0].parse().ok()?,
        utilization_pct: fields[1].parse().ok()?,
        temperature_c: fields[2].parse().ok()?,
        power_w: fields[3].parse().ok()?,
    })
}

fn summarize(samples: &[GpuSample]) -> GpuSummary {
    let count = samples.len() as f64;
    GpuSummary {
        samples: samples.len(),
        used_memory_mib_min: samples
            .iter()
            .map(|sample| sample.used_memory_mib)
            .min()
            .unwrap_or(0),
        used_memory_mib_peak: samples
            .iter()
            .map(|sample| sample.used_memory_mib)
            .max()
            .unwrap_or(0),
        utilization_pct_mean: if count > 0.0 {
            samples
                .iter()
                .map(|sample| sample.utilization_pct)
                .sum::<f64>()
                / count
        } else {
            0.0
        },
        utilization_pct_peak: samples
            .iter()
            .map(|sample| sample.utilization_pct)
            .fold(0.0, f64::max),
        temperature_c_peak: samples
            .iter()
            .map(|sample| sample.temperature_c)
            .fold(0.0, f64::max),
        power_w_peak: samples
            .iter()
            .map(|sample| sample.power_w)
            .fold(0.0, f64::max),
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(std::io::Error::other)?;
    if args.agents == 0 || args.workers == 0 || args.queue == 0 {
        return Err("agents, workers and queue must be positive".into());
    }
    let health = reqwest::get(format!("{}/health", args.endpoint.trim_end_matches('/')))
        .await
        .map_err(|error| format!("Qwen endpoint health preflight failed: {error}"))?;
    if !health.status().is_success() {
        return Err(format!("Qwen endpoint health returned {}", health.status()).into());
    }
    let all_events = load_trace_jsonl(&args.trace)?;
    let agent_ids = all_events
        .iter()
        .map(|event| event.agent_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .take(args.agents)
        .collect::<BTreeSet<_>>();
    if agent_ids.len() != args.agents {
        return Err("trace does not contain the requested agent count".into());
    }
    let events = all_events
        .into_iter()
        .filter(|event| agent_ids.contains(&event.agent_id))
        .collect::<Vec<_>>();
    let expected_events = events.len() as u64;
    let model_ref = ModelRef {
        model_id: args.served_model.clone(),
        revision: args.revision.clone(),
        tokenizer: format!("{}:tokenizer", args.revision),
        state_abi: "qwen-transcript-reprefill-v1".into(),
    };
    let provider = Arc::new(QwenTranscriptProvider::new(
        &args.endpoint,
        &args.served_model,
        model_ref.clone(),
    )?);
    let stop = Arc::new(AtomicBool::new(false));
    let sampler_stop = Arc::clone(&stop);
    let gpu_index = args.gpu_index;
    let sampler = thread::spawn(move || {
        let mut samples = Vec::new();
        while !sampler_stop.load(Ordering::Acquire) {
            if let Some(sample) = sample_gpu(gpu_index) {
                samples.push(sample);
            }
            thread::sleep(Duration::from_millis(200));
        }
        if let Some(sample) = sample_gpu(gpu_index) {
            samples.push(sample);
        }
        samples
    });
    let result = run_long_lived_workload(
        Arc::clone(&provider),
        events,
        WorkloadConfig {
            workers: args.workers,
            queue_capacity_per_worker: args.queue,
            policy: PlacementPolicy::DropReprefill,
            model_ref,
            provider_mode: ProviderMode::QwenTranscriptReprefill,
            provider_resolves_durable_state: true,
        },
    )
    .await;
    stop.store(true, Ordering::Release);
    let samples = sampler.join().map_err(|_| "GPU sampler panicked")?;
    let result = result?;
    let allocated_after = provider.allocated().await;
    let durable_bytes = provider.durable_bytes().await;
    let gate_ok = result.correctness.events == expected_events
        && result.correctness.failed_events == 0
        && result.lifecycle.state_leak_count == 0;
    let output = serde_json::json!({
        "schema_version":"qwen-live-long-lived-bench.v1",
        "endpoint":args.endpoint,
        "served_model":args.served_model,
        "revision":args.revision,
        "quantization":"none",
        "dtype":"float16",
        "provider_mode":"qwen_transcript_reprefill",
        "trace_sha256":sha256_file(&args.trace)?,
        "requested_agents":args.agents,
        "allocated_after":allocated_after,
        "durable_transcript_bytes":durable_bytes,
        "gpu":summarize(&samples),
        "gpu_samples":samples,
        "result":result,
    });
    serde_json::to_writer_pretty(BufWriter::new(File::create(&args.output)?), &output)?;
    println!("{}", serde_json::to_string(&output)?);
    if allocated_after != 0 {
        return Err("Qwen transcript provider leaked sessions".into());
    }
    if !gate_ok {
        return Err("Qwen live correctness/lifecycle gate failed".into());
    }
    Ok(())
}
