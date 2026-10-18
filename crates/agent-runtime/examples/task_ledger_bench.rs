use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rwkv_agent_runtime::{StageStatus, TaskLedger, TaskSpec, TaskStageSpec, TaskStatus};
use serde_json::json;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

#[derive(Clone, Debug)]
struct Config {
    tasks: usize,
    stages: usize,
    concurrency: usize,
    recovery_tasks: usize,
    root: PathBuf,
    output: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let config = parse_args()?;
    if config.root.exists() {
        tokio::fs::remove_dir_all(&config.root)
            .await
            .map_err(|error| format!("remove old benchmark root: {error}"))?;
    }
    let ledger = TaskLedger::new(&config.root).await?;
    let started = Instant::now();
    let semaphore = Arc::new(Semaphore::new(config.concurrency));
    let mut jobs = JoinSet::new();
    for index in 0..config.tasks {
        let ledger = ledger.clone();
        let semaphore = semaphore.clone();
        let stages = config.stages;
        jobs.spawn(async move {
            let _permit = semaphore.acquire_owned().await.map_err(|e| e.to_string())?;
            run_complete_task(&ledger, index, stages).await
        });
    }
    let mut task_elapsed_ms = Vec::with_capacity(config.tasks);
    while let Some(result) = jobs.join_next().await {
        task_elapsed_ms.push(result.map_err(|error| error.to_string())??);
    }
    let normal_elapsed = started.elapsed();

    let recovery_started = Instant::now();
    for index in 0..config.recovery_tasks {
        create_interrupted_task(&ledger, index, config.stages).await?;
    }
    drop(ledger);
    let reopened = TaskLedger::new(&config.root).await?;
    let recovered = reopened.recover_interrupted().await?;
    let mut recovered_ids = Vec::new();
    for index in 0..config.recovery_tasks {
        let task_id = format!("recovery-{index:04}");
        reopened.prepare_resume(&task_id).await?;
        finish_recovered_task(&reopened, &task_id).await?;
        recovered_ids.push(task_id);
    }
    let recovery_elapsed = recovery_started.elapsed();

    let records = reopened
        .list(config.tasks + config.recovery_tasks + 10)
        .await?;
    let expected_records = config.tasks + config.recovery_tasks;
    let all_succeeded = records
        .iter()
        .all(|record| record.status == TaskStatus::Succeeded);
    let recovery_attempts_valid =
        recovered_ids.iter().all(|task_id| {
            let Some(record) = records.iter().find(|record| &record.task_id == task_id) else {
                return false;
            };
            record.recovery_count == 1
                && record.stages.first().is_some_and(|stage| {
                    stage.status == StageStatus::Succeeded && stage.attempts == 1
                })
                && record.stages.get(1).is_some_and(|stage| {
                    stage.status == StageStatus::Succeeded && stage.attempts == 2
                })
                && record
                    .stages
                    .iter()
                    .skip(2)
                    .all(|stage| stage.status == StageStatus::Succeeded && stage.attempts == 1)
        });
    let temp_files = count_temp_files(&config.root).await?;
    let valid = records.len() == expected_records
        && all_succeeded
        && recovered == config.recovery_tasks
        && recovery_attempts_valid
        && temp_files == 0;

    task_elapsed_ms.sort_unstable();
    let checkpoint_transitions = config.tasks * (3 + 2 * config.stages);
    let metrics = json!({
        "schema":"rwkv-task-ledger-bench.v1",
        "created_unix_ms":unix_ms(),
        "config":{
            "tasks":config.tasks,
            "stages_per_task":config.stages,
            "concurrency":config.concurrency,
            "recovery_tasks":config.recovery_tasks,
        },
        "normal_run":{
            "wall_ms":round3(normal_elapsed.as_secs_f64() * 1000.0),
            "tasks_per_second":round3(config.tasks as f64 / normal_elapsed.as_secs_f64()),
            "checkpoint_transitions":checkpoint_transitions,
            "checkpoint_transitions_per_second":round3(checkpoint_transitions as f64 / normal_elapsed.as_secs_f64()),
            "task_latency_ms":{
                "p50":percentile(&task_elapsed_ms, 0.50),
                "p95":percentile(&task_elapsed_ms, 0.95),
                "max":task_elapsed_ms.last().copied().unwrap_or_default(),
            }
        },
        "restart_recovery":{
            "interrupted_detected":recovered,
            "resumed":config.recovery_tasks,
            "wall_ms":round3(recovery_elapsed.as_secs_f64() * 1000.0),
            "completed_stage_replayed":0,
            "interrupted_stage_attempts":2,
            "later_stage_attempts":1,
        },
        "validation":{
            "expected_records":expected_records,
            "actual_records":records.len(),
            "all_succeeded":all_succeeded,
            "recovery_attempts_valid":recovery_attempts_valid,
            "orphan_temp_files":temp_files,
            "passed":valid,
        },
        "environment":{
            "hostname":std::env::var("HOSTNAME").unwrap_or_default(),
            "ledger_root":config.root,
        }
    });
    let encoded = serde_json::to_vec_pretty(&metrics).map_err(|error| error.to_string())?;
    if let Some(output) = &config.output {
        if let Some(parent) = output.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| error.to_string())?;
        }
        tokio::fs::write(output, &encoded)
            .await
            .map_err(|error| format!("write benchmark output: {error}"))?;
    }
    println!("{}", String::from_utf8_lossy(&encoded));
    if !valid {
        return Err("task ledger benchmark validation failed".into());
    }
    Ok(())
}

async fn run_complete_task(
    ledger: &TaskLedger,
    index: usize,
    stages: usize,
) -> Result<u128, String> {
    let started = Instant::now();
    let task_id = format!("normal-{index:04}");
    let spec = staged_spec(&format!("normal task {index}"), stages);
    ledger.create("bench", spec, Some(&task_id)).await?;
    ledger.start_task(&task_id).await?;
    for stage_index in 0..stages {
        let stage_id = format!("stage-{stage_index:02}");
        ledger.start_stage(&task_id, &stage_id).await?;
        ledger
            .complete_stage(
                &task_id,
                &stage_id,
                json!({"status":"ok","stage":stage_index}),
            )
            .await?;
    }
    ledger
        .complete_task(&task_id, json!({"status":"ok","answer":"done"}))
        .await?;
    Ok(started.elapsed().as_millis())
}

async fn create_interrupted_task(
    ledger: &TaskLedger,
    index: usize,
    stages: usize,
) -> Result<(), String> {
    let task_id = format!("recovery-{index:04}");
    ledger
        .create(
            "recovery",
            staged_spec("restart recovery", stages),
            Some(&task_id),
        )
        .await?;
    ledger.start_task(&task_id).await?;
    ledger.start_stage(&task_id, "stage-00").await?;
    ledger
        .complete_stage(&task_id, "stage-00", json!({"status":"ok"}))
        .await?;
    ledger.start_stage(&task_id, "stage-01").await?;
    Ok(())
}

async fn finish_recovered_task(ledger: &TaskLedger, task_id: &str) -> Result<(), String> {
    let record = ledger.start_task(task_id).await?;
    for stage in record.stages.clone() {
        if stage.status == StageStatus::Succeeded {
            continue;
        }
        ledger.start_stage(task_id, &stage.spec.id).await?;
        ledger
            .complete_stage(task_id, &stage.spec.id, json!({"status":"ok"}))
            .await?;
    }
    ledger
        .complete_task(task_id, json!({"status":"ok","answer":"resumed"}))
        .await?;
    Ok(())
}

fn staged_spec(objective: &str, stages: usize) -> TaskSpec {
    let mut task = TaskSpec::new(objective);
    task.stages = (0..stages)
        .map(|index| TaskStageSpec {
            id: format!("stage-{index:02}"),
            objective: format!("execute stage {index}"),
            depends_on: if index == 0 {
                Vec::new()
            } else {
                vec![format!("stage-{:02}", index - 1)]
            },
            acceptance_criteria: Vec::new(),
            constraints: Vec::new(),
            verification_commands: Vec::new(),
            requires_mutation: None,
        })
        .collect();
    task
}

async fn count_temp_files(root: &PathBuf) -> Result<usize, String> {
    let mut directory = tokio::fs::read_dir(root)
        .await
        .map_err(|error| error.to_string())?;
    let mut count = 0;
    while let Some(entry) = directory.next_entry().await.map_err(|e| e.to_string())? {
        if entry.path().extension().and_then(|value| value.to_str()) == Some("tmp") {
            count += 1;
        }
    }
    Ok(count)
}

fn percentile(values: &[u128], quantile: f64) -> u128 {
    if values.is_empty() {
        return 0;
    }
    let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
    values[index.min(values.len() - 1)]
}

fn parse_args() -> Result<Config, String> {
    let mut tasks = 100;
    let mut stages = 8;
    let mut concurrency = 16;
    let mut recovery_tasks = 10;
    let mut root = None;
    let mut output = None;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {argument}"))?;
        match argument.as_str() {
            "--tasks" => tasks = parse_positive(&value, "tasks")?,
            "--stages" => stages = parse_positive(&value, "stages")?,
            "--concurrency" => concurrency = parse_positive(&value, "concurrency")?,
            "--recovery-tasks" => recovery_tasks = parse_positive(&value, "recovery-tasks")?,
            "--root" => root = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {argument}")),
        }
    }
    if !(2..=32).contains(&stages) {
        return Err("stages must be between 2 and 32".into());
    }
    Ok(Config {
        tasks,
        stages,
        concurrency,
        recovery_tasks,
        root: root.unwrap_or_else(|| {
            std::env::temp_dir().join(format!("rwkv-task-ledger-bench-{}", unix_ms()))
        }),
        output,
    })
}

fn parse_positive(value: &str, name: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name}: {error}"))?;
    if value == 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(value)
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn round3(value: f64) -> f64 {
    (value * 1_000.0).round() / 1_000.0
}
