use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::task::JoinSet;

use crate::{
    CheckpointRef, ContinueRequest, CreateRequest, EventTrace, ModelRef, Placement, ProviderMode,
    RestoreRequest, SessionHandle, SnapshotRequest, Split, StateContractError,
    StatefulInferenceProvider,
};

pub const RESULT_SCHEMA_VERSION: &str = "long-lived-agent-result.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementPolicy {
    KeepHot,
    DropReprefill,
    MoveCpu,
}

#[derive(Clone, Debug)]
pub struct WorkloadConfig {
    pub workers: usize,
    pub queue_capacity_per_worker: usize,
    pub policy: PlacementPolicy,
    pub model_ref: ModelRef,
    pub provider_mode: ProviderMode,
    /// True when create(durable_session_ref) restores the durable transcript
    /// internally. The runner then counts replay bytes but never calls model
    /// generation merely to hydrate history.
    pub provider_resolves_durable_state: bool,
}

impl WorkloadConfig {
    pub fn validate(&self) -> Result<(), WorkloadError> {
        if self.workers == 0 || self.queue_capacity_per_worker == 0 {
            return Err(WorkloadError::Invalid(
                "workers and queue capacity must be positive".into(),
            ));
        }
        self.model_ref.validate().map_err(WorkloadError::Contract)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LatencySummary {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeasurementAvailability {
    pub available: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorrectnessSummary {
    pub events: u64,
    pub useful_events: u64,
    pub protocol_valid_events: u64,
    pub cross_talk: u64,
    pub failed_events: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LifecycleSummary {
    pub states_created: u64,
    pub states_released: u64,
    pub state_leak_count: u64,
    pub snapshots: u64,
    pub restores: u64,
    pub checksum_failures: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeMetrics {
    pub wall_ms: f64,
    pub events_per_second: f64,
    pub useful_events_per_second: f64,
    pub event_completion_latency: LatencySummary,
    pub wake_latency: MeasurementAvailability,
    pub ttft: MeasurementAvailability,
    pub queue_backpressure_count: u64,
    pub reprefill_bytes: u64,
    pub restore_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LongLivedResult {
    pub schema_version: String,
    pub split: Split,
    pub provider_mode: ProviderMode,
    pub placement_policy: PlacementPolicy,
    pub workers: usize,
    pub queue_capacity_per_worker: usize,
    pub correctness: CorrectnessSummary,
    pub lifecycle: LifecycleSummary,
    pub metrics: RuntimeMetrics,
    pub errors: Vec<String>,
}

#[derive(Debug, Error)]
pub enum WorkloadError {
    #[error("invalid workload: {0}")]
    Invalid(String),
    #[error("state contract failed: {0}")]
    Contract(#[from] StateContractError),
    #[error("worker task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
    #[error("worker queue closed")]
    QueueClosed,
}

#[derive(Default)]
struct AgentState {
    handle: Option<SessionHandle>,
    checkpoint: Option<CheckpointRef>,
    durable_transcript: String,
}

#[derive(Default)]
struct Counters {
    created: AtomicU64,
    released: AtomicU64,
    snapshots: AtomicU64,
    restores: AtomicU64,
    useful: AtomicU64,
    protocol: AtomicU64,
    cross_talk: AtomicU64,
    failed: AtomicU64,
    reprefill_bytes: AtomicU64,
    restore_bytes: AtomicU64,
    backpressure: AtomicU64,
    event_count: AtomicU64,
    errors: std::sync::Mutex<Vec<String>>,
    latencies_micros: std::sync::Mutex<Vec<u64>>,
}

impl Counters {
    fn error(&self, value: impl Into<String>) {
        self.failed.fetch_add(1, Ordering::Relaxed);
        self.errors
            .lock()
            .expect("error mutex poisoned")
            .push(value.into());
    }
}

fn owner_for(agent_id: &str) -> String {
    format!("owner:{agent_id}")
}

fn durable_ref(agent_id: &str) -> String {
    format!("durable:{agent_id}")
}

fn event_input(event: &EventTrace) -> String {
    format!(
        "Return exactly {}. sentinel={} event_type={} sequence={}",
        event.payload.expected_action,
        event.payload.sentinel,
        event.event_type,
        event.payload.sequence
    )
}

async fn create_session<P: StatefulInferenceProvider>(
    provider: &P,
    config: &WorkloadConfig,
    agent_id: &str,
    counters: &Counters,
) -> Result<SessionHandle, StateContractError> {
    let handle = provider
        .create(CreateRequest {
            owner_id: owner_for(agent_id),
            durable_session_ref: durable_ref(agent_id),
            model_ref: config.model_ref.clone(),
        })
        .await?;
    if handle.provider_mode != config.provider_mode {
        return Err(StateContractError::ProviderMismatch);
    }
    counters.created.fetch_add(1, Ordering::Relaxed);
    Ok(handle)
}

async fn release_session<P: StatefulInferenceProvider>(
    provider: &P,
    agent_id: &str,
    handle: SessionHandle,
    counters: &Counters,
) -> Result<(), StateContractError> {
    let outcome = provider.release(owner_for(agent_id), handle).await?;
    if outcome.released {
        counters.released.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

async fn process_event<P: StatefulInferenceProvider>(
    provider: &P,
    config: &WorkloadConfig,
    state: &mut AgentState,
    event: EventTrace,
    counters: &Counters,
) -> Result<(), StateContractError> {
    let started = Instant::now();
    let owner_id = owner_for(&event.agent_id);
    match config.policy {
        PlacementPolicy::KeepHot => {
            if state.handle.is_none() {
                state.handle =
                    Some(create_session(provider, config, &event.agent_id, counters).await?);
            }
        }
        PlacementPolicy::DropReprefill => {
            let handle = create_session(provider, config, &event.agent_id, counters).await?;
            counters
                .reprefill_bytes
                .fetch_add(state.durable_transcript.len() as u64, Ordering::Relaxed);
            if !config.provider_resolves_durable_state && !state.durable_transcript.is_empty() {
                provider
                    .continue_session(ContinueRequest {
                        owner_id: owner_id.clone(),
                        session_handle: handle.clone(),
                        input: state.durable_transcript.clone(),
                        token_budget: state
                            .durable_transcript
                            .chars()
                            .count()
                            .try_into()
                            .unwrap_or(u32::MAX),
                    })
                    .await?;
            }
            state.handle = Some(handle);
        }
        PlacementPolicy::MoveCpu => {
            if let Some(checkpoint) = state.checkpoint.take() {
                counters
                    .restore_bytes
                    .fetch_add(checkpoint.size_bytes, Ordering::Relaxed);
                state.handle = Some(
                    provider
                        .restore(RestoreRequest {
                            owner_id: owner_id.clone(),
                            checkpoint_ref: checkpoint,
                            expected_model_ref: config.model_ref.clone(),
                        })
                        .await?,
                );
                counters.restores.fetch_add(1, Ordering::Relaxed);
                counters.created.fetch_add(1, Ordering::Relaxed);
            } else {
                state.handle =
                    Some(create_session(provider, config, &event.agent_id, counters).await?);
            }
        }
    }

    let handle = state
        .handle
        .clone()
        .ok_or(StateContractError::StaleHandle)?;
    let input = event_input(&event);
    let output = match provider
        .continue_session(ContinueRequest {
            owner_id: owner_id.clone(),
            session_handle: handle.clone(),
            input: input.clone(),
            token_budget: event.token_budget,
        })
        .await
    {
        Ok(output) => output,
        Err(run_error) => {
            state.handle = None;
            if let Err(release_error) =
                release_session(provider, &event.agent_id, handle, counters).await
            {
                return Err(StateContractError::Provider(format!(
                    "run failed ({run_error}); cleanup failed ({release_error})"
                )));
            }
            return Err(run_error);
        }
    };
    counters.protocol.fetch_add(1, Ordering::Relaxed);
    let expected = &event.payload.expected_action;
    let correct = output.text.contains(expected);
    let foreign = output.text.contains("S-") && !output.text.contains(&event.payload.sentinel);
    if foreign {
        counters.cross_talk.fetch_add(1, Ordering::Relaxed);
    }
    if correct && !foreign {
        counters.useful.fetch_add(1, Ordering::Relaxed);
    }
    state.durable_transcript.push_str(&input);

    match config.policy {
        PlacementPolicy::KeepHot => {}
        PlacementPolicy::DropReprefill => {
            state.handle = None;
            release_session(provider, &event.agent_id, handle, counters).await?;
        }
        PlacementPolicy::MoveCpu => {
            let checkpoint = provider
                .snapshot(SnapshotRequest {
                    owner_id,
                    session_handle: handle.clone(),
                    target_tier: Placement::Cpu,
                })
                .await?;
            counters.snapshots.fetch_add(1, Ordering::Relaxed);
            state.checkpoint = Some(checkpoint);
            state.handle = None;
            release_session(provider, &event.agent_id, handle, counters).await?;
        }
    }
    counters.event_count.fetch_add(1, Ordering::Relaxed);
    counters
        .latencies_micros
        .lock()
        .expect("latency mutex poisoned")
        .push(started.elapsed().as_micros().try_into().unwrap_or(u64::MAX));
    Ok(())
}

fn route(agent_id: &str, workers: usize) -> usize {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in agent_id.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash as usize % workers
}

fn percentile(values: &[u64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index] as f64 / 1000.0
}

pub async fn run_long_lived_workload<P: StatefulInferenceProvider>(
    provider: Arc<P>,
    events: Vec<EventTrace>,
    config: WorkloadConfig,
) -> Result<LongLivedResult, WorkloadError> {
    config.validate()?;
    if events.is_empty() {
        return Err(WorkloadError::Invalid("events must not be empty".into()));
    }
    let split = events[0].split;
    if events.iter().any(|event| event.split != split) {
        return Err(WorkloadError::Invalid(
            "one run must contain exactly one split".into(),
        ));
    }
    for event in &events {
        event
            .validate()
            .map_err(|error| WorkloadError::Invalid(error.to_string()))?;
    }

    let counters = Arc::new(Counters::default());
    let open_workers = Arc::new(AtomicUsize::new(config.workers));
    let mut senders = Vec::with_capacity(config.workers);
    let mut workers = JoinSet::new();
    for _ in 0..config.workers {
        let (sender, mut receiver) = mpsc::channel::<EventTrace>(config.queue_capacity_per_worker);
        senders.push(sender);
        let provider = Arc::clone(&provider);
        let config = config.clone();
        let counters = Arc::clone(&counters);
        let open_workers = Arc::clone(&open_workers);
        workers.spawn(async move {
            let mut agents: HashMap<String, AgentState> = HashMap::new();
            while let Some(event) = receiver.recv().await {
                let agent_id = event.agent_id.clone();
                let state = agents.entry(agent_id).or_default();
                if let Err(error) =
                    process_event(provider.as_ref(), &config, state, event, &counters).await
                {
                    counters.error(format!("{}:{}", error.code(), error));
                }
            }
            for (agent_id, state) in agents {
                if let Some(handle) = state.handle
                    && let Err(error) =
                        release_session(provider.as_ref(), &agent_id, handle, &counters).await
                {
                    counters.error(format!("release:{}", error));
                }
            }
            open_workers.fetch_sub(1, Ordering::Release);
        });
    }

    let started = Instant::now();
    for event in events {
        let index = route(&event.agent_id, config.workers);
        match senders[index].try_send(event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(event)) => {
                counters.backpressure.fetch_add(1, Ordering::Relaxed);
                senders[index]
                    .send(event)
                    .await
                    .map_err(|_| WorkloadError::QueueClosed)?;
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(WorkloadError::QueueClosed);
            }
        }
    }
    drop(senders);
    while let Some(result) = workers.join_next().await {
        result?;
    }
    debug_assert_eq!(open_workers.load(Ordering::Acquire), 0);
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut latencies = counters
        .latencies_micros
        .lock()
        .expect("latency mutex poisoned")
        .clone();
    latencies.sort_unstable();
    let event_count = counters.event_count.load(Ordering::Relaxed);
    let useful = counters.useful.load(Ordering::Relaxed);
    let created = counters.created.load(Ordering::Relaxed);
    let released = counters.released.load(Ordering::Relaxed);
    let errors = counters
        .errors
        .lock()
        .expect("error mutex poisoned")
        .clone();
    let seconds = wall_ms / 1000.0;
    Ok(LongLivedResult {
        schema_version: RESULT_SCHEMA_VERSION.into(),
        split,
        provider_mode: config.provider_mode,
        placement_policy: config.policy,
        workers: config.workers,
        queue_capacity_per_worker: config.queue_capacity_per_worker,
        correctness: CorrectnessSummary {
            events: event_count,
            useful_events: useful,
            protocol_valid_events: counters.protocol.load(Ordering::Relaxed),
            cross_talk: counters.cross_talk.load(Ordering::Relaxed),
            failed_events: counters.failed.load(Ordering::Relaxed),
        },
        lifecycle: LifecycleSummary {
            states_created: created,
            states_released: released,
            state_leak_count: created.saturating_sub(released),
            snapshots: counters.snapshots.load(Ordering::Relaxed),
            restores: counters.restores.load(Ordering::Relaxed),
            checksum_failures: errors
                .iter()
                .filter(|error| error.starts_with("checksum_failure:"))
                .count() as u64,
        },
        metrics: RuntimeMetrics {
            wall_ms,
            events_per_second: if seconds > 0.0 {
                event_count as f64 / seconds
            } else {
                0.0
            },
            useful_events_per_second: if seconds > 0.0 {
                useful as f64 / seconds
            } else {
                0.0
            },
            event_completion_latency: LatencySummary {
                p50_ms: percentile(&latencies, 0.50),
                p95_ms: percentile(&latencies, 0.95),
                p99_ms: percentile(&latencies, 0.99),
                max_ms: latencies.last().copied().unwrap_or(0) as f64 / 1000.0,
            },
            wake_latency: MeasurementAvailability {
                available: false,
                reason: "provider v1 has no explicit inference-ready signal".into(),
            },
            ttft: MeasurementAvailability {
                available: false,
                reason: "provider v1 continuation is non-streaming".into(),
            },
            queue_backpressure_count: counters.backpressure.load(Ordering::Relaxed),
            reprefill_bytes: counters.reprefill_bytes.load(Ordering::Relaxed),
            restore_bytes: counters.restore_bytes.load(Ordering::Relaxed),
        },
        errors,
    })
}

pub fn macro_and_worst_category(results: &[LongLivedResult]) -> BTreeMap<String, f64> {
    let mut output = BTreeMap::new();
    if results.is_empty() {
        return output;
    }
    let rates = results
        .iter()
        .map(|result| {
            if result.correctness.events == 0 {
                0.0
            } else {
                result.correctness.useful_events as f64 / result.correctness.events as f64
            }
        })
        .collect::<Vec<_>>();
    output.insert(
        "macro_useful_rate".into(),
        rates.iter().sum::<f64>() / rates.len() as f64,
    );
    output.insert(
        "worst_category_useful_rate".into(),
        rates.into_iter().fold(f64::INFINITY, f64::min),
    );
    output
}
