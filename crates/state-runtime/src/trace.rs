use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use ring::digest::{Context, SHA256, digest};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const EVENT_SCHEMA_VERSION: &str = "long-lived-agent-event.v1";
pub const MANIFEST_SCHEMA_VERSION: &str = "long-lived-agent-manifest.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Dev,
    PublicRegression,
    SealedBlind,
    OodTransfer,
}

impl Split {
    pub fn visible_during_development(self) -> bool {
        !matches!(self, Self::SealedBlind)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::PublicRegression => "public_regression",
            Self::SealedBlind => "sealed_blind",
            Self::OodTransfer => "ood_transfer",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EventPayload {
    pub sentinel: String,
    pub expected_action: String,
    pub sequence: u32,
    pub history_bucket: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EventTrace {
    pub schema_version: String,
    pub event_id: String,
    pub split: Split,
    pub agent_id: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub payload: EventPayload,
    pub at_ms: u64,
    pub wait_ms: u64,
    pub token_budget: u32,
}

impl EventTrace {
    pub fn validate(&self) -> Result<(), TraceError> {
        if self.schema_version != EVENT_SCHEMA_VERSION {
            return Err(TraceError::Invalid(
                "unsupported event schema_version".into(),
            ));
        }
        for (name, value) in [
            ("event_id", &self.event_id),
            ("agent_id", &self.agent_id),
            ("type", &self.event_type),
            ("sentinel", &self.payload.sentinel),
            ("expected_action", &self.payload.expected_action),
            ("history_bucket", &self.payload.history_bucket),
        ] {
            if value.trim().is_empty() {
                return Err(TraceError::Invalid(format!("{name} must not be empty")));
            }
        }
        if self.wait_ms == 0 || self.token_budget == 0 || self.payload.sequence == 0 {
            return Err(TraceError::Invalid(
                "wait_ms, token_budget and sequence must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TraceGeneration {
    pub seed: u64,
    pub rounds_per_agent: u32,
    pub min_wait_ms: u64,
    pub max_wait_ms: u64,
    pub min_token_budget: u32,
    pub max_token_budget: u32,
    pub event_types: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AntiOverfitManifest {
    pub gold_visible_during_development: bool,
    pub case_specific_routing_allowed: bool,
    pub failure_trace_training_allowed: bool,
    pub formal_runs_allowed: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TraceManifest {
    pub schema_version: String,
    pub split: Split,
    pub trace_sha256: String,
    pub event_count: usize,
    pub agent_count: usize,
    pub generator_sha256: String,
    pub generation: TraceGeneration,
    pub anti_overfit: AntiOverfitManifest,
}

impl TraceManifest {
    pub fn validate(&self) -> Result<(), TraceError> {
        if self.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(TraceError::Invalid(
                "unsupported manifest schema_version".into(),
            ));
        }
        if self.trace_sha256.len() != 64 || self.generator_sha256.len() != 64 {
            return Err(TraceError::Invalid(
                "manifest hashes must be lowercase SHA-256".into(),
            ));
        }
        if self.event_count == 0 || self.agent_count == 0 {
            return Err(TraceError::Invalid(
                "manifest counts must be positive".into(),
            ));
        }
        if self.anti_overfit.gold_visible_during_development
            != self.split.visible_during_development()
        {
            return Err(TraceError::Invalid(
                "anti-overfit visibility does not match split".into(),
            ));
        }
        if self.anti_overfit.case_specific_routing_allowed
            || self.anti_overfit.failure_trace_training_allowed
        {
            return Err(TraceError::Invalid(
                "case routing and failure-trace training must be disabled".into(),
            ));
        }
        if matches!(self.split, Split::SealedBlind) && self.anti_overfit.formal_runs_allowed != 1 {
            return Err(TraceError::Invalid(
                "sealed blind permits exactly one formal run".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DuplicateReport {
    pub trace_count: usize,
    pub duplicate_event_ids: Vec<String>,
    pub duplicate_payload_sha256: Vec<String>,
    pub clean: bool,
}

#[derive(Debug, Error)]
pub enum TraceError {
    #[error("invalid trace: {0}")]
    Invalid(String),
    #[error("trace I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("trace JSON failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn range_u64(&mut self, min: u64, max: u64) -> u64 {
        min + self.next() % (max - min + 1)
    }
}

fn hex_sha256(value: &[u8]) -> String {
    let digest = digest(&SHA256, value);
    let mut output = String::with_capacity(64);
    for byte in digest.as_ref() {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

pub fn build_trace(
    split: Split,
    agent_count: usize,
    generation: &TraceGeneration,
) -> Result<Vec<EventTrace>, TraceError> {
    if agent_count == 0
        || generation.rounds_per_agent == 0
        || generation.event_types.is_empty()
        || generation.min_wait_ms == 0
        || generation.min_wait_ms > generation.max_wait_ms
        || generation.min_token_budget == 0
        || generation.min_token_budget > generation.max_token_budget
        || generation
            .event_types
            .iter()
            .any(|value| value.trim().is_empty())
    {
        return Err(TraceError::Invalid(
            "invalid trace generation bounds".into(),
        ));
    }
    let mut rng = SplitMix64(generation.seed);
    let mut events = Vec::with_capacity(agent_count * generation.rounds_per_agent as usize);
    for agent_index in 1..=agent_count {
        let mut at_ms = 0_u64;
        for sequence in 1..=generation.rounds_per_agent {
            let wait_ms = rng.range_u64(generation.min_wait_ms, generation.max_wait_ms);
            at_ms = at_ms.saturating_add(wait_ms);
            let token_budget = rng.range_u64(
                generation.min_token_budget.into(),
                generation.max_token_budget.into(),
            ) as u32;
            let type_index = rng.next() as usize % generation.event_types.len();
            let event_id = format!("{}-a{agent_index:05}-e{sequence:03}", split.as_str());
            let sentinel = format!(
                "S-{}",
                &hex_sha256(
                    format!(
                        "{}:{agent_index}:{sequence}:{}",
                        generation.seed,
                        split.as_str()
                    )
                    .as_bytes()
                )[..16]
            );
            let event = EventTrace {
                schema_version: EVENT_SCHEMA_VERSION.into(),
                event_id,
                split,
                agent_id: format!("agent-{agent_index:05}"),
                event_type: generation.event_types[type_index].clone(),
                payload: EventPayload {
                    expected_action: format!("ack:{sentinel}"),
                    sentinel,
                    sequence,
                    history_bucket: match sequence {
                        1 => "1k",
                        2 => "4k",
                        3 => "16k",
                        _ => "ood_long",
                    }
                    .into(),
                },
                at_ms,
                wait_ms,
                token_budget,
            };
            event.validate()?;
            events.push(event);
        }
    }
    events.sort_by(|left, right| (left.at_ms, &left.event_id).cmp(&(right.at_ms, &right.event_id)));
    Ok(events)
}

pub fn write_trace_jsonl(path: &Path, events: &[EventTrace]) -> Result<(), TraceError> {
    let mut writer = BufWriter::new(File::create(path)?);
    for event in events {
        event.validate()?;
        serde_json::to_writer(&mut writer, event)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

pub fn load_trace_jsonl(path: &Path) -> Result<Vec<EventTrace>, TraceError> {
    let reader = BufReader::new(File::open(path)?);
    let mut events = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let value = line?;
        if value.trim().is_empty() {
            continue;
        }
        let event: EventTrace = serde_json::from_str(&value)
            .map_err(|error| TraceError::Invalid(format!("line {}: {error}", index + 1)))?;
        event.validate()?;
        events.push(event);
    }
    if events.is_empty() {
        return Err(TraceError::Invalid("trace contains no events".into()));
    }
    let ids = events
        .iter()
        .map(|event| event.event_id.as_str())
        .collect::<BTreeSet<_>>();
    if ids.len() != events.len() {
        return Err(TraceError::Invalid("duplicate event_id".into()));
    }
    if events
        .windows(2)
        .any(|pair| (pair[0].at_ms, &pair[0].event_id) > (pair[1].at_ms, &pair[1].event_id))
    {
        return Err(TraceError::Invalid(
            "events must be sorted by at_ms,event_id".into(),
        ));
    }
    if events
        .iter()
        .map(|event| event.split)
        .collect::<BTreeSet<_>>()
        .len()
        != 1
    {
        return Err(TraceError::Invalid(
            "trace must contain exactly one split".into(),
        ));
    }
    Ok(events)
}

pub fn sha256_file(path: &Path) -> Result<String, TraceError> {
    let mut context = Context::new(&SHA256);
    let mut file = File::open(path)?;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        context.update(&buffer[..read]);
    }
    let digest = context.finish();
    let mut output = String::with_capacity(64);
    for byte in digest.as_ref() {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    Ok(output)
}

pub fn duplicate_report(traces: &[&[EventTrace]]) -> DuplicateReport {
    let mut ids: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    let mut payloads: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for (trace_index, trace) in traces.iter().enumerate() {
        for event in *trace {
            ids.entry(&event.event_id).or_default().insert(trace_index);
            let payload = serde_json::to_vec(&event.payload).unwrap_or_default();
            payloads
                .entry(hex_sha256(&payload))
                .or_default()
                .insert(trace_index);
        }
    }
    let duplicate_event_ids = ids
        .into_iter()
        .filter(|(_, origins)| origins.len() > 1)
        .map(|(id, _)| id.to_string())
        .collect::<Vec<_>>();
    let duplicate_payload_sha256 = payloads
        .into_iter()
        .filter(|(_, origins)| origins.len() > 1)
        .map(|(hash, _)| hash)
        .collect::<Vec<_>>();
    DuplicateReport {
        trace_count: traces.len(),
        clean: duplicate_event_ids.is_empty() && duplicate_payload_sha256.is_empty(),
        duplicate_event_ids,
        duplicate_payload_sha256,
    }
}
