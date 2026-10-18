use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ring::digest::{SHA256, digest};
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, mpsc, oneshot};

pub const DEBUG_TRACE_SCHEMA_VERSION: &str = "rwkv-agent.debug-trace.v1";
const FILE_NAMES: [&str; 5] = [
    "service-events.jsonl",
    "model.jsonl",
    "tools.jsonl",
    "state.jsonl",
    "stream.jsonl",
];
const DEFAULT_QUEUE_CAPACITY: usize = 256;
const MAX_PAGE_LIMIT: usize = 1_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DebugTraceMode {
    #[default]
    Off,
    Redacted,
    Full,
}

impl DebugTraceMode {
    pub fn enabled(self) -> bool {
        self != Self::Off
    }
}

impl std::fmt::Display for DebugTraceMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Off => "off",
            Self::Redacted => "redacted",
            Self::Full => "full",
        })
    }
}

impl FromStr for DebugTraceMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "redacted" => Ok(Self::Redacted),
            "full" => Ok(Self::Full),
            _ => Err("debug mode must be one of: off, redacted, full".into()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DebugTraceConfig {
    pub mode: DebugTraceMode,
    pub directory: PathBuf,
    pub retention: Duration,
    pub max_bytes: u64,
    pub api_enabled: bool,
    pub queue_capacity: usize,
}

impl Default for DebugTraceConfig {
    fn default() -> Self {
        Self {
            mode: DebugTraceMode::Off,
            directory: PathBuf::from("var/debug-traces"),
            retention: Duration::from_secs(24 * 60 * 60),
            max_bytes: 2 * 1024 * 1024 * 1024,
            api_enabled: false,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugCapture {
    pub mode: DebugTraceMode,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugTraceReadiness {
    pub enabled: bool,
    pub mode: DebugTraceMode,
    pub directory: PathBuf,
    pub writeable: bool,
    pub queue_depth: usize,
    pub incomplete_total: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebugTraceEvent {
    pub schema_version: String,
    pub trace_id: String,
    pub sequence: u64,
    pub timestamp_unix_ns: u128,
    pub elapsed_us: u64,
    pub request_id: String,
    pub owner_id: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_id: Option<String>,
    pub component: String,
    pub event_type: String,
    pub payload: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebugTraceManifest {
    pub schema_version: String,
    pub trace_id: String,
    pub mode: DebugTraceMode,
    pub request_id: String,
    pub owner_id: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    pub runtime_revision: String,
    pub model_revision: String,
    pub tokenizer_revision: String,
    pub state_abi_revision: String,
    pub configuration_identity: Value,
    pub started_unix_ns: u128,
    pub finished_unix_ns: u128,
    pub complete: bool,
    pub completion_reason: String,
    pub first_error: String,
    pub dropped_records: u64,
    pub record_counts: BTreeMap<String, u64>,
    pub file_bytes: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default)]
pub struct DebugTraceFilter {
    pub request_id: Option<String>,
    pub task_id: Option<String>,
    pub session_id: Option<String>,
    pub after_trace_id: Option<String>,
    pub limit: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebugTracePage {
    pub traces: Vec<DebugTraceManifest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_after_trace_id: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugTraceFileKind {
    ServiceEvents,
    Model,
    Tools,
    State,
    Stream,
    TaskRecord,
    FinalResponse,
    Checksums,
}

impl DebugTraceFileKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "service-events" => Some(Self::ServiceEvents),
            "model" => Some(Self::Model),
            "tools" => Some(Self::Tools),
            "state" => Some(Self::State),
            "stream" => Some(Self::Stream),
            "task-record" => Some(Self::TaskRecord),
            "final-response" => Some(Self::FinalResponse),
            "checksums" => Some(Self::Checksums),
            _ => None,
        }
    }

    fn file_name(self) -> &'static str {
        match self {
            Self::ServiceEvents => "service-events.jsonl",
            Self::Model => "model.jsonl",
            Self::Tools => "tools.jsonl",
            Self::State => "state.jsonl",
            Self::Stream => "stream.jsonl",
            Self::TaskRecord => "task-record.json",
            Self::FinalResponse => "final-response.json",
            Self::Checksums => "SHA256SUMS",
        }
    }
}

#[derive(Clone)]
pub struct DebugTraceStore {
    inner: Arc<StoreInner>,
}

struct StoreInner {
    config: DebugTraceConfig,
    runtime_revision: String,
    model_revision: String,
    tokenizer_revision: String,
    state_abi_revision: String,
    configuration_identity: Value,
    writeable: AtomicBool,
    initialization_error: Mutex<Option<String>>,
    queue_depth: Arc<AtomicUsize>,
    incomplete_total: Arc<AtomicU64>,
}

#[derive(Clone)]
pub struct DebugTraceHandle {
    inner: Arc<HandleInner>,
}

struct HandleInner {
    trace_id: String,
    mode: DebugTraceMode,
    request_id: String,
    owner_id: String,
    session_id: String,
    task_id: Option<String>,
    started: Instant,
    sender: Mutex<SenderState>,
    queue_depth: Arc<AtomicUsize>,
    incomplete_total: Arc<AtomicU64>,
    finished: AtomicBool,
    first_error: Mutex<Option<String>>,
    dropped_records: AtomicU64,
    rotation_directory: PathBuf,
    retention: Duration,
    max_bytes: u64,
}

struct SenderState {
    sequence: u64,
    sender: mpsc::Sender<WriterCommand>,
}

pub struct DebugTraceStart {
    pub trace_id: Option<String>,
    pub handle: Option<DebugTraceHandle>,
    pub capture: Option<DebugCapture>,
}

enum WriterCommand {
    Record {
        category: &'static str,
        event: DebugTraceEvent,
    },
    Finish {
        event: DebugTraceEvent,
        task_record: Option<Value>,
        final_response: Option<Value>,
        requested_complete: bool,
        reason: String,
        acknowledgement: oneshot::Sender<Result<DebugTraceManifest, String>>,
    },
}

struct WriterContext {
    partial_dir: PathBuf,
    final_dir: PathBuf,
    files: HashMap<&'static str, tokio::fs::File>,
    manifest_seed: DebugTraceManifest,
    counts: BTreeMap<String, u64>,
    first_error: Option<String>,
    dropped_records: u64,
}

impl DebugTraceStore {
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        config: DebugTraceConfig,
        runtime_revision: String,
        model_revision: String,
        tokenizer_revision: String,
        state_abi_revision: String,
        configuration_identity: Value,
    ) -> Result<Self, String> {
        if config.mode.enabled()
            && (config.retention.is_zero() || config.max_bytes == 0 || config.queue_capacity == 0)
        {
            return Err("debug retention, max bytes and queue capacity must be positive".into());
        }
        let store = Self {
            inner: Arc::new(StoreInner {
                config,
                runtime_revision,
                model_revision,
                tokenizer_revision,
                state_abi_revision,
                configuration_identity,
                writeable: AtomicBool::new(false),
                initialization_error: Mutex::new(None),
                queue_depth: Arc::new(AtomicUsize::new(0)),
                incomplete_total: Arc::new(AtomicU64::new(0)),
            }),
        };
        if store.inner.config.mode.enabled() {
            if let Err(error) = store.initialize().await {
                *store.inner.initialization_error.lock().await = Some(error);
                store.inner.incomplete_total.fetch_add(1, Ordering::Relaxed);
            } else {
                store.inner.writeable.store(true, Ordering::Release);
            }
        }
        Ok(store)
    }

    async fn initialize(&self) -> Result<(), String> {
        reject_symlink(&self.inner.config.directory).await?;
        tokio::fs::create_dir_all(&self.inner.config.directory)
            .await
            .map_err(|error| format!("create debug trace directory: {error}"))?;
        set_owner_only_directory(&self.inner.config.directory).await?;
        reject_symlink(&self.inner.config.directory).await?;
        let probe = self
            .inner
            .config
            .directory
            .join(format!(".write-probe-{}", random_trace_id()?));
        atomic_write(&probe, b"writeable\n").await?;
        tokio::fs::remove_file(&probe)
            .await
            .map_err(|error| format!("remove debug write probe: {error}"))?;
        self.recover_partials().await?;
        self.rotate().await
    }

    pub fn mode(&self) -> DebugTraceMode {
        self.inner.config.mode
    }

    pub fn api_enabled(&self) -> bool {
        self.inner.config.api_enabled && self.mode().enabled()
    }

    pub async fn readiness(&self) -> DebugTraceReadiness {
        DebugTraceReadiness {
            enabled: self.mode().enabled(),
            mode: self.mode(),
            directory: self.inner.config.directory.clone(),
            writeable: self.inner.writeable.load(Ordering::Acquire),
            queue_depth: self.inner.queue_depth.load(Ordering::Relaxed),
            incomplete_total: self.inner.incomplete_total.load(Ordering::Relaxed),
        }
    }

    pub async fn start(
        &self,
        request_id: &str,
        owner_id: &str,
        session_id: &str,
        task_id: Option<&str>,
        payload: Value,
    ) -> DebugTraceStart {
        self.start_lazy(request_id, owner_id, session_id, task_id, || payload)
            .await
    }

    pub async fn start_lazy<F>(
        &self,
        request_id: &str,
        owner_id: &str,
        session_id: &str,
        task_id: Option<&str>,
        payload: F,
    ) -> DebugTraceStart
    where
        F: FnOnce() -> Value,
    {
        if !self.mode().enabled() {
            return DebugTraceStart {
                trace_id: None,
                handle: None,
                capture: None,
            };
        }
        let trace_id = match random_trace_id() {
            Ok(value) => value,
            Err(error) => {
                self.inner.incomplete_total.fetch_add(1, Ordering::Relaxed);
                return failed_start(None, self.mode(), error);
            }
        };
        if !self.inner.writeable.load(Ordering::Acquire) {
            let error = self
                .inner
                .initialization_error
                .lock()
                .await
                .clone()
                .unwrap_or_else(|| "debug trace directory is not writeable".into());
            self.inner.incomplete_total.fetch_add(1, Ordering::Relaxed);
            return failed_start(Some(trace_id), self.mode(), error);
        }
        match self
            .start_inner(trace_id.clone(), request_id, owner_id, session_id, task_id)
            .await
        {
            Ok(handle) => {
                if let Err(error) = handle
                    .record("service", "service", "trace_started", None, payload())
                    .await
                {
                    self.inner.incomplete_total.fetch_add(1, Ordering::Relaxed);
                    failed_start(Some(trace_id), self.mode(), error)
                } else {
                    DebugTraceStart {
                        trace_id: Some(trace_id),
                        handle: Some(handle),
                        capture: Some(DebugCapture {
                            mode: self.mode(),
                            status: "active".into(),
                            error: None,
                        }),
                    }
                }
            }
            Err(error) => {
                self.inner.incomplete_total.fetch_add(1, Ordering::Relaxed);
                failed_start(Some(trace_id), self.mode(), error)
            }
        }
    }

    async fn start_inner(
        &self,
        trace_id: String,
        request_id: &str,
        owner_id: &str,
        session_id: &str,
        task_id: Option<&str>,
    ) -> Result<DebugTraceHandle, String> {
        validate_trace_id(&trace_id)?;
        let partial_dir = self
            .inner
            .config
            .directory
            .join(format!("{trace_id}.partial"));
        let final_dir = self.inner.config.directory.join(&trace_id);
        tokio::fs::create_dir(&partial_dir)
            .await
            .map_err(|error| format!("create partial debug trace: {error}"))?;
        set_owner_only_directory(&partial_dir).await?;
        let mut files = HashMap::new();
        for name in FILE_NAMES {
            let path = partial_dir.join(name);
            let file = tokio::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)
                .await
                .map_err(|error| format!("create debug trace file {name}: {error}"))?;
            set_owner_only_file(&path).await?;
            files.insert(name, file);
        }
        atomic_write(&partial_dir.join("task-record.json"), b"null\n").await?;
        atomic_write(&partial_dir.join("final-response.json"), b"null\n").await?;
        let (sender, receiver) = mpsc::channel(self.inner.config.queue_capacity);
        let started_unix_ns = unix_ns();
        let manifest_seed = DebugTraceManifest {
            schema_version: DEBUG_TRACE_SCHEMA_VERSION.into(),
            trace_id: trace_id.clone(),
            mode: self.mode(),
            request_id: request_id.into(),
            owner_id: owner_id.into(),
            session_id: session_id.into(),
            task_id: task_id.map(str::to_string),
            runtime_revision: self.inner.runtime_revision.clone(),
            model_revision: self.inner.model_revision.clone(),
            tokenizer_revision: self.inner.tokenizer_revision.clone(),
            state_abi_revision: self.inner.state_abi_revision.clone(),
            configuration_identity: self.inner.configuration_identity.clone(),
            started_unix_ns,
            finished_unix_ns: 0,
            complete: false,
            completion_reason: "active".into(),
            first_error: String::new(),
            dropped_records: 0,
            record_counts: BTreeMap::new(),
            file_bytes: BTreeMap::new(),
        };
        let context = WriterContext {
            partial_dir,
            final_dir,
            files,
            manifest_seed,
            counts: BTreeMap::new(),
            first_error: None,
            dropped_records: 0,
        };
        let queue_depth = self.inner.queue_depth.clone();
        tokio::spawn(writer_loop(receiver, context, queue_depth));
        Ok(DebugTraceHandle {
            inner: Arc::new(HandleInner {
                trace_id,
                mode: self.mode(),
                request_id: request_id.into(),
                owner_id: owner_id.into(),
                session_id: session_id.into(),
                task_id: task_id.map(str::to_string),
                started: Instant::now(),
                sender: Mutex::new(SenderState {
                    sequence: 0,
                    sender,
                }),
                queue_depth: self.inner.queue_depth.clone(),
                incomplete_total: self.inner.incomplete_total.clone(),
                finished: AtomicBool::new(false),
                first_error: Mutex::new(None),
                dropped_records: AtomicU64::new(0),
                rotation_directory: self.inner.config.directory.clone(),
                retention: self.inner.config.retention,
                max_bytes: self.inner.config.max_bytes,
            }),
        })
    }

    pub async fn list_for_owner(
        &self,
        owner_id: &str,
        filter: DebugTraceFilter,
    ) -> Result<DebugTracePage, String> {
        let limit = filter.limit.clamp(1, MAX_PAGE_LIMIT);
        let mut manifests = self.read_manifests().await?;
        manifests.retain(|item| {
            item.owner_id == owner_id
                && filter
                    .request_id
                    .as_ref()
                    .is_none_or(|value| item.request_id == *value)
                && filter
                    .task_id
                    .as_ref()
                    .is_none_or(|value| item.task_id.as_deref() == Some(value))
                && filter
                    .session_id
                    .as_ref()
                    .is_none_or(|value| item.session_id == *value)
        });
        manifests.sort_by(|left, right| {
            right
                .started_unix_ns
                .cmp(&left.started_unix_ns)
                .then_with(|| right.trace_id.cmp(&left.trace_id))
        });
        if let Some(after) = filter.after_trace_id.as_deref() {
            let position = manifests
                .iter()
                .position(|item| item.trace_id == after)
                .ok_or_else(|| "debug trace pagination cursor not found".to_string())?;
            manifests.drain(..=position);
        }
        let has_more = manifests.len() > limit;
        manifests.truncate(limit);
        let next_after_trace_id = has_more
            .then(|| manifests.last().map(|item| item.trace_id.clone()))
            .flatten();
        Ok(DebugTracePage {
            traces: manifests,
            next_after_trace_id,
        })
    }

    pub async fn manifest_for_owner(
        &self,
        trace_id: &str,
        owner_id: &str,
    ) -> Result<DebugTraceManifest, String> {
        validate_trace_id(trace_id)?;
        let path = self.inner.config.directory.join(trace_id);
        reject_symlink(&path).await?;
        let manifest = read_manifest(&path.join("manifest.json")).await?;
        if manifest.owner_id != owner_id {
            return Err("debug trace not found".into());
        }
        Ok(manifest)
    }

    pub async fn events_for_owner(
        &self,
        trace_id: &str,
        owner_id: &str,
        after_sequence: u64,
        limit: usize,
    ) -> Result<Vec<DebugTraceEvent>, String> {
        self.manifest_for_owner(trace_id, owner_id).await?;
        let limit = limit.clamp(1, MAX_PAGE_LIMIT);
        let path = self
            .inner
            .config
            .directory
            .join(trace_id)
            .join("service-events.jsonl");
        reject_symlink(&path).await?;
        let raw = tokio::fs::read_to_string(&path)
            .await
            .map_err(|error| format!("read debug trace events: {error}"))?;
        raw.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str::<DebugTraceEvent>(line).map_err(|e| e.to_string()))
            .filter_map(|result| match result {
                Ok(event) if event.sequence > after_sequence => Some(Ok(event)),
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            })
            .take(limit)
            .collect()
    }

    pub async fn file_for_owner(
        &self,
        trace_id: &str,
        owner_id: &str,
        kind: DebugTraceFileKind,
    ) -> Result<Vec<u8>, String> {
        if !self.api_enabled() {
            return Err("debug trace raw file API is disabled".into());
        }
        self.manifest_for_owner(trace_id, owner_id).await?;
        let path = self
            .inner
            .config
            .directory
            .join(trace_id)
            .join(kind.file_name());
        reject_symlink(&path).await?;
        tokio::fs::read(&path)
            .await
            .map_err(|error| format!("read debug trace file: {error}"))
    }

    async fn read_manifests(&self) -> Result<Vec<DebugTraceManifest>, String> {
        if !self.mode().enabled() {
            return Ok(Vec::new());
        }
        let mut directory = tokio::fs::read_dir(&self.inner.config.directory)
            .await
            .map_err(|error| format!("read debug trace directory: {error}"))?;
        let mut manifests = Vec::new();
        while let Some(entry) = directory.next_entry().await.map_err(|e| e.to_string())? {
            let file_type = entry.file_type().await.map_err(|e| e.to_string())?;
            if !file_type.is_dir() || file_type.is_symlink() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if validate_trace_id(&name).is_err() {
                continue;
            }
            if let Ok(manifest) = read_manifest(&entry.path().join("manifest.json")).await {
                manifests.push(manifest);
            }
        }
        Ok(manifests)
    }

    async fn recover_partials(&self) -> Result<(), String> {
        let mut directory = tokio::fs::read_dir(&self.inner.config.directory)
            .await
            .map_err(|error| format!("read debug partial directory: {error}"))?;
        while let Some(entry) = directory.next_entry().await.map_err(|e| e.to_string())? {
            let file_type = entry.file_type().await.map_err(|e| e.to_string())?;
            if !file_type.is_dir() || file_type.is_symlink() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(trace_id) = name.strip_suffix(".partial") else {
                continue;
            };
            if validate_trace_id(trace_id).is_err() {
                continue;
            }
            let path = entry.path();
            let seed = partial_manifest_seed(&path, trace_id, &self.inner).await;
            let mut manifest = match seed {
                Ok(mut manifest) => {
                    manifest.finished_unix_ns = unix_ns();
                    manifest.complete = false;
                    manifest.completion_reason = "process_restart_partial_recovery".into();
                    manifest.first_error = "writer did not finalize before process restart".into();
                    manifest
                }
                Err(error) => {
                    self.inner.incomplete_total.fetch_add(1, Ordering::Relaxed);
                    *self.inner.initialization_error.lock().await = Some(error);
                    continue;
                }
            };
            let recovered_events = recover_event_terminal(&path, &manifest).await?;
            for name in ["task-record.json", "final-response.json"] {
                let file = path.join(name);
                if !tokio::fs::try_exists(&file)
                    .await
                    .map_err(|error| error.to_string())?
                {
                    atomic_write(&file, b"null\n").await?;
                }
            }
            manifest
                .record_counts
                .insert("service".into(), recovered_events);
            write_manifest_and_checksums(&path, &mut manifest).await?;
            let final_path = self.inner.config.directory.join(trace_id);
            if tokio::fs::try_exists(&final_path)
                .await
                .map_err(|e| e.to_string())?
            {
                tokio::fs::remove_dir_all(&path)
                    .await
                    .map_err(|error| format!("remove duplicate partial trace: {error}"))?;
            } else {
                tokio::fs::rename(&path, &final_path)
                    .await
                    .map_err(|error| format!("commit recovered debug trace: {error}"))?;
            }
            self.inner.incomplete_total.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    async fn rotate(&self) -> Result<(), String> {
        rotate_directory(
            &self.inner.config.directory,
            self.inner.config.retention,
            self.inner.config.max_bytes,
        )
        .await
    }
}

async fn rotate_directory(root: &Path, retention: Duration, max_bytes: u64) -> Result<(), String> {
    let now = SystemTime::now();
    let mut entries = Vec::new();
    let mut directory = tokio::fs::read_dir(root)
        .await
        .map_err(|error| format!("read debug trace directory for rotation: {error}"))?;
    while let Some(entry) = directory.next_entry().await.map_err(|e| e.to_string())? {
        let file_type = entry.file_type().await.map_err(|e| e.to_string())?;
        if !file_type.is_dir() || file_type.is_symlink() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if validate_trace_id(&name).is_err() {
            continue;
        }
        let modified = entry
            .metadata()
            .await
            .and_then(|meta| meta.modified())
            .unwrap_or(UNIX_EPOCH);
        let bytes = directory_size(&entry.path()).await?;
        entries.push((entry.path(), modified, bytes));
    }
    entries.sort_by_key(|(_, modified, _)| *modified);
    let mut total = entries.iter().map(|(_, _, bytes)| *bytes).sum::<u64>();
    for (path, modified, bytes) in entries {
        let expired = now.duration_since(modified).unwrap_or_default() > retention;
        if expired || total > max_bytes {
            reject_symlink(&path).await?;
            match tokio::fs::remove_dir_all(&path).await {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(format!("rotate debug trace {}: {error}", path.display()));
                }
            }
            total = total.saturating_sub(bytes);
        }
    }
    Ok(())
}

impl DebugTraceHandle {
    pub fn trace_id(&self) -> &str {
        &self.inner.trace_id
    }

    pub fn mode(&self) -> DebugTraceMode {
        self.inner.mode
    }

    pub async fn record(
        &self,
        category: &'static str,
        component: &str,
        event_type: &str,
        stage_id: Option<&str>,
        payload: Value,
    ) -> Result<(), String> {
        if self.inner.finished.load(Ordering::Acquire) {
            return Err("debug trace is already finished".into());
        }
        let mut sender = self.inner.sender.lock().await;
        sender.sequence += 1;
        let event = self.event(sender.sequence, component, event_type, stage_id, payload);
        self.inner.queue_depth.fetch_add(1, Ordering::Relaxed);
        if sender
            .sender
            .send(WriterCommand::Record { category, event })
            .await
            .is_err()
        {
            self.inner.queue_depth.fetch_sub(1, Ordering::Relaxed);
            self.note_failure("debug trace writer stopped before accepting a record")
                .await;
            return Err("debug trace writer stopped before accepting a record".into());
        }
        Ok(())
    }

    pub async fn finish(
        &self,
        task_record: Option<Value>,
        final_response: Option<Value>,
        requested_complete: bool,
        reason: impl Into<String>,
    ) -> Result<DebugTraceManifest, String> {
        if self.inner.finished.swap(true, Ordering::AcqRel) {
            return Err("debug trace already has a terminal record".into());
        }
        let reason = reason.into();
        let mut sender = self.inner.sender.lock().await;
        sender.sequence += 1;
        let local_error = self.inner.first_error.lock().await.clone();
        let dropped = self.inner.dropped_records.load(Ordering::Relaxed);
        let complete = requested_complete && local_error.is_none() && dropped == 0;
        let event = self.event(
            sender.sequence,
            "service",
            "trace_finished",
            None,
            json!({
                "complete":complete,
                "reason":if complete {reason.clone()} else {local_error.clone().unwrap_or_else(||reason.clone())},
                "dropped_records":dropped,
            }),
        );
        let (acknowledgement, receiver) = oneshot::channel();
        self.inner.queue_depth.fetch_add(1, Ordering::Relaxed);
        if sender
            .sender
            .send(WriterCommand::Finish {
                event,
                task_record: task_record.map(|value| self.capture(value)),
                final_response: final_response.map(|value| self.capture(value)),
                requested_complete: complete,
                reason,
                acknowledgement,
            })
            .await
            .is_err()
        {
            self.inner.queue_depth.fetch_sub(1, Ordering::Relaxed);
            self.inner.incomplete_total.fetch_add(1, Ordering::Relaxed);
            return Err("debug trace writer stopped before finalization".into());
        }
        drop(sender);
        let result = match receiver.await {
            Ok(Ok(manifest)) => {
                if !manifest.complete {
                    self.inner.incomplete_total.fetch_add(1, Ordering::Relaxed);
                }
                Ok(manifest)
            }
            Ok(Err(error)) => {
                self.inner.incomplete_total.fetch_add(1, Ordering::Relaxed);
                Err(error)
            }
            Err(_) => {
                self.inner.incomplete_total.fetch_add(1, Ordering::Relaxed);
                Err("debug trace writer dropped finalization acknowledgement".into())
            }
        };
        if let Err(error) = rotate_directory(
            &self.inner.rotation_directory,
            self.inner.retention,
            self.inner.max_bytes,
        )
        .await
        {
            self.inner.incomplete_total.fetch_add(1, Ordering::Relaxed);
            return Err(format!("debug trace rotation failed: {error}"));
        }
        result
    }

    fn event(
        &self,
        sequence: u64,
        component: &str,
        event_type: &str,
        stage_id: Option<&str>,
        payload: Value,
    ) -> DebugTraceEvent {
        DebugTraceEvent {
            schema_version: DEBUG_TRACE_SCHEMA_VERSION.into(),
            trace_id: self.inner.trace_id.clone(),
            sequence,
            timestamp_unix_ns: unix_ns(),
            elapsed_us: self
                .inner
                .started
                .elapsed()
                .as_micros()
                .min(u64::MAX as u128) as u64,
            request_id: self.inner.request_id.clone(),
            owner_id: self.inner.owner_id.clone(),
            session_id: self.inner.session_id.clone(),
            task_id: self.inner.task_id.clone(),
            stage_id: stage_id.map(str::to_string),
            component: component.into(),
            event_type: event_type.into(),
            payload: self.capture(payload),
        }
    }

    fn capture(&self, value: Value) -> Value {
        match self.inner.mode {
            DebugTraceMode::Full => value,
            DebugTraceMode::Redacted => redact_value(value),
            DebugTraceMode::Off => Value::Null,
        }
    }

    async fn note_failure(&self, error: &str) {
        self.inner.dropped_records.fetch_add(1, Ordering::Relaxed);
        let mut first = self.inner.first_error.lock().await;
        if first.is_none() {
            *first = Some(error.into());
        }
    }
}

async fn writer_loop(
    mut receiver: mpsc::Receiver<WriterCommand>,
    mut context: WriterContext,
    queue_depth: Arc<AtomicUsize>,
) {
    while let Some(command) = receiver.recv().await {
        queue_depth.fetch_sub(1, Ordering::Relaxed);
        match command {
            WriterCommand::Record { category, event } => {
                context.write_event(category, &event).await;
            }
            WriterCommand::Finish {
                event,
                task_record,
                final_response,
                requested_complete,
                reason,
                acknowledgement,
            } => {
                context.write_event("service", &event).await;
                let result = context
                    .finalize(task_record, final_response, requested_complete, reason)
                    .await;
                let _ = acknowledgement.send(result);
                return;
            }
        }
    }
}

impl WriterContext {
    async fn write_event(&mut self, category: &'static str, event: &DebugTraceEvent) {
        let encoded = match serde_json::to_vec(event) {
            Ok(mut value) => {
                value.push(b'\n');
                value
            }
            Err(error) => {
                self.fail(format!("serialize debug trace event: {error}"));
                return;
            }
        };
        self.write_to("service-events.jsonl", &encoded).await;
        if let Some(name) = category_file(category)
            && name != "service-events.jsonl"
        {
            self.write_to(name, &encoded).await;
        }
        *self.counts.entry(category.into()).or_default() += 1;
    }

    async fn write_to(&mut self, name: &'static str, encoded: &[u8]) {
        let Some(file) = self.files.get_mut(name) else {
            self.fail(format!("debug trace category file missing: {name}"));
            return;
        };
        if let Err(error) = file.write_all(encoded).await {
            self.fail(format!("write debug trace {name}: {error}"));
        }
    }

    fn fail(&mut self, error: String) {
        self.dropped_records += 1;
        if self.first_error.is_none() {
            self.first_error = Some(error);
        }
    }

    async fn finalize(
        mut self,
        task_record: Option<Value>,
        final_response: Option<Value>,
        requested_complete: bool,
        reason: String,
    ) -> Result<DebugTraceManifest, String> {
        let mut flush_errors = Vec::new();
        for (name, file) in &mut self.files {
            if let Err(error) = file.flush().await {
                flush_errors.push(format!("flush debug trace {name}: {error}"));
            }
            if let Err(error) = file.sync_all().await {
                flush_errors.push(format!("sync debug trace {name}: {error}"));
            }
        }
        for error in flush_errors {
            self.fail(error);
        }
        let files = std::mem::take(&mut self.files);
        drop(files);
        let task_record = task_record.unwrap_or(Value::Null);
        match serde_json::to_vec_pretty(&task_record) {
            Ok(encoded) => {
                if let Err(error) =
                    atomic_write(&self.partial_dir.join("task-record.json"), &encoded).await
                {
                    self.fail(error);
                }
            }
            Err(error) => self.fail(format!("serialize debug task record: {error}")),
        }
        let final_response = final_response.unwrap_or(Value::Null);
        match serde_json::to_vec_pretty(&final_response) {
            Ok(encoded) => {
                if let Err(error) =
                    atomic_write(&self.partial_dir.join("final-response.json"), &encoded).await
                {
                    self.fail(error);
                }
            }
            Err(error) => self.fail(format!("serialize debug final response: {error}")),
        }
        let mut manifest = self.manifest_seed;
        manifest.finished_unix_ns = unix_ns();
        manifest.complete = requested_complete && self.first_error.is_none();
        manifest.completion_reason = if manifest.complete {
            reason
        } else {
            "debug_trace_incomplete".into()
        };
        manifest.first_error = self.first_error.clone().unwrap_or_default();
        manifest.dropped_records = self.dropped_records;
        manifest.record_counts = self.counts;
        write_manifest_and_checksums(&self.partial_dir, &mut manifest).await?;
        reject_symlink(&self.partial_dir).await?;
        tokio::fs::rename(&self.partial_dir, &self.final_dir)
            .await
            .map_err(|error| format!("commit debug trace directory: {error}"))?;
        if manifest.complete {
            Ok(manifest)
        } else {
            Err(format!("debug_trace_incomplete: {}", manifest.first_error))
        }
    }
}

fn failed_start(trace_id: Option<String>, mode: DebugTraceMode, error: String) -> DebugTraceStart {
    DebugTraceStart {
        trace_id,
        handle: None,
        capture: Some(DebugCapture {
            mode,
            status: "incomplete".into(),
            error: Some(error),
        }),
    }
}

fn category_file(category: &str) -> Option<&'static str> {
    match category {
        "service" => Some("service-events.jsonl"),
        "model" => Some("model.jsonl"),
        "tools" => Some("tools.jsonl"),
        "state" => Some("state.jsonl"),
        "stream" => Some("stream.jsonl"),
        _ => None,
    }
}

fn redact_value(value: Value) -> Value {
    fn allowed(key: &str) -> bool {
        matches!(
            key,
            "status"
                | "error_code"
                | "code"
                | "reason"
                | "success"
                | "state_id"
                | "stop_reason"
                | "mode"
                | "route"
                | "name"
                | "turn"
                | "step"
                | "retry"
                | "sequence"
                | "tokens"
                | "token_count"
                | "output_tokens"
                | "seen_tokens"
                | "elapsed_ms"
                | "elapsed_us"
                | "bytes"
                | "chars"
                | "complete"
                | "dropped_records"
        )
    }
    match value {
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| {
                    let value = if allowed(&key) {
                        match value {
                            Value::Array(values) => json!({"count":values.len()}),
                            Value::Object(values) => redact_value(Value::Object(values)),
                            value => value,
                        }
                    } else {
                        redact_body(value)
                    };
                    (key, value)
                })
                .collect(),
        ),
        value => redact_body(value),
    }
}

fn redact_body(value: Value) -> Value {
    match value {
        Value::String(value) => {
            json!({"redacted":true,"bytes":value.len(),"chars":value.chars().count()})
        }
        Value::Array(values) => json!({"redacted":true,"count":values.len()}),
        Value::Object(values) => {
            let encoded_bytes = serde_json::to_vec(&values).map_or(0, |value| value.len());
            json!({"redacted":true,"keys":values.len(),"bytes":encoded_bytes})
        }
        Value::Null => Value::Null,
        value => value,
    }
}

fn random_trace_id() -> Result<String, String> {
    let mut bytes = [0_u8; 16];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| "secure random trace id generation failed".to_string())?;
    Ok(hex(&bytes))
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn validate_trace_id(value: &str) -> Result<(), String> {
    if value.len() == 32
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err("invalid debug trace id".into())
    }
}

fn unix_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

async fn reject_symlink(path: &Path) -> Result<(), String> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(format!(
            "debug trace path is a symbolic link: {}",
            path.display()
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("inspect debug trace path: {error}")),
    }
}

async fn set_owner_only_directory(path: &Path) -> Result<(), String> {
    set_owner_only(path, 0o700).await
}

async fn set_owner_only_file(path: &Path) -> Result<(), String> {
    set_owner_only(path, 0o600).await
}

#[cfg(unix)]
async fn set_owner_only(path: &Path, mode: u32) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
        .await
        .map_err(|error| format!("set owner-only debug trace permissions: {error}"))
}

#[cfg(not(unix))]
async fn set_owner_only(_path: &Path, _mode: u32) -> Result<(), String> {
    Ok(())
}

async fn atomic_write(path: &Path, encoded: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "debug trace path has no parent".to_string())?;
    reject_symlink(parent).await?;
    let temp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("trace"),
        random_trace_id()?
    ));
    let mut file = tokio::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp)
        .await
        .map_err(|error| format!("create atomic debug trace file: {error}"))?;
    set_owner_only_file(&temp).await?;
    file.write_all(encoded)
        .await
        .map_err(|error| format!("write atomic debug trace file: {error}"))?;
    file.sync_all()
        .await
        .map_err(|error| format!("sync atomic debug trace file: {error}"))?;
    drop(file);
    tokio::fs::rename(&temp, path)
        .await
        .map_err(|error| format!("commit atomic debug trace file: {error}"))
}

async fn read_manifest(path: &Path) -> Result<DebugTraceManifest, String> {
    reject_symlink(path).await?;
    let raw = tokio::fs::read(path)
        .await
        .map_err(|error| format!("read debug trace manifest: {error}"))?;
    serde_json::from_slice(&raw).map_err(|error| format!("parse debug trace manifest: {error}"))
}

async fn collect_file_bytes(path: &Path) -> Result<BTreeMap<String, u64>, String> {
    let mut output = BTreeMap::new();
    let mut directory = tokio::fs::read_dir(path)
        .await
        .map_err(|error| format!("read debug trace files: {error}"))?;
    while let Some(entry) = directory.next_entry().await.map_err(|e| e.to_string())? {
        let file_type = entry.file_type().await.map_err(|e| e.to_string())?;
        if !file_type.is_file() || file_type.is_symlink() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let bytes = entry.metadata().await.map_err(|e| e.to_string())?.len();
        output.insert(name, bytes);
    }
    Ok(output)
}

async fn write_checksums(path: &Path) -> Result<(), String> {
    let mut names = Vec::new();
    let mut directory = tokio::fs::read_dir(path)
        .await
        .map_err(|error| format!("read debug trace files for checksum: {error}"))?;
    while let Some(entry) = directory.next_entry().await.map_err(|e| e.to_string())? {
        let file_type = entry.file_type().await.map_err(|e| e.to_string())?;
        if file_type.is_file() && !file_type.is_symlink() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name != "SHA256SUMS" && !name.ends_with(".tmp") {
                names.push(name);
            }
        }
    }
    names.sort();
    let mut sums = String::new();
    for name in names {
        let bytes = tokio::fs::read(path.join(&name))
            .await
            .map_err(|error| format!("read debug trace checksum input: {error}"))?;
        let hash = digest(&SHA256, &bytes);
        writeln!(&mut sums, "{}  {name}", hex(hash.as_ref())).map_err(|e| e.to_string())?;
    }
    atomic_write(&path.join("SHA256SUMS"), sums.as_bytes()).await
}

async fn write_manifest_and_checksums(
    path: &Path,
    manifest: &mut DebugTraceManifest,
) -> Result<(), String> {
    for _ in 0..8 {
        manifest.file_bytes = collect_file_bytes(path).await?;
        let embedded_sizes = manifest.file_bytes.clone();
        let encoded = serde_json::to_vec_pretty(manifest).map_err(|e| e.to_string())?;
        atomic_write(&path.join("manifest.json"), &encoded).await?;
        write_checksums(path).await?;
        if collect_file_bytes(path).await? == embedded_sizes {
            return Ok(());
        }
    }
    Err("debug trace manifest file sizes did not converge".into())
}

async fn directory_size(path: &Path) -> Result<u64, String> {
    let mut total = 0_u64;
    let mut directory = tokio::fs::read_dir(path)
        .await
        .map_err(|error| format!("read debug trace directory size: {error}"))?;
    while let Some(entry) = directory.next_entry().await.map_err(|e| e.to_string())? {
        let file_type = entry.file_type().await.map_err(|e| e.to_string())?;
        if file_type.is_file() && !file_type.is_symlink() {
            total = total.saturating_add(entry.metadata().await.map_err(|e| e.to_string())?.len());
        }
    }
    Ok(total)
}

async fn partial_manifest_seed(
    path: &Path,
    trace_id: &str,
    store: &StoreInner,
) -> Result<DebugTraceManifest, String> {
    let first = tokio::fs::read_to_string(path.join("service-events.jsonl"))
        .await
        .unwrap_or_default()
        .lines()
        .next()
        .and_then(|line| serde_json::from_str::<DebugTraceEvent>(line).ok());
    let (request_id, owner_id, session_id, task_id, mode, started_unix_ns) = first
        .map(|event| {
            (
                event.request_id,
                event.owner_id,
                event.session_id,
                event.task_id,
                store.config.mode,
                event.timestamp_unix_ns,
            )
        })
        .unwrap_or_else(|| {
            (
                "unknown".into(),
                "unknown".into(),
                "unknown".into(),
                None,
                store.config.mode,
                unix_ns(),
            )
        });
    Ok(DebugTraceManifest {
        schema_version: DEBUG_TRACE_SCHEMA_VERSION.into(),
        trace_id: trace_id.into(),
        mode,
        request_id,
        owner_id,
        session_id,
        task_id,
        runtime_revision: store.runtime_revision.clone(),
        model_revision: store.model_revision.clone(),
        tokenizer_revision: store.tokenizer_revision.clone(),
        state_abi_revision: store.state_abi_revision.clone(),
        configuration_identity: store.configuration_identity.clone(),
        started_unix_ns,
        finished_unix_ns: 0,
        complete: false,
        completion_reason: String::new(),
        first_error: String::new(),
        dropped_records: 0,
        record_counts: BTreeMap::new(),
        file_bytes: collect_file_bytes(path).await?,
    })
}

async fn recover_event_terminal(path: &Path, manifest: &DebugTraceManifest) -> Result<u64, String> {
    let event_path = path.join("service-events.jsonl");
    let raw = tokio::fs::read_to_string(&event_path)
        .await
        .unwrap_or_default();
    let mut events = raw
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<DebugTraceEvent>(line).ok())
        .filter(|event| event.event_type != "trace_finished")
        .collect::<Vec<_>>();
    let start = events
        .iter()
        .position(|event| event.event_type == "trace_started")
        .map(|index| events.remove(index))
        .unwrap_or_else(|| DebugTraceEvent {
            schema_version: DEBUG_TRACE_SCHEMA_VERSION.into(),
            trace_id: manifest.trace_id.clone(),
            sequence: 1,
            timestamp_unix_ns: manifest.started_unix_ns,
            elapsed_us: 0,
            request_id: manifest.request_id.clone(),
            owner_id: manifest.owner_id.clone(),
            session_id: manifest.session_id.clone(),
            task_id: manifest.task_id.clone(),
            stage_id: None,
            component: "service".into(),
            event_type: "trace_started".into(),
            payload: json!({"recovered":true}),
        });
    events.retain(|event| event.event_type != "trace_started");
    events.insert(0, start);
    let terminal = DebugTraceEvent {
        schema_version: DEBUG_TRACE_SCHEMA_VERSION.into(),
        trace_id: manifest.trace_id.clone(),
        sequence: 0,
        timestamp_unix_ns: manifest.finished_unix_ns,
        elapsed_us: 0,
        request_id: manifest.request_id.clone(),
        owner_id: manifest.owner_id.clone(),
        session_id: manifest.session_id.clone(),
        task_id: manifest.task_id.clone(),
        stage_id: None,
        component: "service".into(),
        event_type: "trace_finished".into(),
        payload: json!({
            "complete":false,
            "reason":"process_restart_partial_recovery",
            "dropped_records":manifest.dropped_records,
        }),
    };
    events.push(terminal);
    let mut encoded = Vec::new();
    for (index, event) in events.iter_mut().enumerate() {
        event.sequence = index as u64 + 1;
        serde_json::to_writer(&mut encoded, event).map_err(|e| e.to_string())?;
        encoded.push(b'\n');
    }
    atomic_write(&event_path, &encoded).await?;
    Ok(events.len() as u64)
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    async fn store(temp: &TempDir, mode: DebugTraceMode) -> DebugTraceStore {
        store_with_config(DebugTraceConfig {
            mode,
            directory: temp.path().join("debug"),
            retention: Duration::from_secs(3600),
            max_bytes: 16 * 1024 * 1024,
            api_enabled: true,
            queue_capacity: 2,
        })
        .await
    }

    async fn store_with_config(config: DebugTraceConfig) -> DebugTraceStore {
        DebugTraceStore::new(
            config,
            "runtime-test".into(),
            "model-test".into(),
            "tokenizer-test".into(),
            "state-test".into(),
            json!({"test":true}),
        )
        .await
        .unwrap()
    }

    #[test]
    fn checked_in_debug_schema_tracks_runtime_version_and_trace_id() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../../contracts/debug-trace-v1.schema.json"
        ))
        .unwrap();
        assert_eq!(
            schema
                .pointer("/$defs/event/properties/schema_version/const")
                .and_then(Value::as_str),
            Some(DEBUG_TRACE_SCHEMA_VERSION)
        );
        assert_eq!(
            schema
                .pointer("/$defs/traceId/pattern")
                .and_then(Value::as_str),
            Some("^[0-9a-f]{32}$")
        );
        assert!(schema.pointer("/$defs/manifest").is_some());
        assert!(schema.pointer("/$defs/capture").is_some());
    }

    #[tokio::test]
    async fn off_mode_creates_no_directory_or_trace_id() {
        let temp = TempDir::new().unwrap();
        let store = store(&temp, DebugTraceMode::Off).await;
        let start = store
            .start("request", "owner", "session", Some("task"), json!({}))
            .await;
        assert!(start.trace_id.is_none());
        assert!(start.handle.is_none());
        assert!(!temp.path().join("debug").exists());
    }

    #[tokio::test]
    async fn full_mode_round_trips_content_and_strict_terminal_sequence() {
        let temp = TempDir::new().unwrap();
        let store = store(&temp, DebugTraceMode::Full).await;
        let start = store
            .start(
                "request",
                "owner",
                "session",
                Some("task"),
                json!({"prompt":"secret-start"}),
            )
            .await;
        let handle = start.handle.unwrap();
        handle
            .record(
                "model",
                "model",
                "provider_response",
                Some("stage"),
                json!({"raw_output":"secret-output"}),
            )
            .await
            .unwrap();
        let manifest = handle
            .finish(
                Some(json!({"task":"secret-task"})),
                Some(json!({"answer":"secret-answer"})),
                true,
                "complete",
            )
            .await
            .unwrap();
        assert!(manifest.complete);
        let events = store
            .events_for_owner(&manifest.trace_id, "owner", 0, 100)
            .await
            .unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_type, "trace_started");
        assert_eq!(events[2].event_type, "trace_finished");
        assert_eq!(
            events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        let raw = store
            .file_for_owner(&manifest.trace_id, "owner", DebugTraceFileKind::Model)
            .await
            .unwrap();
        assert!(String::from_utf8(raw).unwrap().contains("secret-output"));
        let trace_dir = temp.path().join("debug").join(&manifest.trace_id);
        for name in [
            "manifest.json",
            "service-events.jsonl",
            "model.jsonl",
            "tools.jsonl",
            "state.jsonl",
            "stream.jsonl",
            "task-record.json",
            "final-response.json",
            "SHA256SUMS",
        ] {
            assert!(trace_dir.join(name).is_file(), "missing {name}");
        }
        let sums = tokio::fs::read_to_string(trace_dir.join("SHA256SUMS"))
            .await
            .unwrap();
        for line in sums.lines() {
            let (expected, name) = line.split_once("  ").unwrap();
            let bytes = tokio::fs::read(trace_dir.join(name)).await.unwrap();
            assert_eq!(expected, hex(digest(&SHA256, &bytes).as_ref()));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&trace_dir).unwrap().permissions().mode() & 0o777,
                0o700
            );
            assert_eq!(
                std::fs::metadata(trace_dir.join("model.jsonl"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        assert!(
            store
                .manifest_for_owner(&manifest.trace_id, "other")
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn redacted_mode_never_persists_body_strings() {
        let temp = TempDir::new().unwrap();
        let store = store(&temp, DebugTraceMode::Redacted).await;
        let start = store
            .start(
                "request",
                "owner",
                "session",
                None,
                json!({"prompt":"secret-start"}),
            )
            .await;
        let handle = start.handle.unwrap();
        handle
            .record(
                "tools",
                "sandbox",
                "tool_completed",
                None,
                json!({"name":"read_file","arguments":{"path":"secret-path"},"result":"secret-result","status":"ok"}),
            )
            .await
            .unwrap();
        let manifest = handle
            .finish(
                None,
                Some(json!({"answer":"secret-answer"})),
                true,
                "complete",
            )
            .await
            .unwrap();
        let raw = tokio::fs::read_to_string(
            temp.path()
                .join("debug")
                .join(manifest.trace_id)
                .join("service-events.jsonl"),
        )
        .await
        .unwrap();
        for secret in [
            "secret-start",
            "secret-path",
            "secret-result",
            "secret-answer",
        ] {
            assert!(!raw.contains(secret));
        }
        assert!(raw.contains("redacted"));
    }

    #[tokio::test]
    async fn bounded_concurrent_writers_preserve_every_sequence() {
        let temp = TempDir::new().unwrap();
        let store = store(&temp, DebugTraceMode::Full).await;
        let handle = store
            .start("request", "owner", "session", None, json!({}))
            .await
            .handle
            .unwrap();
        let mut workers = Vec::new();
        for worker in 0..8 {
            let handle = handle.clone();
            workers.push(tokio::spawn(async move {
                for item in 0..25 {
                    handle
                        .record(
                            "model",
                            "concurrency_test",
                            "record",
                            None,
                            json!({"worker":worker,"item":item}),
                        )
                        .await
                        .unwrap();
                }
            }));
        }
        for worker in workers {
            worker.await.unwrap();
        }
        let manifest = handle.finish(None, None, true, "complete").await.unwrap();
        assert_eq!(manifest.dropped_records, 0);
        let events = store
            .events_for_owner(&manifest.trace_id, "owner", 0, 1_000)
            .await
            .unwrap();
        assert_eq!(events.len(), 202);
        assert!(
            events
                .iter()
                .enumerate()
                .all(|(index, event)| event.sequence == index as u64 + 1)
        );
    }

    #[tokio::test]
    async fn list_pagination_and_filters_are_owner_scoped() {
        let temp = TempDir::new().unwrap();
        let store = store(&temp, DebugTraceMode::Redacted).await;
        for index in 0..3 {
            let handle = store
                .start(
                    &format!("request-{index}"),
                    "owner",
                    "session",
                    Some(&format!("task-{index}")),
                    json!({}),
                )
                .await
                .handle
                .unwrap();
            handle.finish(None, None, true, "complete").await.unwrap();
        }
        let first = store
            .list_for_owner(
                "owner",
                DebugTraceFilter {
                    limit: 2,
                    ..DebugTraceFilter::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(first.traces.len(), 2);
        let second = store
            .list_for_owner(
                "owner",
                DebugTraceFilter {
                    after_trace_id: first.next_after_trace_id,
                    limit: 2,
                    ..DebugTraceFilter::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(second.traces.len(), 1);
        assert!(
            store
                .list_for_owner(
                    "other",
                    DebugTraceFilter {
                        limit: 10,
                        ..DebugTraceFilter::default()
                    }
                )
                .await
                .unwrap()
                .traces
                .is_empty()
        );
    }

    #[tokio::test]
    async fn startup_recovers_partial_directory_as_incomplete() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("debug");
        let trace_id = "0123456789abcdef0123456789abcdef";
        let partial = root.join(format!("{trace_id}.partial"));
        tokio::fs::create_dir_all(&partial).await.unwrap();
        for name in FILE_NAMES {
            tokio::fs::write(partial.join(name), b"").await.unwrap();
        }
        let first = DebugTraceEvent {
            schema_version: DEBUG_TRACE_SCHEMA_VERSION.into(),
            trace_id: trace_id.into(),
            sequence: 1,
            timestamp_unix_ns: unix_ns(),
            elapsed_us: 0,
            request_id: "request".into(),
            owner_id: "owner".into(),
            session_id: "session".into(),
            task_id: None,
            stage_id: None,
            component: "service".into(),
            event_type: "trace_started".into(),
            payload: json!({}),
        };
        tokio::fs::write(
            partial.join("service-events.jsonl"),
            format!("{}\n", serde_json::to_string(&first).unwrap()),
        )
        .await
        .unwrap();
        let store = store_with_config(DebugTraceConfig {
            mode: DebugTraceMode::Full,
            directory: root.clone(),
            ..DebugTraceConfig::default()
        })
        .await;
        let manifest = store.manifest_for_owner(trace_id, "owner").await.unwrap();
        assert!(!manifest.complete);
        assert_eq!(
            manifest.completion_reason,
            "process_restart_partial_recovery"
        );
        assert!(!partial.exists());
        assert!(root.join(trace_id).join("SHA256SUMS").exists());
        let events = store
            .events_for_owner(trace_id, "owner", 0, 100)
            .await
            .unwrap();
        assert_eq!(events.first().unwrap().event_type, "trace_started");
        assert_eq!(events.last().unwrap().event_type, "trace_finished");
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event_type == "trace_finished")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn rotation_enforces_total_byte_limit() {
        let temp = TempDir::new().unwrap();
        let store = store_with_config(DebugTraceConfig {
            mode: DebugTraceMode::Full,
            directory: temp.path().join("debug"),
            max_bytes: 1,
            ..DebugTraceConfig::default()
        })
        .await;
        let handle = store
            .start("request", "owner", "session", None, json!({"large":"body"}))
            .await
            .handle
            .unwrap();
        let trace_id = handle.trace_id().to_string();
        handle.finish(None, None, true, "complete").await.unwrap();
        assert!(!temp.path().join("debug").join(trace_id).exists());
    }

    #[tokio::test]
    async fn rotation_removes_traces_older_than_retention() {
        let temp = TempDir::new().unwrap();
        let store = store_with_config(DebugTraceConfig {
            mode: DebugTraceMode::Full,
            directory: temp.path().join("debug"),
            retention: Duration::from_secs(1),
            ..DebugTraceConfig::default()
        })
        .await;
        let first = store
            .start("request-1", "owner", "session", None, json!({}))
            .await
            .handle
            .unwrap();
        let first_trace_id = first.trace_id().to_string();
        first.finish(None, None, true, "complete").await.unwrap();
        assert!(temp.path().join("debug").join(&first_trace_id).exists());

        tokio::time::sleep(Duration::from_millis(1_100)).await;
        let second = store
            .start("request-2", "owner", "session", None, json!({}))
            .await
            .handle
            .unwrap();
        second.finish(None, None, true, "complete").await.unwrap();

        assert!(!temp.path().join("debug").join(first_trace_id).exists());
    }

    #[tokio::test]
    async fn initialization_failure_is_observable_without_a_writer() {
        let temp = TempDir::new().unwrap();
        let root = temp.path().join("not-a-directory");
        tokio::fs::write(&root, b"file").await.unwrap();
        let store = store_with_config(DebugTraceConfig {
            mode: DebugTraceMode::Full,
            directory: root,
            ..DebugTraceConfig::default()
        })
        .await;
        let readiness = store.readiness().await;
        assert!(readiness.enabled);
        assert!(!readiness.writeable);
        let start = store
            .start(
                "request",
                "owner",
                "session",
                None,
                json!({"secret":"body"}),
            )
            .await;
        assert!(start.trace_id.is_some());
        assert!(start.handle.is_none());
        assert_eq!(start.capture.unwrap().status, "incomplete");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symbolic_link_trace_root_is_rejected() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let target = temp.path().join("target");
        tokio::fs::create_dir(&target).await.unwrap();
        let link = temp.path().join("debug-link");
        symlink(&target, &link).unwrap();
        let store = store_with_config(DebugTraceConfig {
            mode: DebugTraceMode::Full,
            directory: link,
            ..DebugTraceConfig::default()
        })
        .await;
        assert!(!store.readiness().await.writeable);
        assert!(
            tokio::fs::read_dir(target)
                .await
                .unwrap()
                .next_entry()
                .await
                .unwrap()
                .is_none()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn writer_finalization_failure_never_reports_complete() {
        let temp = TempDir::new().unwrap();
        let store = store(&temp, DebugTraceMode::Full).await;
        let handle = store
            .start("request", "owner", "session", None, json!({}))
            .await
            .handle
            .unwrap();
        let partial = temp
            .path()
            .join("debug")
            .join(format!("{}.partial", handle.trace_id()));
        tokio::fs::remove_dir_all(&partial).await.unwrap();
        let error = handle
            .finish(None, Some(json!({"answer":"body"})), true, "complete")
            .await
            .unwrap_err();
        assert!(error.contains("debug") || error.contains("atomic"));
        assert!(store.readiness().await.incomplete_total > 0);
    }
}
