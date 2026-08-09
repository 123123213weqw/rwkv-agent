use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rwkv_agent_core::{TaskSpec, TaskStageSpec};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

const LEDGER_SCHEMA_VERSION: u32 = 1;
const MAX_EVENTS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Interrupted,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Interrupted,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskStageRecord {
    pub spec: TaskStageSpec,
    pub status: StageStatus,
    pub attempts: u32,
    pub started_unix_ms: Option<u64>,
    pub completed_unix_ms: Option<u64>,
    pub response: Option<Value>,
    pub error: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LedgerEvent {
    pub sequence: u64,
    pub unix_ms: u64,
    pub kind: String,
    pub stage_id: Option<String>,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskRecord {
    pub ledger_schema_version: u32,
    pub task_id: String,
    pub session_id: String,
    pub task_spec: TaskSpec,
    pub status: TaskStatus,
    pub current_stage: Option<String>,
    pub stages: Vec<TaskStageRecord>,
    pub created_unix_ms: u64,
    pub updated_unix_ms: u64,
    pub revision: u64,
    pub recovery_count: u32,
    pub final_response: Option<Value>,
    pub error: String,
    pub events: Vec<LedgerEvent>,
}

#[derive(Clone)]
pub struct TaskLedger {
    root: Arc<PathBuf>,
    lock: Arc<Mutex<()>>,
    sequence: Arc<AtomicU64>,
}

impl TaskLedger {
    pub async fn new(root: impl Into<PathBuf>) -> Result<Self, String> {
        let root = root.into();
        tokio::fs::create_dir_all(&root)
            .await
            .map_err(|error| format!("create task ledger directory: {error}"))?;
        Ok(Self {
            root: Arc::new(root),
            lock: Arc::new(Mutex::new(())),
            sequence: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn root(&self) -> &Path {
        self.root.as_ref()
    }

    pub async fn create(
        &self,
        session_id: &str,
        task_spec: TaskSpec,
        requested_id: Option<&str>,
    ) -> Result<TaskRecord, String> {
        let task_spec = task_spec.normalize().map_err(|error| error.to_string())?;
        let stages = task_spec
            .ordered_stages()
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|spec| TaskStageRecord {
                spec,
                status: StageStatus::Pending,
                attempts: 0,
                started_unix_ms: None,
                completed_unix_ms: None,
                response: None,
                error: String::new(),
            })
            .collect::<Vec<_>>();
        let task_id = match requested_id {
            Some(value) => normalize_task_id(value)?,
            None => self.generate_id(),
        };
        let _guard = self.lock.lock().await;
        if tokio::fs::try_exists(self.path(&task_id))
            .await
            .map_err(|error| error.to_string())?
        {
            return Err(format!("task_id `{task_id}` already exists"));
        }
        let now = unix_ms();
        let mut record = TaskRecord {
            ledger_schema_version: LEDGER_SCHEMA_VERSION,
            task_id,
            session_id: session_id.to_string(),
            task_spec,
            status: TaskStatus::Pending,
            current_stage: None,
            stages,
            created_unix_ms: now,
            updated_unix_ms: now,
            revision: 0,
            recovery_count: 0,
            final_response: None,
            error: String::new(),
            events: Vec::new(),
        };
        push_event(
            &mut record,
            "created",
            None,
            "task persisted before execution",
        );
        self.write_locked(&record).await?;
        Ok(record)
    }

    pub async fn get(&self, task_id: &str) -> Result<TaskRecord, String> {
        let task_id = normalize_task_id(task_id)?;
        let _guard = self.lock.lock().await;
        self.read_locked(&task_id).await
    }

    pub async fn list(&self, limit: usize) -> Result<Vec<TaskRecord>, String> {
        let _guard = self.lock.lock().await;
        let mut directory = tokio::fs::read_dir(self.root.as_ref())
            .await
            .map_err(|error| format!("read task ledger directory: {error}"))?;
        let mut records = Vec::new();
        while let Some(entry) = directory.next_entry().await.map_err(|e| e.to_string())? {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let raw = tokio::fs::read(&path)
                .await
                .map_err(|error| format!("read task record {}: {error}", path.display()))?;
            records.push(parse_record(&raw, &path)?);
        }
        records.sort_by_key(|record| std::cmp::Reverse(record.updated_unix_ms));
        records.truncate(limit.min(1_000));
        Ok(records)
    }

    pub async fn start_task(&self, task_id: &str) -> Result<TaskRecord, String> {
        self.mutate(task_id, |record| {
            if matches!(record.status, TaskStatus::Succeeded | TaskStatus::Cancelled) {
                return Err(format!("task {} is already terminal", record.task_id));
            }
            record.status = TaskStatus::Running;
            record.error.clear();
            push_event(
                record,
                "task_started",
                None,
                "stage controller claimed task",
            );
            Ok(())
        })
        .await
    }

    pub async fn start_stage(&self, task_id: &str, stage_id: &str) -> Result<TaskRecord, String> {
        let stage_id = stage_id.to_string();
        self.mutate(task_id, move |record| {
            let stage = stage_mut(record, &stage_id)?;
            if stage.status == StageStatus::Succeeded {
                return Err(format!("stage `{stage_id}` already succeeded"));
            }
            stage.status = StageStatus::Running;
            stage.attempts += 1;
            stage.started_unix_ms = Some(unix_ms());
            stage.completed_unix_ms = None;
            stage.error.clear();
            record.status = TaskStatus::Running;
            record.current_stage = Some(stage_id.clone());
            push_event(
                record,
                "stage_started",
                Some(&stage_id),
                "checkpoint before execution",
            );
            Ok(())
        })
        .await
    }

    pub async fn complete_stage(
        &self,
        task_id: &str,
        stage_id: &str,
        response: Value,
    ) -> Result<TaskRecord, String> {
        let stage_id = stage_id.to_string();
        self.mutate(task_id, move |record| {
            let stage = stage_mut(record, &stage_id)?;
            if stage.status != StageStatus::Running {
                return Err(format!("stage `{stage_id}` is not running"));
            }
            stage.status = StageStatus::Succeeded;
            stage.completed_unix_ms = Some(unix_ms());
            stage.response = Some(response);
            stage.error.clear();
            record.current_stage = None;
            push_event(
                record,
                "stage_succeeded",
                Some(&stage_id),
                "response checkpointed",
            );
            Ok(())
        })
        .await
    }

    pub async fn fail_stage(
        &self,
        task_id: &str,
        stage_id: &str,
        error: &str,
        response: Option<Value>,
    ) -> Result<TaskRecord, String> {
        let stage_id = stage_id.to_string();
        let error = error.to_string();
        self.mutate(task_id, move |record| {
            let stage = stage_mut(record, &stage_id)?;
            stage.status = StageStatus::Failed;
            stage.completed_unix_ms = Some(unix_ms());
            stage.response = response;
            stage.error = error.clone();
            record.status = TaskStatus::Failed;
            record.current_stage = Some(stage_id.clone());
            record.error = error.clone();
            push_event(record, "stage_failed", Some(&stage_id), &error);
            Ok(())
        })
        .await
    }

    pub async fn complete_task(
        &self,
        task_id: &str,
        response: Value,
    ) -> Result<TaskRecord, String> {
        self.mutate(task_id, move |record| {
            if record
                .stages
                .iter()
                .any(|stage| stage.status != StageStatus::Succeeded)
            {
                return Err("cannot complete task before every stage succeeds".into());
            }
            record.status = TaskStatus::Succeeded;
            record.current_stage = None;
            record.final_response = Some(response);
            record.error.clear();
            push_event(
                record,
                "task_succeeded",
                None,
                "all stage checkpoints succeeded",
            );
            Ok(())
        })
        .await
    }

    pub async fn prepare_resume(&self, task_id: &str) -> Result<TaskRecord, String> {
        self.mutate(task_id, |record| {
            if !matches!(record.status, TaskStatus::Failed | TaskStatus::Interrupted) {
                return Err(format!("task {} is not resumable", record.task_id));
            }
            for stage in &mut record.stages {
                if matches!(stage.status, StageStatus::Failed | StageStatus::Interrupted) {
                    stage.status = StageStatus::Pending;
                    stage.completed_unix_ms = None;
                    stage.error.clear();
                }
            }
            record.status = TaskStatus::Pending;
            record.current_stage = None;
            record.error.clear();
            push_event(record, "resume_prepared", None, "completed stages retained");
            Ok(())
        })
        .await
    }

    pub async fn cancel(&self, task_id: &str) -> Result<TaskRecord, String> {
        self.mutate(task_id, |record| {
            if record.status == TaskStatus::Succeeded {
                return Err("cannot cancel a succeeded task".into());
            }
            record.status = TaskStatus::Cancelled;
            record.current_stage = None;
            for stage in &mut record.stages {
                if matches!(stage.status, StageStatus::Pending | StageStatus::Running) {
                    stage.status = StageStatus::Cancelled;
                }
            }
            push_event(
                record,
                "task_cancelled",
                None,
                "controller cancellation persisted",
            );
            Ok(())
        })
        .await
    }

    /// On Controller startup, a previously running stage has lost its volatile
    /// RWKV state. Persist it as interrupted so an explicit resume can replay
    /// only that stage while retaining completed checkpoints.
    pub async fn recover_interrupted(&self) -> Result<usize, String> {
        let ids = self
            .list(1_000)
            .await?
            .into_iter()
            .filter(|record| record.status == TaskStatus::Running)
            .map(|record| record.task_id)
            .collect::<Vec<_>>();
        for id in &ids {
            self.mutate(id, |record| {
                for stage in &mut record.stages {
                    if stage.status == StageStatus::Running {
                        stage.status = StageStatus::Interrupted;
                        stage.completed_unix_ms = Some(unix_ms());
                        stage.error = "controller restarted before stage checkpoint".into();
                    }
                }
                record.status = TaskStatus::Interrupted;
                record.recovery_count += 1;
                record.error = "controller restarted before task completion".into();
                let current_stage = record.current_stage.clone();
                push_event(
                    record,
                    "startup_recovery",
                    current_stage.as_deref(),
                    "volatile model state was lost",
                );
                Ok(())
            })
            .await?;
        }
        Ok(ids.len())
    }

    async fn mutate<F>(&self, task_id: &str, update: F) -> Result<TaskRecord, String>
    where
        F: FnOnce(&mut TaskRecord) -> Result<(), String>,
    {
        let task_id = normalize_task_id(task_id)?;
        let _guard = self.lock.lock().await;
        let mut record = self.read_locked(&task_id).await?;
        update(&mut record)?;
        record.revision += 1;
        record.updated_unix_ms = unix_ms();
        self.write_locked(&record).await?;
        Ok(record)
    }

    async fn read_locked(&self, task_id: &str) -> Result<TaskRecord, String> {
        let path = self.path(task_id);
        let raw = tokio::fs::read(&path)
            .await
            .map_err(|error| format!("read task `{task_id}`: {error}"))?;
        parse_record(&raw, &path)
    }

    async fn write_locked(&self, record: &TaskRecord) -> Result<(), String> {
        let encoded = serde_json::to_vec_pretty(record).map_err(|error| error.to_string())?;
        let path = self.path(&record.task_id);
        let temp = self.root.join(format!(
            ".{}.{}.tmp",
            record.task_id,
            self.sequence.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = tokio::fs::File::create(&temp)
            .await
            .map_err(|error| format!("create task checkpoint: {error}"))?;
        file.write_all(&encoded)
            .await
            .map_err(|error| format!("write task checkpoint: {error}"))?;
        file.sync_all()
            .await
            .map_err(|error| format!("sync task checkpoint: {error}"))?;
        drop(file);
        tokio::fs::rename(&temp, &path)
            .await
            .map_err(|error| format!("commit task checkpoint: {error}"))?;
        Ok(())
    }

    fn path(&self, task_id: &str) -> PathBuf {
        self.root.join(format!("{task_id}.json"))
    }

    fn generate_id(&self) -> String {
        format!(
            "task-{}-{:04}",
            unix_ms(),
            self.sequence.fetch_add(1, Ordering::Relaxed) % 10_000
        )
    }
}

fn stage_mut<'a>(
    record: &'a mut TaskRecord,
    stage_id: &str,
) -> Result<&'a mut TaskStageRecord, String> {
    record
        .stages
        .iter_mut()
        .find(|stage| stage.spec.id == stage_id)
        .ok_or_else(|| format!("unknown stage `{stage_id}`"))
}

fn push_event(record: &mut TaskRecord, kind: &str, stage_id: Option<&str>, detail: &str) {
    let sequence = record.events.last().map_or(1, |event| event.sequence + 1);
    record.events.push(LedgerEvent {
        sequence,
        unix_ms: unix_ms(),
        kind: kind.into(),
        stage_id: stage_id.map(str::to_string),
        detail: detail.chars().take(1_000).collect(),
    });
    if record.events.len() > MAX_EVENTS {
        record.events.drain(..record.events.len() - MAX_EVENTS);
    }
}

fn parse_record(raw: &[u8], path: &Path) -> Result<TaskRecord, String> {
    let record: TaskRecord = serde_json::from_slice(raw)
        .map_err(|error| format!("invalid task record {}: {error}", path.display()))?;
    if record.ledger_schema_version != LEDGER_SCHEMA_VERSION {
        return Err(format!(
            "unsupported task ledger schema {} in {}",
            record.ledger_schema_version,
            path.display()
        ));
    }
    Ok(record)
}

fn normalize_task_id(value: &str) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty()
        || value.chars().count() > 128
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "._-".contains(character))
    {
        return Err("task_id must be 1-128 ASCII letters, digits, `.`, `_`, or `-`".into());
    }
    Ok(value.into())
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn checkpoints_survive_reopen_and_resume_retains_completed_stages() {
        let directory = tempfile::tempdir().unwrap();
        let ledger = TaskLedger::new(directory.path()).await.unwrap();
        let mut task = TaskSpec::new("two stages");
        task.stages = vec![
            TaskStageSpec {
                id: "one".into(),
                objective: "first".into(),
                depends_on: Vec::new(),
                acceptance_criteria: Vec::new(),
                constraints: Vec::new(),
                verification_commands: Vec::new(),
                requires_mutation: None,
            },
            TaskStageSpec {
                id: "two".into(),
                objective: "second".into(),
                depends_on: vec!["one".into()],
                acceptance_criteria: Vec::new(),
                constraints: Vec::new(),
                verification_commands: Vec::new(),
                requires_mutation: None,
            },
        ];
        let record = ledger
            .create("session", task, Some("stable-id"))
            .await
            .unwrap();
        assert_eq!(record.stages.len(), 2);
        ledger.start_task("stable-id").await.unwrap();
        ledger.start_stage("stable-id", "one").await.unwrap();
        ledger
            .complete_stage("stable-id", "one", serde_json::json!({"status":"ok"}))
            .await
            .unwrap();
        ledger.start_stage("stable-id", "two").await.unwrap();

        let reopened = TaskLedger::new(directory.path()).await.unwrap();
        assert_eq!(reopened.recover_interrupted().await.unwrap(), 1);
        let interrupted = reopened.get("stable-id").await.unwrap();
        assert_eq!(interrupted.status, TaskStatus::Interrupted);
        assert_eq!(interrupted.stages[0].status, StageStatus::Succeeded);
        assert_eq!(interrupted.stages[1].status, StageStatus::Interrupted);
        let resumed = reopened.prepare_resume("stable-id").await.unwrap();
        assert_eq!(resumed.stages[0].status, StageStatus::Succeeded);
        assert_eq!(resumed.stages[1].status, StageStatus::Pending);
    }

    #[tokio::test]
    async fn invalid_ids_and_illegal_completion_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let ledger = TaskLedger::new(directory.path()).await.unwrap();
        assert!(
            ledger
                .create("s", TaskSpec::new("x"), Some("../x"))
                .await
                .is_err()
        );
        ledger
            .create("s", TaskSpec::new("x"), Some("x"))
            .await
            .unwrap();
        assert!(ledger.complete_task("x", Value::Null).await.is_err());
    }
}
