use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rwkv_agent_core::{
    AgentEvent, AgentLoop, AgentRunRequest, AnswerDecision, ArgumentSpec, CancellationToken,
    RequestIdentity, RunContext, RunLimits, SERVICE_API_VERSION, TOOL_CALL_JSON_PREFIX, TaskSpec,
    ToolCall, ToolDefinition, ToolExecutor, ToolRegistry, VecEventSink,
};
use serde_json::{Value, json};
use tokio::sync::{Mutex, Notify, mpsc};

use crate::cloud_plugin::{
    CloudLease, CloudPluginClient, CloudPluginConfig, CloudPluginFallback, CloudStateReference,
};
use crate::command::{CommandPolicy, SandboxedCommand};
use crate::data_client::DataPlaneClient;
use crate::debug_trace::{
    DebugCapture, DebugTraceConfig, DebugTraceEvent, DebugTraceFileKind, DebugTraceFilter,
    DebugTraceHandle, DebugTraceManifest, DebugTracePage, DebugTraceStart, DebugTraceStore,
};
use crate::prompt;
use crate::research::ResearchRunner;
use crate::session::{Exchange, SessionStore};
use crate::sidecar::{GateDecision, SidecarClient, SidecarState};
use crate::task_ledger::{StageStatus, TaskLedger, TaskRecord, TaskStatus};

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub runtime_revision: String,
    pub model_urls: Vec<String>,
    pub data_plane_url: String,
    pub session_dir: PathBuf,
    pub tool_gate_threshold: f64,
    pub pasted_text_gate_threshold: f64,
    pub long_text_capture_chars: usize,
    pub chat_state_capacity: usize,
    pub max_tool_steps: usize,
    pub max_model_tokens_per_turn: u32,
    pub direct_chat_max_tokens: u32,
    pub max_run_elapsed: Duration,
    pub shutdown_grace: Duration,
    pub cloud_plugin: CloudPluginConfig,
    pub command: CommandPolicy,
    pub debug_trace: DebugTraceConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            runtime_revision: env!("CARGO_PKG_VERSION").into(),
            model_urls: vec!["http://127.0.0.1:8417".into()],
            data_plane_url: "http://127.0.0.1:8121".into(),
            session_dir: PathBuf::from("var/rust-agent-sessions"),
            tool_gate_threshold: -3.2,
            pasted_text_gate_threshold: -5.5,
            long_text_capture_chars: 4000,
            chat_state_capacity: 3,
            max_tool_steps: 6,
            max_model_tokens_per_turn: 192,
            direct_chat_max_tokens: 96,
            max_run_elapsed: Duration::from_secs(600),
            shutdown_grace: Duration::from_secs(200),
            cloud_plugin: CloudPluginConfig::default(),
            command: CommandPolicy::default(),
            debug_trace: DebugTraceConfig::default(),
        }
    }
}

#[derive(Clone)]
pub struct AgentService {
    config: Arc<RuntimeConfig>,
    sidecar: SidecarClient,
    data: DataPlaneClient,
    sessions: SessionStore,
    chat_states: Arc<Mutex<ChatStateCache>>,
    command: SandboxedCommand,
    cloud_plugin: CloudPluginClient,
    task_ledger: TaskLedger,
    active_tasks: Arc<Mutex<HashMap<String, CancellationToken>>>,
    task_completion: Arc<Notify>,
    recovered_tasks: usize,
    debug_trace: DebugTraceStore,
}

#[derive(Clone, Debug)]
struct CachedChatState {
    session_id: String,
    hot_state: Option<SidecarState>,
    state_ref: Option<CloudStateReference>,
    state_version: u64,
    active_lease: Option<CloudLease>,
    blocked_error: Option<String>,
    history_len: usize,
    stop_reason: String,
    last_used: u64,
}

impl CachedChatState {
    fn hot_state(&self) -> Result<&SidecarState, String> {
        self.hot_state
            .as_ref()
            .ok_or_else(|| "chat State is not resident on a Worker".into())
    }

    fn state_identity(&self) -> String {
        self.hot_state
            .as_ref()
            .map(|state| state.state_id.clone())
            .or_else(|| self.state_ref.as_ref().map(|state| state.state_id.clone()))
            .unwrap_or_else(|| "unavailable".into())
    }

    fn is_durable(&self) -> bool {
        self.hot_state.is_none() && self.state_ref.is_some() && self.blocked_error.is_none()
    }
}

#[derive(Default)]
struct ChatStateCache {
    values: HashMap<String, CachedChatState>,
    clock: u64,
    metrics: HashMap<&'static str, u64>,
}

impl ChatStateCache {
    fn count(&mut self, key: &'static str) {
        *self.metrics.entry(key).or_default() += 1;
    }

    fn take(
        &mut self,
        session_id: &str,
        history_len: usize,
    ) -> (Option<CachedChatState>, Option<CachedChatState>) {
        let value = self.values.remove(session_id);
        match value {
            Some(value) if value.history_len == history_len => {
                self.count("hits");
                (Some(value), None)
            }
            Some(value) => {
                self.count("transcript_mismatches");
                (None, Some(value))
            }
            None => {
                self.count("misses");
                (None, None)
            }
        }
    }

    fn put(&mut self, mut value: CachedChatState, capacity: usize) -> Vec<CachedChatState> {
        self.clock += 1;
        value.last_used = self.clock;
        let mut evicted = self
            .values
            .insert(value.session_id.clone(), value)
            .into_iter()
            .collect::<Vec<_>>();
        while self.values.len() > capacity {
            let key = self
                .values
                .iter()
                .min_by_key(|(_, value)| value.last_used)
                .map(|(key, _)| key.clone());
            if let Some(key) = key {
                if let Some(value) = self.values.remove(&key) {
                    evicted.push(value);
                    self.count("evictions");
                }
            } else {
                break;
            }
        }
        self.count("stores");
        evicted
    }

    fn pop(&mut self, session_id: &str) -> Option<CachedChatState> {
        let value = self.values.remove(session_id);
        if value.is_some() {
            self.count("invalidations");
        }
        value
    }
}

#[derive(Clone)]
struct RuntimeTools {
    data: DataPlaneClient,
    command: SandboxedCommand,
    workspace: Option<WorkspaceExecution>,
}

#[derive(Clone, Debug)]
struct WorkspaceExecution {
    task_spec: TaskSpec,
    requires_mutation: bool,
    calls: usize,
    mutations: usize,
    verification_attempts: usize,
    verification_passed: bool,
    last_status: String,
    last_command: String,
    last_stdout: String,
    last_stderr: String,
    failure_evidence: String,
    inspection_evidence: String,
    grounding_evidence: String,
    inspected_files: HashSet<String>,
    inspected_contents: HashMap<String, String>,
    inspection_target_count: usize,
    existing_files: HashSet<String>,
    inspectable_files: HashSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MutationPlan {
    tool: &'static str,
    path: String,
}

impl WorkspaceExecution {
    #[cfg(test)]
    fn new(task: &str) -> Self {
        Self::from_task_spec(&TaskSpec::new(task), "")
    }

    fn from_task_spec(task: &TaskSpec, display_directory: &str) -> Self {
        let mut task_spec = task.clone();
        task_spec.objective = normalize_workspace_value(&task_spec.objective, display_directory);
        for value in &mut task_spec.acceptance_criteria {
            *value = normalize_workspace_value(value, display_directory);
        }
        for value in &mut task_spec.constraints {
            *value = normalize_workspace_value(value, display_directory);
        }
        for value in &mut task_spec.verification_commands {
            *value = normalize_workspace_value(value, display_directory);
        }
        task_spec.working_directory = Some("/workspace".into());
        Self {
            requires_mutation: task_spec
                .requires_mutation
                .unwrap_or_else(|| task_requires_mutation(&task_spec.objective)),
            task_spec,
            calls: 0,
            mutations: 0,
            verification_attempts: 0,
            verification_passed: false,
            last_status: String::new(),
            last_command: String::new(),
            last_stdout: String::new(),
            last_stderr: String::new(),
            failure_evidence: String::new(),
            inspection_evidence: String::new(),
            grounding_evidence: String::new(),
            inspected_files: HashSet::new(),
            inspected_contents: HashMap::new(),
            inspection_target_count: 3,
            existing_files: HashSet::new(),
            inspectable_files: HashSet::new(),
        }
    }

    fn from_task_spec_with_inventory(
        task: &TaskSpec,
        display_directory: &str,
        inventory: &[String],
    ) -> Self {
        let mut state = Self::from_task_spec(task, display_directory);
        state.existing_files = inventory
            .iter()
            .map(String::as_str)
            .filter_map(inventory_path)
            .collect();
        state.inspectable_files = inventory
            .iter()
            .filter(|row| !row.ends_with("(0 bytes)"))
            .map(String::as_str)
            .filter_map(inventory_path)
            .collect();
        let nonempty_files = inventory
            .iter()
            .filter(|row| !row.ends_with("(0 bytes)"))
            .count();
        let referenced_existing = task_reference_paths(&state.task_spec)
            .into_iter()
            .filter(|path| state.existing_files.contains(path))
            .collect::<HashSet<_>>()
            .len();
        state.inspection_target_count = if referenced_existing > 0 {
            referenced_existing.clamp(1, 4)
        } else {
            nonempty_files.clamp(1, 4)
        };
        state
    }

    fn record(&mut self, command: &str, result: &Value, changed: bool) {
        self.calls += 1;
        if changed {
            self.mutations += 1;
            self.inspected_files.clear();
            self.inspected_contents.clear();
            self.inspection_evidence.clear();
        }
        let status = result
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("error");
        let stdout = result
            .get("stdout")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        let explicit_combined_verifier = changed
            && status == "ok"
            && command.to_lowercase().contains("python3 -c")
            && stdout.lines().any(|line| line.trim() == "VALID");
        let verification = if self.task_spec.verification_commands.is_empty() {
            (command_looks_like_verification(command)
                && (status == "ok" || !command_looks_like_inspection(command))
                && (!self.requires_mutation || !command_looks_like_inspection(command)))
                || explicit_combined_verifier
        } else {
            self.task_spec
                .verification_commands
                .iter()
                .any(|expected| expected.trim() == command.trim())
        };
        if verification {
            self.verification_attempts += 1;
        }
        self.last_status = status.to_string();
        self.last_command = command.trim().to_string();
        self.last_stdout = stdout.to_string();
        self.last_stderr = result
            .get("stderr")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .or_else(|| result.get("message").and_then(Value::as_str))
            .unwrap_or_default()
            .trim()
            .to_string();
        if status != "ok" && (verification || self.failure_evidence.is_empty()) {
            self.failure_evidence = format!(
                "Command `{}` failed with: {}",
                command.trim(),
                self.last_stderr.chars().take(6_000).collect::<String>(),
            );
        }
        if status == "ok"
            && let Some(path) = command
                .strip_prefix("write_file(")
                .and_then(|value| value.strip_suffix(')'))
        {
            self.existing_files.insert(normalize_workspace_path(path));
        }
        if status == "ok" && command_looks_like_inspection(command) && !self.last_stdout.is_empty()
        {
            let explicit_path = command
                .strip_prefix("read_file(")
                .and_then(|value| value.strip_suffix(')'))
                .map(normalize_workspace_path);
            if let Some(path) = &explicit_path {
                self.inspected_contents
                    .insert(path.clone(), self.last_stdout.clone());
            }
            let paths = explicit_path
                .into_iter()
                .chain(workspace_path_tokens(command));
            for path in paths {
                self.existing_files.insert(path.clone());
                self.inspected_files.insert(path);
            }
            let evidence = format!(
                "Command `{}` showed:\n{}",
                command.trim(),
                self.last_stdout.chars().take(1200).collect::<String>(),
            );
            if !self.inspection_evidence.contains(&evidence) {
                if !self.inspection_evidence.is_empty() {
                    self.inspection_evidence.push_str("\n\n");
                }
                self.inspection_evidence.push_str(&evidence);
                self.inspection_evidence = self
                    .inspection_evidence
                    .chars()
                    .take(6_000)
                    .collect::<String>();
            }
            if !self.grounding_evidence.contains(&evidence) {
                if !self.grounding_evidence.is_empty() {
                    self.grounding_evidence.push_str("\n\n");
                }
                self.grounding_evidence.push_str(&evidence);
                self.grounding_evidence = self
                    .grounding_evidence
                    .chars()
                    .take(10_000)
                    .collect::<String>();
            }
        }
        self.verification_passed =
            status == "ok" && verification && (!self.requires_mutation || self.mutations > 0);
    }

    fn next_instruction(&self) -> Option<String> {
        if self.verification_passed {
            return None;
        }
        let task = self
            .task_spec
            .objective
            .chars()
            .take(600)
            .collect::<String>();
        let declared_verification = self.declared_verification_instruction();
        if self.calls == 0 {
            return Some(format!(
                "Controller phase=INSPECT. Read only the named existing inputs needed for the task. Do not guess paths: the startup inventory is authoritative. After the relevant inputs are inspected, the Controller will run the declared verifier before the first repair. Original workspace task: {task}{declared_verification}"
            ));
        }
        if self.should_run_verifier() || self.should_request_model_verifier() {
            return Some(format!(
                "Controller phase=VERIFY; independent verification is required now. Run the exact declared verifier when one exists, otherwise run one real focused or full check. Do not edit or inspect again before its exit status is observed. Original workspace task: {task}{declared_verification}"
            ));
        }
        if let Some(plan) = self.mutation_plan()
            && self.mutation_ready(&plan)
        {
            return Some(format!(
                "Controller phase=MUTATE. target_path={} controller_tool={}. Latest verifier evidence:\n{}\nExact current target evidence:\n{}\nGrounded read-only task evidence (use to infer requirements; never copy tests/specifications into the target):\n{}\nReturn only the complete replacement file text inside one strict <answer> envelope. Infer the corrective source logic from the actual assertion/exception and change the defect; never return unchanged content. Do not JSON-escape it and do not add reasoning, Markdown fences, tests, specification, comments, or unrelated code. The Controller will deterministically package the answer as the required file ToolCall, run it through the strict Registry and Sandbox, then execute the declared verifier. Do not read or rerun the unchanged verifier. Original workspace task: {task}{declared_verification}",
                plan.path,
                plan.tool,
                self.failure_evidence
                    .chars()
                    .take(6_000)
                    .collect::<String>(),
                self.inspected_contents
                    .get(&plan.path)
                    .map(String::as_str)
                    .unwrap_or_default()
                    .chars()
                    .take(4_500)
                    .collect::<String>(),
                self.grounding_evidence
                    .chars()
                    .take(10_000)
                    .collect::<String>(),
            ));
        }
        if self.verification_attempts > 0 && !self.verification_passed {
            let plan = self
                .mutation_plan()
                .map(|plan| format!(" required_tool={} target_path={}.", plan.tool, plan.path))
                .unwrap_or_default();
            return Some(format!(
                "Controller phase=DIAGNOSE.{plan} The declared verifier failed with:\n{}\nRead only the named repair target if current source evidence is missing; otherwise make the smallest corrective mutation. Do not repeat the failed verifier unchanged. Original workspace task: {task}{declared_verification}",
                self.failure_evidence
                    .chars()
                    .take(6_000)
                    .collect::<String>(),
            ));
        }
        if self.last_status != "ok" {
            return Some(format!(
                "Controller phase=DIAGNOSE. The last real action `{}` failed with: {}. Do not repeat it unchanged. Read one grounded relevant source or choose the controller-required mutation target. Original workspace task: {task}{declared_verification}",
                self.last_command.chars().take(300).collect::<String>(),
                self.last_stderr.chars().take(800).collect::<String>(),
            ));
        }
        let inspected = self.inspected_files.len();
        Some(format!(
            "Controller phase=INSPECT. Read a distinct named input or source that is relevant to the objective; {inspected}/{} inspection targets have been observed. Do not repeat a completed read and do not invent a missing path. Original workspace task: {task}{declared_verification}",
            self.inspection_target_count,
        ))
    }

    fn committed_answer(&self) -> Option<String> {
        if !self.verification_passed {
            return None;
        }
        let evidence = if !self.last_stdout.is_empty() {
            self.last_stdout.chars().take(800).collect::<String>()
        } else {
            format!("{} exited successfully", self.last_command)
        };
        Some(format!(
            "Completed the workspace task. Final verification passed: {evidence}"
        ))
    }

    fn should_refresh_worker(&self) -> bool {
        self.mutation_plan()
            .is_some_and(|plan| self.mutation_ready(&plan))
            || self.last_status != "ok"
    }

    fn should_run_verifier(&self) -> bool {
        if self.verification_passed || self.task_spec.verification_commands.is_empty() {
            return false;
        }
        let initial_baseline = self.verification_attempts == 0
            && self.mutations == 0
            && self.inspected_files.len() >= self.inspection_target_count;
        let after_mutation = self.last_status == "ok"
            && command_looks_like_mutation(&self.last_command)
            && !self.last_command.starts_with("read_file(");
        initial_baseline || after_mutation
    }

    fn should_request_model_verifier(&self) -> bool {
        self.task_spec.verification_commands.is_empty()
            && self.mutations > 0
            && self.last_status == "ok"
            && command_looks_like_mutation(&self.last_command)
    }

    fn mutation_ready(&self, plan: &MutationPlan) -> bool {
        if !self.requires_mutation
            || self.verification_passed
            || self.should_run_verifier()
            || self.should_request_model_verifier()
        {
            return false;
        }
        let diagnosis_exists = self.verification_attempts > 0
            || self.task_spec.verification_commands.is_empty()
                && self.inspected_files.len() >= self.inspection_target_count;
        diagnosis_exists && (plan.tool == "write_file" || self.inspected_files.contains(&plan.path))
    }

    fn phase_rejection(&self, call: &ToolCall) -> Option<Value> {
        if matches!(call.name.as_str(), "write_file" | "edit_file") {
            let path = call
                .arguments
                .get("path")
                .and_then(Value::as_str)
                .map(normalize_workspace_path)
                .unwrap_or_default();
            if self.path_is_protected(&path) {
                return Some(json!({
                    "status":"rejected",
                    "code":"protected_path",
                    "phase":"mutate",
                    "target_path":path,
                    "message":"TaskSpec constraints mark this path as read-only; select the controller-required repair target"
                }));
            }
        }
        if let Some(plan) = self.mutation_plan()
            && self.mutation_ready(&plan)
        {
            let actual_path = call
                .arguments
                .get("path")
                .and_then(Value::as_str)
                .map(normalize_workspace_path)
                .unwrap_or_default();
            if call.name != plan.tool || actual_path != plan.path {
                return Some(json!({
                    "status":"rejected",
                    "code":"wrong_phase_action",
                    "phase":"mutate",
                    "required_tool":plan.tool,
                    "target_path":plan.path,
                    "message":"Use exactly the required mutation tool and grounded target path; do not inspect, verify, or guess another path"
                }));
            }
        }
        if self.should_run_verifier() {
            let expected = self
                .task_spec
                .verification_commands
                .iter()
                .find(|command| !command.trim().is_empty())?;
            let actual = call.arguments.get("command").and_then(Value::as_str);
            if call.name != "run_command" || actual != Some(expected.as_str()) {
                return Some(json!({
                    "status":"rejected",
                    "code":"wrong_phase_action",
                    "phase":"verify",
                    "required_tool":"run_command",
                    "required_command":expected,
                    "message":"Run the exact declared verifier before any further inspection or mutation"
                }));
            }
        }
        if self.should_request_model_verifier() && call.name != "run_command" {
            return Some(json!({
                "status":"rejected",
                "code":"wrong_phase_action",
                "phase":"verify",
                "required_tool":"run_command",
                "message":"The workspace changed; run one real focused or full verifier before another mutation"
            }));
        }
        None
    }

    fn controller_followup_call(&self) -> Option<ToolCall> {
        if self.should_run_verifier() {
            let command = self
                .task_spec
                .verification_commands
                .iter()
                .find(|command| !command.trim().is_empty())?
                .clone();
            return Some(ToolCall {
                name: "run_command".into(),
                arguments: json!({"command":command})
                    .as_object()
                    .expect("object literal")
                    .clone(),
            });
        }
        if let Some(plan) = self.mutation_plan()
            && self.verification_attempts > 0
            && !self.verification_passed
            && plan.tool == "edit_file"
            && !self.inspected_files.contains(&plan.path)
        {
            return Some(ToolCall {
                name: "read_file".into(),
                arguments: json!({"path":plan.path})
                    .as_object()
                    .expect("object literal")
                    .clone(),
            });
        }
        if self.verification_attempts == 0
            && let Some(path) = self.next_inspection_path()
        {
            return Some(ToolCall {
                name: "read_file".into(),
                arguments: json!({"path":path})
                    .as_object()
                    .expect("object literal")
                    .clone(),
            });
        }
        None
    }

    fn tool_call_prefix(&self) -> String {
        if self.verification_passed {
            return TOOL_CALL_JSON_PREFIX.to_string();
        }
        if let Some(plan) = self.mutation_plan()
            && self.mutation_ready(&plan)
        {
            return "<answer>".into();
        }
        if self.should_request_model_verifier() {
            return "<tool_call>{\"name\":\"run_command\",\"arguments\":{\"command\":".into();
        }
        TOOL_CALL_JSON_PREFIX.to_string()
    }

    fn next_inspection_path(&self) -> Option<String> {
        if self.inspected_files.len() >= self.inspection_target_count {
            return None;
        }
        let referenced = task_reference_paths(&self.task_spec)
            .into_iter()
            .filter(|path| self.inspectable_files.contains(path))
            .filter(|path| !self.inspected_files.contains(path))
            .collect::<BTreeSet<_>>();
        unique_first(referenced).or_else(|| {
            unique_first(
                self.inspectable_files
                    .iter()
                    .filter(|path| !self.inspected_files.contains(*path))
                    .cloned()
                    .collect(),
            )
        })
    }

    fn controller_tool_from_answer(&self, answer: &str) -> Option<ToolCall> {
        let plan = self.mutation_plan()?;
        if !self.mutation_ready(&plan) {
            return None;
        }
        let replacement = replacement_file_content(answer)?;
        let path = plan.path.clone();
        let arguments = match plan.tool {
            "write_file" => json!({"path":path,"content":replacement}),
            "edit_file" => {
                let current = self.inspected_contents.get(&plan.path)?;
                if file_content_equivalent(current, &replacement) {
                    return None;
                }
                json!({"path":path,"old_text":current,"new_text":replacement})
            }
            _ => return None,
        };
        Some(ToolCall {
            name: plan.tool.into(),
            arguments: arguments
                .as_object()
                .expect("controller file arguments are an object")
                .clone(),
        })
    }

    fn answer_payload_rejection(&self, answer: &str) -> Option<String> {
        let plan = self.mutation_plan()?;
        if !self.mutation_ready(&plan) {
            return None;
        }
        let Some(replacement) = replacement_file_content(answer) else {
            return Some(format!(
                "The replacement payload for `{}` was empty or exceeded the bounded file limit. Return one smaller complete replacement file body.",
                plan.path
            ));
        };
        if plan.tool == "edit_file"
            && self
                .inspected_contents
                .get(&plan.path)
                .is_some_and(|current| file_content_equivalent(current, &replacement))
        {
            return Some(format!(
                "The proposed replacement for `{}` is content-identical to the current file and cannot fix the real verifier failure. Use the target evidence and change only the defective source logic.",
                plan.path
            ));
        }
        None
    }

    fn mutation_plan(&self) -> Option<MutationPlan> {
        let protected = self.protected_paths();
        let safe = |path: &String| is_source_path(path) && !path_is_protected_by(path, &protected);

        let failure = workspace_path_tokens(&self.last_stderr)
            .into_iter()
            .filter(|path| self.existing_files.contains(path) && safe(path))
            .collect::<BTreeSet<_>>();
        let verifier = self
            .task_spec
            .verification_commands
            .iter()
            .flat_map(|value| workspace_path_tokens(value))
            .filter(safe)
            .collect::<BTreeSet<_>>();
        let objective = std::iter::once(&self.task_spec.objective)
            .chain(self.task_spec.acceptance_criteria.iter())
            .flat_map(|value| workspace_path_tokens(value))
            .filter(safe)
            .collect::<BTreeSet<_>>();
        let inventory = self
            .existing_files
            .iter()
            .filter(|path| safe(path))
            .cloned()
            .collect::<BTreeSet<_>>();

        let path = unique_path(failure)
            .or_else(|| unique_path(objective))
            .or_else(|| unique_path(verifier))
            .or_else(|| unique_path(inventory))?;
        Some(MutationPlan {
            tool: if self.existing_files.contains(&path) {
                "edit_file"
            } else {
                "write_file"
            },
            path,
        })
    }

    fn protected_paths(&self) -> BTreeSet<String> {
        self.task_spec
            .constraints
            .iter()
            .filter(|constraint| constraint_is_protective(constraint))
            .flat_map(|constraint| workspace_path_tokens(constraint))
            .collect()
    }

    fn path_is_protected(&self, path: &str) -> bool {
        path_is_protected_by(path, &self.protected_paths())
    }

    fn declared_verification_instruction(&self) -> String {
        if self.task_spec.verification_commands.is_empty() {
            return String::new();
        }
        format!(
            " Declared verification command(s), one of which must succeed exactly: {}.",
            self.task_spec.verification_commands.join(" | ")
        )
    }
}

fn inventory_path(row: &str) -> Option<String> {
    let (path, _) = row.rsplit_once(" (")?;
    (!path.trim().is_empty() && !path.starts_with("... inventory truncated"))
        .then(|| normalize_workspace_path(path))
}

fn normalize_workspace_path(path: &str) -> String {
    path.trim()
        .trim_matches(|character: char| {
            matches!(
                character,
                '`' | '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}' | ','
            )
        })
        .strip_prefix("/workspace/")
        .unwrap_or_else(|| path.trim().trim_start_matches("./"))
        .trim_end_matches('.')
        .to_string()
}

fn workspace_path_tokens(value: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let flush = |current: &mut String, tokens: &mut Vec<String>| {
        if current.is_empty() {
            return;
        }
        let path = normalize_workspace_path(current);
        let basename = path.rsplit('/').next().unwrap_or_default();
        if !path.is_empty()
            && path != "."
            && path != ".."
            && (path.ends_with('/') || basename.contains('.'))
            && !path.starts_with('/')
            && !path.split('/').any(|part| part == "..")
        {
            tokens.push(path);
        }
        current.clear();
    };
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-' | '/') {
            current.push(character);
        } else {
            flush(&mut current, &mut tokens);
        }
    }
    flush(&mut current, &mut tokens);
    tokens
}

fn task_reference_paths(task: &TaskSpec) -> BTreeSet<String> {
    std::iter::once(&task.objective)
        .chain(task.acceptance_criteria.iter())
        .chain(task.constraints.iter())
        .chain(task.verification_commands.iter())
        .flat_map(|value| workspace_path_tokens(value))
        .collect()
}

fn constraint_is_protective(value: &str) -> bool {
    let lower = value.to_lowercase();
    [
        "do not modify",
        "must not modify",
        "never modify",
        "read-only",
        "readonly",
        "不得修改",
        "禁止修改",
        "不要修改",
    ]
    .into_iter()
    .any(|marker| lower.contains(marker))
}

fn path_is_protected_by(path: &str, protected: &BTreeSet<String>) -> bool {
    protected.iter().any(|candidate| {
        let prefix = candidate.trim_end_matches('/');
        path == prefix || path.starts_with(&format!("{prefix}/"))
    })
}

fn is_source_path(path: &str) -> bool {
    let lower = path.to_lowercase();
    [
        ".py", ".rs", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".c", ".cc", ".cpp", ".h",
        ".hpp", ".rb", ".php", ".sh", ".bash", ".zsh", ".lua", ".sql",
    ]
    .into_iter()
    .any(|suffix| lower.ends_with(suffix))
}

fn unique_path(paths: BTreeSet<String>) -> Option<String> {
    (paths.len() == 1).then(|| paths.into_iter().next().expect("one path"))
}

fn unique_first(paths: BTreeSet<String>) -> Option<String> {
    paths.into_iter().next()
}

fn replacement_file_content(answer: &str) -> Option<String> {
    let answer = answer.trim();
    if answer.is_empty() {
        return None;
    }
    let value = if let Some(open) = answer.find("```") {
        let after_open = &answer[open + 3..];
        let body_start = after_open.find('\n').map_or(0, |index| index + 1);
        let body = &after_open[body_start..];
        let close = body.find("```").unwrap_or(body.len());
        &body[..close]
    } else {
        answer
    };
    let value = value.trim_matches('\n');
    (!value.trim().is_empty() && value.len() <= 64 * 1024).then(|| value.to_string())
}

fn file_content_equivalent(left: &str, right: &str) -> bool {
    left.trim_end_matches(['\r', '\n']) == right.trim_end_matches(['\r', '\n'])
}

fn normalize_workspace_value(value: &str, display_directory: &str) -> String {
    if display_directory.is_empty() {
        value.trim().to_string()
    } else {
        value
            .replace(display_directory, "/workspace")
            .trim()
            .to_string()
    }
}

struct ToolRunOptions<'a> {
    task_spec: &'a TaskSpec,
    context: &'a str,
    has_text: bool,
    gate: GateDecision,
    started: Instant,
    command: SandboxedCommand,
    workspace_label: Option<&'a str>,
    cancellation: CancellationToken,
    debug_trace: Option<&'a DebugTraceHandle>,
    stage_id: Option<&'a str>,
}

impl ToolExecutor for RuntimeTools {
    async fn execute(&mut self, call: ToolCall, context: RunContext) -> Result<Value, String> {
        context.check().map_err(|e| e.to_string())?;
        if matches!(
            call.name.as_str(),
            "run_command" | "read_file" | "write_file" | "edit_file"
        ) {
            if let Some(result) = self
                .workspace
                .as_ref()
                .and_then(|workspace| workspace.phase_rejection(&call))
            {
                let action = format!("{}(controller_phase_rejected)", call.name);
                if let Some(workspace) = &mut self.workspace {
                    workspace.record(&action, &result, false);
                }
                return Ok(result);
            }
            let before = if self.workspace.is_some() {
                Some(self.command.workspace_fingerprint().await?)
            } else {
                None
            };
            let (action, outcome) = match call.name.as_str() {
                "run_command" => {
                    let command = required_string(&call.arguments, "run_command", "command")?;
                    (command.to_string(), self.command.execute(command).await)
                }
                "read_file" => {
                    let path = required_string(&call.arguments, "read_file", "path")?;
                    (
                        format!("read_file({path})"),
                        self.command.read_file(path).await,
                    )
                }
                "write_file" => {
                    let path = required_string(&call.arguments, "write_file", "path")?;
                    let content = required_string(&call.arguments, "write_file", "content")?;
                    (
                        format!("write_file({path})"),
                        self.command.write_file(path, content).await,
                    )
                }
                "edit_file" => {
                    let path = required_string(&call.arguments, "edit_file", "path")?;
                    let old_text = required_string(&call.arguments, "edit_file", "old_text")?;
                    let new_text = required_string(&call.arguments, "edit_file", "new_text")?;
                    (
                        format!("edit_file({path})"),
                        self.command.edit_file(path, old_text, new_text).await,
                    )
                }
                _ => unreachable!(),
            };
            // File-tool policy failures (missing files, unsafe paths, invalid
            // exact edits) are task observations, not executor transport
            // failures. Preserve and record them so the workspace controller
            // can detect no progress, refresh a stale worker, and provide the
            // real error to the next action.
            let result =
                outcome.unwrap_or_else(|message| json!({"status":"error","message":message}));
            if let (Some(workspace), Some(before)) = (&mut self.workspace, before) {
                let after = self.command.workspace_fingerprint().await?;
                workspace.record(&action, &result, before != after);
            }
            Ok(result)
        } else {
            self.data.execute(call, context).await
        }
    }

    async fn validate_answer(
        &mut self,
        answer: &str,
        _context: RunContext,
    ) -> Result<AnswerDecision, String> {
        if let Some(feedback) = self
            .workspace
            .as_ref()
            .and_then(|workspace| workspace.answer_payload_rejection(answer))
        {
            return Ok(AnswerDecision::Retry {
                feedback,
                require_tool: false,
            });
        }
        match self
            .workspace
            .as_ref()
            .and_then(WorkspaceExecution::next_instruction)
        {
            Some(feedback) => Ok(AnswerDecision::Retry {
                feedback,
                require_tool: true,
            }),
            None => Ok(AnswerDecision::Accept),
        }
    }

    fn continuation_feedback(&self) -> String {
        self.workspace
            .as_ref()
            .and_then(WorkspaceExecution::next_instruction)
            .unwrap_or_default()
    }

    fn answer_retry_root_prompt(&self, feedback: &str) -> Option<String> {
        self.workspace.as_ref().map(|workspace| {
            let phase = workspace.next_instruction().unwrap_or_default();
            format!(
                "Protocol repair required. {feedback} Regenerate one smaller complete envelope; never append role labels, a fake Tool result, or a second envelope. {phase}"
            )
        })
    }

    fn require_tool_after_observation(&self) -> bool {
        self.workspace
            .as_ref()
            .and_then(WorkspaceExecution::next_instruction)
            .is_some()
    }

    fn committed_answer_after_observation(&self) -> Option<String> {
        self.workspace
            .as_ref()
            .and_then(WorkspaceExecution::committed_answer)
    }

    fn recovery_root_prompt_after_observation(&self) -> Option<String> {
        self.workspace
            .as_ref()
            .and_then(WorkspaceExecution::next_instruction)
    }

    fn refresh_worker_after_observation(&self) -> bool {
        self.workspace
            .as_ref()
            .is_some_and(WorkspaceExecution::should_refresh_worker)
    }

    fn refresh_worker_after_protocol_rejection(&self) -> bool {
        self.workspace.is_some()
    }

    fn tool_call_prefix(&self) -> String {
        self.workspace
            .as_ref()
            .map(WorkspaceExecution::tool_call_prefix)
            .unwrap_or_else(|| TOOL_CALL_JSON_PREFIX.to_string())
    }

    fn controller_tool_after_observation(&mut self) -> Option<ToolCall> {
        self.workspace
            .as_ref()
            .and_then(WorkspaceExecution::controller_followup_call)
    }

    fn controller_tool_from_answer(&mut self, answer: &str) -> Option<ToolCall> {
        self.workspace
            .as_ref()
            .and_then(|workspace| workspace.controller_tool_from_answer(answer))
    }
}

impl AgentService {
    pub async fn new(config: RuntimeConfig) -> Result<Self, String> {
        if config.runtime_revision.trim().is_empty() {
            return Err("runtime_revision must not be empty".into());
        }
        if !(-20.0..=20.0).contains(&config.tool_gate_threshold)
            || !(-20.0..=20.0).contains(&config.pasted_text_gate_threshold)
        {
            return Err("tool Gate thresholds must be between -20 and 20".into());
        }
        if config.chat_state_capacity == 0
            || config.max_tool_steps == 0
            || config.max_model_tokens_per_turn == 0
            || config.direct_chat_max_tokens == 0
            || config.direct_chat_max_tokens > 1024
            || config.max_run_elapsed.is_zero()
            || config.shutdown_grace.is_zero()
        {
            return Err("State and tool capacities must be positive".into());
        }
        let sidecar = SidecarClient::new(config.model_urls.clone())?;
        let data = DataPlaneClient::new(config.data_plane_url.clone())?;
        let sessions = SessionStore::new(config.session_dir.clone()).await?;
        let task_ledger = TaskLedger::new(config.session_dir.join("task-ledger")).await?;
        let recovered_tasks = task_ledger.recover_interrupted().await?;
        let command = SandboxedCommand::new(config.command.clone());
        let cloud_plugin = CloudPluginClient::new(config.cloud_plugin.clone())?;
        cloud_plugin.initialize(&config.runtime_revision).await?;
        let debug_trace = DebugTraceStore::new(
            config.debug_trace.clone(),
            config.runtime_revision.clone(),
            config.model_urls.join(","),
            "provider-owned".into(),
            "rwkv-recurrent-state-http-v1".into(),
            json!({
                "runtime_revision":config.runtime_revision,
                "model_urls":config.model_urls,
                "data_plane_url":config.data_plane_url,
                "command_enabled":config.command.enabled,
                "cloud_plugin_enabled":config.cloud_plugin.enabled,
                "max_tool_steps":config.max_tool_steps,
                "max_model_tokens_per_turn":config.max_model_tokens_per_turn,
            }),
        )
        .await?;
        Ok(Self {
            config: Arc::new(config),
            sidecar,
            data,
            sessions,
            chat_states: Arc::new(Mutex::new(ChatStateCache::default())),
            command,
            cloud_plugin,
            task_ledger,
            active_tasks: Arc::new(Mutex::new(HashMap::new())),
            task_completion: Arc::new(Notify::new()),
            recovered_tasks,
            debug_trace,
        })
    }

    pub fn liveness(&self) -> Value {
        json!({
            "status":"alive",
            "api_version":SERVICE_API_VERSION,
            "control_plane":"rust",
            "runtime_revision":self.config.runtime_revision,
        })
    }

    pub async fn readiness(&self) -> Value {
        let (model, model_ready, model_error) = match self.sidecar.health().await {
            Ok(model) => {
                let ready = !model.is_empty()
                    && model.iter().all(|item| {
                        matches!(
                            item.get("status").and_then(Value::as_str),
                            Some("ready" | "ok")
                        )
                    });
                (model, ready, String::new())
            }
            Err(error) => (Vec::new(), false, error),
        };
        let (data, data_ready, data_error) = match self.data.health().await {
            Ok(data) => {
                let ready = matches!(
                    data.get("status").and_then(Value::as_str),
                    Some("ready" | "ok")
                );
                (data, ready, String::new())
            }
            Err(error) => (Value::Null, false, error),
        };
        let (state_capacity, state_capacity_ready) = state_capacity_status(&model, model_ready);
        let states = self.chat_states.lock().await;
        let hot_chat_states = states
            .values
            .values()
            .filter(|state| state.hot_state.is_some())
            .count();
        let durable_chat_states = states
            .values
            .values()
            .filter(|state| state.is_durable())
            .count();
        let blocked_chat_states = states
            .values
            .values()
            .filter(|state| state.blocked_error.is_some())
            .count();
        let active_tasks = self.active_tasks.lock().await.len();
        let sandbox_ready = !self.config.command.enabled || self.command.available();
        let debug_trace = self.debug_trace.readiness().await;
        let cloud_plugin = self.cloud_plugin.readiness().await;
        let cloud_plugin_ready = !self.cloud_plugin.blocks_readiness().await;
        let ready = model_ready
            && data_ready
            && sandbox_ready
            && state_capacity_ready
            && cloud_plugin_ready;
        json!({
            "status":if ready {"ready"} else {"unavailable"},
            "api_version":SERVICE_API_VERSION,
            "control_plane":"rust",
            "runtime_revision":self.config.runtime_revision,
            "tools":self.tool_names(),
            "model":model,
            "data_plane":data,
            "components":{
                "model_sidecar":{"status":if model_ready {"ready"} else {"unavailable"},"error":model_error},
                "data_plane":{"status":if data_ready {"ready"} else {"unavailable"},"error":data_error},
                "sandbox":{"status":if sandbox_ready {"ready"} else {"unavailable"},"required":self.config.command.enabled,
                    "mode":self.command.sandbox_mode()},
                "state_capacity":state_capacity,
                "task_ledger":{"status":"ready","schema_version":crate::task_ledger::LEDGER_SCHEMA_VERSION,
                    "active_tasks":active_tasks},
                "debug_trace":debug_trace,
                "statepool_cloud_plugin":cloud_plugin,
            },
            "configuration":{
                "runtime_revision":self.config.runtime_revision,
                "model_urls":self.config.model_urls,
                "data_plane_url":self.config.data_plane_url,
                "session_dir":self.config.session_dir,
                "command_enabled":self.config.command.enabled,
                "debug_trace_mode":self.config.debug_trace.mode,
                "debug_trace_directory":self.config.debug_trace.directory,
                "debug_trace_api":self.config.debug_trace.api_enabled,
                "cloud_plugin_enabled":self.config.cloud_plugin.enabled,
                "cloud_plugin_fallback":self.config.cloud_plugin.fallback,
            },
            "context":{
                "mode":"recurrent_session_state_with_transcript_fallback",
                "history_messages":12,
                "long_term_memory":false,
                "session_state":{
                    "enabled":true,"mode":if self.cloud_plugin.state_lifecycle_enabled() {"statepool_hot_warm_cold"} else {"gpu_recurrent_lru"},
                    "capacity":self.config.chat_state_capacity,
                    "allocated":hot_chat_states,"durable":durable_chat_states,"blocked":blocked_chat_states,
                    "cached_total":states.values.len(),"metrics":states.metrics,
                }
            },
            "tool_gate":{"mode":"semantic_single_token","threshold":self.config.tool_gate_threshold,
                "pasted_text_threshold":self.config.pasted_text_gate_threshold},
            "state_parallel_search":{"enabled":true,"endpoint":"/v1/agent/run_stateful","max_branches":4,"max_rounds":3},
            "command":{"enabled":self.config.command.enabled,"available":self.command.available(),"sandbox":self.command.sandbox_mode()},
            "agent_limits":{"max_tool_steps":self.config.max_tool_steps,"max_model_tokens_per_turn":self.config.max_model_tokens_per_turn,
                "direct_chat_max_tokens":self.config.direct_chat_max_tokens,
                "max_run_seconds":self.config.max_run_elapsed.as_secs(),
                "shutdown_grace_seconds":self.config.shutdown_grace.as_secs()},
            "task_ledger":{"enabled":true,"directory":self.task_ledger.root(),"recovered_on_startup":self.recovered_tasks},
            "debug_trace":debug_trace,
        })
    }

    /// Compatibility alias for clients released before liveness and
    /// readiness became separate canonical endpoints.
    pub async fn health(&self) -> Result<Value, String> {
        Ok(self.readiness().await)
    }

    pub fn debug_trace_api_enabled(&self) -> bool {
        self.debug_trace.api_enabled()
    }

    pub async fn debug_traces_for_owner(
        &self,
        owner_id: &str,
        filter: DebugTraceFilter,
    ) -> Result<DebugTracePage, String> {
        self.debug_trace.list_for_owner(owner_id, filter).await
    }

    pub async fn debug_trace_manifest_for_owner(
        &self,
        trace_id: &str,
        owner_id: &str,
    ) -> Result<DebugTraceManifest, String> {
        self.debug_trace
            .manifest_for_owner(trace_id, owner_id)
            .await
    }

    pub async fn debug_trace_events_for_owner(
        &self,
        trace_id: &str,
        owner_id: &str,
        after_sequence: u64,
        limit: usize,
    ) -> Result<Vec<DebugTraceEvent>, String> {
        self.debug_trace
            .events_for_owner(trace_id, owner_id, after_sequence, limit)
            .await
    }

    pub async fn debug_trace_file_for_owner(
        &self,
        trace_id: &str,
        owner_id: &str,
        kind: DebugTraceFileKind,
    ) -> Result<Vec<u8>, String> {
        self.debug_trace
            .file_for_owner(trace_id, owner_id, kind)
            .await
    }

    pub async fn gate(
        &self,
        message: &str,
        threshold: Option<f64>,
        context: &str,
        has_pasted_text: bool,
    ) -> Result<Value, String> {
        let decision = self
            .sidecar
            .gate(
                message,
                threshold.unwrap_or(if has_pasted_text {
                    self.config.pasted_text_gate_threshold
                } else {
                    self.config.tool_gate_threshold
                }),
                context,
                has_pasted_text,
            )
            .await?;
        serde_json::to_value(decision).map_err(|e| e.to_string())
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
        session_id: &str,
    ) -> Result<Value, String> {
        self.call_tool_with_workspace(name, arguments, session_id, None)
            .await
    }

    pub async fn call_tool_with_workspace(
        &self,
        name: &str,
        arguments: Value,
        session_id: &str,
        working_directory: Option<&str>,
    ) -> Result<Value, String> {
        if matches!(
            name,
            "run_command" | "read_file" | "write_file" | "edit_file"
        ) {
            let executor = match working_directory {
                Some(value) => self.command.scoped(value).await?,
                None if name == "run_command" => self.command.clone(),
                None => return Err(format!("{name} requires working_directory")),
            };
            return match name {
                "run_command" => {
                    executor
                        .execute(required_string_value(&arguments, name, "command")?)
                        .await
                }
                "read_file" => {
                    executor
                        .read_file(required_string_value(&arguments, name, "path")?)
                        .await
                }
                "write_file" => {
                    executor
                        .write_file(
                            required_string_value(&arguments, name, "path")?,
                            required_string_value(&arguments, name, "content")?,
                        )
                        .await
                }
                "edit_file" => {
                    executor
                        .edit_file(
                            required_string_value(&arguments, name, "path")?,
                            required_string_value(&arguments, name, "old_text")?,
                            required_string_value(&arguments, name, "new_text")?,
                        )
                        .await
                }
                _ => unreachable!(),
            };
        }
        self.data.call_tool(name, arguments, session_id, "").await
    }

    pub async fn call_tool_with_identity(
        &self,
        name: &str,
        arguments: Value,
        identity: &RequestIdentity,
        working_directory: Option<&str>,
    ) -> Result<Value, String> {
        let start = self
            .debug_trace
            .start_lazy(
                &identity.request_id,
                &identity.owner_id,
                &identity.session_id,
                None,
                || json!({"endpoint":"/v1/tools/call","name":name,"arguments":arguments,"working_directory":working_directory}),
            )
            .await;
        debug_record(
            start.handle.as_ref(),
            "tools",
            "tool_registry",
            "tool_requested",
            None,
            || json!({"name":name,"arguments":arguments,"working_directory":working_directory}),
        )
        .await;
        let result = self
            .call_tool_with_workspace(name, arguments, &identity.session_id, working_directory)
            .await;
        debug_record(
            start.handle.as_ref(),
            "tools",
            "sandbox",
            "tool_completed",
            None,
            || match &result {
                Ok(value) => json!({"name":name,"status":"ok","result":value}),
                Err(error) => json!({"name":name,"status":"error","error":error}),
            },
        )
        .await;
        self.finish_standalone_trace(start, result, "tool_call_finished")
            .await
    }

    pub async fn run(&self, message: &str, session_id: &str) -> Result<Value, String> {
        self.run_with_workspace(message, session_id, None).await
    }

    pub async fn run_with_workspace(
        &self,
        message: &str,
        session_id: &str,
        working_directory: Option<&str>,
    ) -> Result<Value, String> {
        self.run_task(
            TaskSpec::legacy(
                message,
                working_directory.map(std::string::ToString::to_string),
            ),
            session_id,
        )
        .await
    }

    pub async fn run_with_workspace_stream(
        &self,
        message: &str,
        session_id: &str,
        working_directory: Option<&str>,
        events: mpsc::Sender<Value>,
    ) -> Result<Value, String> {
        self.run_task_stream(
            TaskSpec::legacy(
                message,
                working_directory.map(std::string::ToString::to_string),
            ),
            session_id,
            events,
        )
        .await
    }

    pub async fn run_task(&self, task_spec: TaskSpec, session_id: &str) -> Result<Value, String> {
        self.run_task_with_id(task_spec, session_id, None).await
    }

    pub async fn run_task_with_id(
        &self,
        task_spec: TaskSpec,
        session_id: &str,
        task_id: Option<&str>,
    ) -> Result<Value, String> {
        let session_id = SessionStore::normalize_session_id(session_id)?;
        let identity =
            RequestIdentity::resolve(SERVICE_API_VERSION, None, None, &session_id, || {
                legacy_request_id(task_id)
            })?;
        self.run_task_with_identity(task_spec, &identity, task_id)
            .await
    }

    pub async fn run_task_with_identity(
        &self,
        task_spec: TaskSpec,
        identity: &RequestIdentity,
        task_id: Option<&str>,
    ) -> Result<Value, String> {
        let record = self
            .task_ledger
            .create_with_identity(
                &identity.owner_id,
                &identity.request_id,
                &identity.session_id,
                task_spec,
                task_id,
            )
            .await?;
        if record.status != TaskStatus::Pending {
            return replay_task_record(&record);
        }
        self.execute_task_record(&record.task_id, None).await
    }

    pub async fn run_task_stream(
        &self,
        task_spec: TaskSpec,
        session_id: &str,
        events: mpsc::Sender<Value>,
    ) -> Result<Value, String> {
        self.run_task_stream_with_id(task_spec, session_id, None, events)
            .await
    }

    pub async fn run_task_stream_with_id(
        &self,
        task_spec: TaskSpec,
        session_id: &str,
        task_id: Option<&str>,
        events: mpsc::Sender<Value>,
    ) -> Result<Value, String> {
        let session_id = SessionStore::normalize_session_id(session_id)?;
        let identity =
            RequestIdentity::resolve(SERVICE_API_VERSION, None, None, &session_id, || {
                legacy_request_id(task_id)
            })?;
        self.run_task_stream_with_identity(task_spec, &identity, task_id, events)
            .await
    }

    pub async fn run_task_stream_with_identity(
        &self,
        task_spec: TaskSpec,
        identity: &RequestIdentity,
        task_id: Option<&str>,
        events: mpsc::Sender<Value>,
    ) -> Result<Value, String> {
        let record = self
            .task_ledger
            .create_with_identity(
                &identity.owner_id,
                &identity.request_id,
                &identity.session_id,
                task_spec,
                task_id,
            )
            .await?;
        let prepared_debug = if record.status == TaskStatus::Pending {
            Some(self.start_task_debug_trace(&record).await?)
        } else {
            None
        };
        let mut task_created =
            json!({"type":"task_created","task_id":record.task_id,"status":record.status});
        if let Some(debug) = prepared_debug.as_ref() {
            attach_debug_capture(
                &mut task_created,
                debug.trace_id.as_deref(),
                debug.capture.as_ref(),
            );
            debug_record(
                debug.handle.as_ref(),
                "stream",
                "runtime_stream",
                "task_created",
                None,
                || task_created.clone(),
            )
            .await;
        }
        events
            .send(task_created)
            .await
            .map_err(|_| "stream receiver closed before task creation event".to_string())?;
        if record.status != TaskStatus::Pending {
            let response = replay_task_record(&record)?;
            events
                .send(json!({"type":"final","response":response.clone()}))
                .await
                .map_err(|_| "stream receiver closed before final event".to_string())?;
            return Ok(response);
        }
        match self
            .execute_task_record_with_cancellation(
                &record.task_id,
                Some(&events),
                CancellationToken::default(),
                prepared_debug,
            )
            .await
        {
            Ok(response) => {
                events
                    .send(json!({"type":"final","response":response.clone()}))
                    .await
                    .map_err(|_| "stream receiver closed before final event".to_string())?;
                Ok(response)
            }
            Err(error) => {
                let _ = events.send(json!({"type":"error","error":error})).await;
                Err(error)
            }
        }
    }

    pub async fn resume_task(&self, task_id: &str) -> Result<Value, String> {
        self.task_ledger.prepare_resume(task_id).await?;
        self.execute_task_record(task_id, None).await
    }

    pub async fn task(&self, task_id: &str) -> Result<TaskRecord, String> {
        self.task_ledger.get(task_id).await
    }

    pub async fn task_for_owner(
        &self,
        task_id: &str,
        owner_id: &str,
    ) -> Result<TaskRecord, String> {
        self.task_ledger.get_for_owner(task_id, owner_id).await
    }

    pub async fn tasks(&self, limit: usize) -> Result<Vec<TaskRecord>, String> {
        self.task_ledger.list(limit).await
    }

    pub async fn tasks_for_owner(
        &self,
        owner_id: &str,
        limit: usize,
    ) -> Result<Vec<TaskRecord>, String> {
        self.task_ledger.list_for_owner(owner_id, limit).await
    }

    pub async fn cancel_task(&self, task_id: &str) -> Result<TaskRecord, String> {
        let record = self.task_ledger.cancel(task_id).await?;
        if let Some(cancellation) = self.active_tasks.lock().await.get(task_id).cloned() {
            cancellation.cancel();
        }
        Ok(record)
    }

    pub async fn resume_task_for_owner(
        &self,
        task_id: &str,
        owner_id: &str,
    ) -> Result<Value, String> {
        self.task_ledger
            .prepare_resume_for_owner(task_id, owner_id)
            .await?;
        self.execute_task_record(task_id, None).await
    }

    pub async fn cancel_task_for_owner(
        &self,
        task_id: &str,
        owner_id: &str,
    ) -> Result<TaskRecord, String> {
        let record = self.task_ledger.cancel_for_owner(task_id, owner_id).await?;
        if let Some(cancellation) = self.active_tasks.lock().await.get(task_id).cloned() {
            cancellation.cancel();
        }
        Ok(record)
    }

    async fn execute_task_record(
        &self,
        task_id: &str,
        events: Option<&mpsc::Sender<Value>>,
    ) -> Result<Value, String> {
        self.execute_task_record_with_cancellation(
            task_id,
            events,
            CancellationToken::default(),
            None,
        )
        .await
    }

    async fn execute_task_record_with_cancellation(
        &self,
        task_id: &str,
        events: Option<&mpsc::Sender<Value>>,
        cancellation: CancellationToken,
        prepared_debug: Option<DebugTraceStart>,
    ) -> Result<Value, String> {
        let debug_start = match prepared_debug {
            Some(value) => value,
            None => {
                let initial_record = self.task_ledger.get(task_id).await?;
                self.start_task_debug_trace(&initial_record).await?
            }
        };
        {
            let mut active = self.active_tasks.lock().await;
            if active
                .insert(task_id.to_string(), cancellation.clone())
                .is_some()
            {
                return Err(format!("task `{task_id}` is already running"));
            }
        }
        let result = self
            .execute_task_record_inner(task_id, events, &cancellation, debug_start.handle.as_ref())
            .await;
        self.active_tasks.lock().await.remove(task_id);
        self.task_completion.notify_waiters();
        let mut capture = debug_start.capture;
        if let Some(handle) = debug_start.handle.as_ref() {
            debug_record(
                Some(handle),
                "service",
                "service_api",
                "response_ready",
                None,
                || match &result {
                    Ok(value) => json!({"status":"ok","response":value}),
                    Err(error) => json!({"status":"error","error":error}),
                },
            )
            .await;
            if events.is_some() {
                let event_type = if result.is_ok() { "final" } else { "error" };
                debug_record(
                    Some(handle),
                    "stream",
                    "runtime_stream",
                    event_type,
                    None,
                    || match &result {
                        Ok(value) => json!({"type":"final","response":value}),
                        Err(error) => json!({"type":"error","error":error}),
                    },
                )
                .await;
            }
            let task_record = self
                .task_ledger
                .get(task_id)
                .await
                .ok()
                .and_then(|value| serde_json::to_value(value).ok());
            let final_response = match &result {
                Ok(value) => Some(value.clone()),
                Err(error) => Some(json!({"status":"error","error":error})),
            };
            let reason = match &result {
                Ok(value) if value.get("status").and_then(Value::as_str) == Some("cancelled") => {
                    "task_cancelled"
                }
                Ok(_) => "task_finished",
                Err(_) => "task_failed",
            };
            capture = Some(
                match handle
                    .finish(task_record, final_response, true, reason)
                    .await
                {
                    Ok(_) => DebugCapture {
                        mode: handle.mode(),
                        status: "complete".into(),
                        error: None,
                    },
                    Err(error) => DebugCapture {
                        mode: handle.mode(),
                        status: "incomplete".into(),
                        error: Some(error),
                    },
                },
            );
            self.task_ledger
                .set_debug_trace(
                    task_id,
                    debug_start.trace_id.clone(),
                    capture
                        .as_ref()
                        .and_then(|value| serde_json::to_value(value).ok()),
                )
                .await?;
        }
        match result {
            Ok(mut value) => {
                attach_debug_capture(
                    &mut value,
                    debug_start.trace_id.as_deref(),
                    capture.as_ref(),
                );
                Ok(value)
            }
            Err(error) => {
                if let Some(trace_id) = debug_start.trace_id {
                    let suffix = debug_error_markers(capture.as_ref());
                    Err(format!("{error}; debug_trace_id={trace_id}{suffix}"))
                } else {
                    Err(error)
                }
            }
        }
    }

    async fn start_task_debug_trace(&self, record: &TaskRecord) -> Result<DebugTraceStart, String> {
        let debug_start = self
            .debug_trace
            .start_lazy(
                &record.request_id,
                &record.owner_id,
                &record.session_id,
                Some(&record.task_id),
                || {
                    json!({
                        "task_spec":record.task_spec,
                        "task_status":record.status,
                        "task_revision":record.revision,
                    })
                },
            )
            .await;
        if debug_start.trace_id.is_some() || debug_start.capture.is_some() {
            self.task_ledger
                .set_debug_trace(
                    &record.task_id,
                    debug_start.trace_id.clone(),
                    debug_start
                        .capture
                        .as_ref()
                        .and_then(|value| serde_json::to_value(value).ok()),
                )
                .await?;
        }
        debug_record(
            debug_start.handle.as_ref(),
            "service",
            "service_api",
            "request_received",
            None,
            || {
                json!({
                    "api_version":SERVICE_API_VERSION,
                    "request_id":record.request_id,
                    "owner_id":record.owner_id,
                    "session_id":record.session_id,
                    "task_id":record.task_id,
                    "task_spec":record.task_spec,
                })
            },
        )
        .await;
        Ok(debug_start)
    }

    async fn execute_task_record_inner(
        &self,
        task_id: &str,
        events: Option<&mpsc::Sender<Value>>,
        cancellation: &CancellationToken,
        debug_trace: Option<&DebugTraceHandle>,
    ) -> Result<Value, String> {
        if cancellation.is_cancelled() {
            return self.cancelled_task_response(task_id).await;
        }
        let mut record = self.task_ledger.start_task(task_id).await?;
        debug_record(
            debug_trace,
            "service",
            "task_ledger",
            "task_started",
            None,
            || json!({"task_id":task_id,"revision":record.revision,"status":record.status}),
        )
        .await;
        let stage_count = record.stages.len();
        let mut final_response = None;
        for index in 0..stage_count {
            if cancellation.is_cancelled() {
                return self.cancelled_task_response(task_id).await;
            }
            let stage = record.stages[index].clone();
            if stage.status == StageStatus::Succeeded {
                final_response = stage.response.clone().or(final_response);
                continue;
            }
            let dependencies_ready = stage.spec.depends_on.iter().all(|dependency| {
                record.stages.iter().any(|candidate| {
                    candidate.spec.id == *dependency && candidate.status == StageStatus::Succeeded
                })
            });
            if !dependencies_ready {
                let error = format!(
                    "stage `{}` has an incomplete dependency checkpoint",
                    stage.spec.id
                );
                self.task_ledger
                    .fail_stage(task_id, &stage.spec.id, &error, None)
                    .await?;
                return Err(error);
            }
            record = self
                .task_ledger
                .start_stage(task_id, &stage.spec.id)
                .await?;
            debug_record(
                debug_trace,
                "service",
                "task_ledger",
                "stage_started",
                Some(&stage.spec.id),
                || json!({"stage_index":index,"stage_count":stage_count,"stage":stage.spec}),
            )
            .await;
            if let Some(events) = events {
                debug_record(
                    debug_trace,
                    "stream",
                    "runtime_stream",
                    "stage_started",
                    Some(&stage.spec.id),
                    || json!({"type":"stage_started","task_id":task_id,"stage_id":stage.spec.id,"stage_index":index,"stage_count":stage_count}),
                )
                .await;
                let _ = events
                    .send(json!({"type":"stage_started","task_id":task_id,"stage_id":stage.spec.id,"stage_index":index,"stage_count":stage_count}))
                    .await;
            }
            let stage_task = record
                .task_spec
                .task_for_stage(&stage.spec, index + 1 == stage_count)
                .map_err(|error| error.to_string())?;
            let result = self
                .run_task_inner(
                    stage_task,
                    &record.session_id,
                    events,
                    cancellation,
                    debug_trace,
                    Some(&stage.spec.id),
                )
                .await;
            if cancellation.is_cancelled() {
                return self.cancelled_task_response(task_id).await;
            }
            let response = match result {
                Ok(response) if response.get("status").and_then(Value::as_str) == Some("ok") => {
                    response
                }
                Ok(mut response) => {
                    let error = response
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("stage returned a non-ok status")
                        .to_string();
                    record = self
                        .task_ledger
                        .fail_stage(task_id, &stage.spec.id, &error, Some(response.clone()))
                        .await?;
                    attach_task_ledger(&mut response, &record);
                    return Ok(response);
                }
                Err(error) => {
                    self.task_ledger
                        .fail_stage(task_id, &stage.spec.id, &error, None)
                        .await?;
                    return Err(error);
                }
            };
            let latest = self.task_ledger.get(task_id).await?;
            if latest.status == TaskStatus::Cancelled {
                let mut cancelled = json!({
                    "status":"cancelled",
                    "error":"task cancelled at the next durable stage boundary",
                    "answer":"",
                });
                attach_task_ledger(&mut cancelled, &latest);
                return Ok(cancelled);
            }
            record = self
                .task_ledger
                .complete_stage(task_id, &stage.spec.id, response.clone())
                .await?;
            debug_record(
                debug_trace,
                "service",
                "task_ledger",
                "stage_completed",
                Some(&stage.spec.id),
                || json!({"status":record.status,"response":response}),
            )
            .await;
            if let Some(events) = events {
                debug_record(
                    debug_trace,
                    "stream",
                    "runtime_stream",
                    "stage_completed",
                    Some(&stage.spec.id),
                    || json!({"type":"stage_completed","task_id":task_id,"stage_id":stage.spec.id,"stage_index":index,"stage_count":stage_count}),
                )
                .await;
                let _ = events
                    .send(json!({"type":"stage_completed","task_id":task_id,"stage_id":stage.spec.id,"stage_index":index,"stage_count":stage_count}))
                    .await;
            }
            final_response = Some(response);
        }
        let response = final_response.ok_or_else(|| "task has no executable stage".to_string())?;
        record = self
            .task_ledger
            .complete_task(task_id, response.clone())
            .await?;
        debug_record(
            debug_trace,
            "service",
            "task_ledger",
            "task_completed",
            None,
            || json!({"status":record.status,"revision":record.revision}),
        )
        .await;
        let mut response = response;
        attach_task_ledger(&mut response, &record);
        Ok(response)
    }

    async fn cancelled_task_response(&self, task_id: &str) -> Result<Value, String> {
        let record = match self.task_ledger.get(task_id).await? {
            record if record.status == TaskStatus::Cancelled => record,
            _ => self.task_ledger.cancel(task_id).await?,
        };
        let mut response = json!({
            "status":"cancelled",
            "error_code":"cancelled",
            "error":"task cancelled at a durable controller boundary",
            "answer":"",
        });
        attach_task_ledger(&mut response, &record);
        Ok(response)
    }

    async fn run_task_inner(
        &self,
        task_spec: TaskSpec,
        session_id: &str,
        events: Option<&mpsc::Sender<Value>>,
        cancellation: &CancellationToken,
        debug_trace: Option<&DebugTraceHandle>,
        stage_id: Option<&str>,
    ) -> Result<Value, String> {
        if cancellation.is_cancelled() {
            return Err("task cancelled".into());
        }
        let task_spec = task_spec.normalize().map_err(|error| error.to_string())?;
        debug_record(
            debug_trace,
            "service",
            "runtime",
            "task_spec_normalized",
            stage_id,
            || json!({"task_spec":task_spec}),
        )
        .await;
        let message = task_spec.objective.as_str();
        let workspace = match task_spec.working_directory.as_deref() {
            Some(value) => {
                let value = value.trim();
                Some((value.to_string(), self.command.scoped(value).await?))
            }
            None => None,
        };
        let session_id = SessionStore::normalize_session_id(session_id)?;
        let _turn = self.sessions.lock(&session_id).await?;
        if workspace.is_none() && message.chars().count() >= self.config.long_text_capture_chars {
            self.invalidate_chat_state(&session_id).await?;
            let tool_result = self.data.capture_text(&session_id, message).await?;
            if cancellation.is_cancelled() {
                return Err("task cancelled".into());
            }
            let chars = message.chars().count();
            let answer = if contains_chinese(message) {
                format!("已接收长文本，共{chars}个字符。请继续提问。")
            } else {
                format!("Received {chars} characters of long text. Ask a question when ready.")
            };
            self.sessions
                .append(
                    &session_id,
                    &format!("[pasted long text: {chars} characters]"),
                    &answer,
                )
                .await?;
            return Ok(json!({
                "status":"ok","session_id":session_id,"route":{"mode":"document_capture","tool":null},
                "tool_result":tool_result,"answer":answer,
                "trace":{"task_spec":&task_spec,"model_called":false,"context":{"mode":"transient_session_text"},"control_plane":"rust"},
            }));
        }
        let started = Instant::now();
        if let Some(events) = events {
            debug_record(
                debug_trace,
                "stream",
                "runtime_stream",
                "phase",
                stage_id,
                || json!({"type":"phase","phase":"routing"}),
            )
            .await;
            let _ = events.send(json!({"type":"phase","phase":"routing"})).await;
        }
        let history = self.sessions.history(&session_id, 12).await?;
        let context = prompt::bounded_context(&history, 8000);
        let routing_context = prompt::bounded_context(&history, 2000);
        let text_status = self.data.text_status(&session_id).await?;
        if cancellation.is_cancelled() {
            return Err("task cancelled".into());
        }
        let has_text = text_status
            .get("active")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let gate = if workspace.is_some() {
            GateDecision {
                use_tool: true,
                tool: Some("run_command".into()),
                trace: serde_json::Map::from_iter([
                    ("mode".into(), json!("explicit_workspace")),
                    ("label".into(), json!("tool")),
                ]),
            }
        } else {
            self.sidecar
                .gate(
                    message,
                    if has_text {
                        self.config.pasted_text_gate_threshold
                    } else {
                        self.config.tool_gate_threshold
                    },
                    &routing_context,
                    has_text,
                )
                .await?
        };
        debug_record(
            debug_trace,
            "model",
            "routing",
            "gate_completed",
            stage_id,
            || json!({"message":message,"routing_context":routing_context,"decision":gate}),
        )
        .await;
        if cancellation.is_cancelled() {
            return Err("task cancelled".into());
        }
        if !gate.use_tool {
            return self
                .direct_chat(
                    &task_spec,
                    message,
                    &session_id,
                    history,
                    context,
                    gate,
                    started,
                    events,
                    cancellation,
                    debug_trace,
                    stage_id,
                )
                .await;
        }
        if let Some(events) = events {
            debug_record(
                debug_trace,
                "stream",
                "runtime_stream",
                "phase",
                stage_id,
                || json!({"type":"phase","phase":"tool"}),
            )
            .await;
            let _ = events.send(json!({"type":"phase","phase":"tool"})).await;
        }
        self.invalidate_chat_state(&session_id).await?;
        let (workspace_label, command) = match workspace {
            Some((label, command)) => (Some(label), command),
            None => (None, self.command.clone()),
        };
        self.tool_run(
            message,
            &session_id,
            ToolRunOptions {
                task_spec: &task_spec,
                context: &context,
                has_text,
                gate,
                started,
                command,
                workspace_label: workspace_label.as_deref(),
                cancellation: cancellation.clone(),
                debug_trace,
                stage_id,
            },
        )
        .await
    }

    pub async fn research(
        &self,
        message: &str,
        session_id: &str,
        branch_width: usize,
        max_rounds: usize,
    ) -> Result<Value, String> {
        self.research_inner(message, session_id, branch_width, max_rounds, None)
            .await
    }

    async fn research_inner(
        &self,
        message: &str,
        session_id: &str,
        branch_width: usize,
        max_rounds: usize,
        debug_trace: Option<&DebugTraceHandle>,
    ) -> Result<Value, String> {
        let message = message.trim();
        if message.is_empty() {
            return Err("message must not be empty".into());
        }
        let session_id = SessionStore::normalize_session_id(session_id)?;
        let _turn = self.sessions.lock(&session_id).await?;
        self.invalidate_chat_state(&session_id).await?;
        let runner = ResearchRunner::new(self.sidecar.clone(), self.data.clone());
        let result = runner
            .run(
                message,
                &session_id,
                branch_width,
                max_rounds,
                CancellationToken::default(),
                self.config.max_run_elapsed,
                debug_trace,
            )
            .await?;
        self.sessions
            .append(
                &session_id,
                message,
                result
                    .get("answer")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
            .await?;
        Ok(result)
    }

    pub async fn research_with_identity(
        &self,
        message: &str,
        identity: &RequestIdentity,
        branch_width: usize,
        max_rounds: usize,
    ) -> Result<Value, String> {
        let start = self
            .debug_trace
            .start_lazy(
                &identity.request_id,
                &identity.owner_id,
                &identity.session_id,
                None,
                || json!({"endpoint":"/v1/research","message":message,"branch_width":branch_width,"max_rounds":max_rounds}),
            )
            .await;
        debug_record(
            start.handle.as_ref(),
            "model",
            "research",
            "research_requested",
            None,
            || json!({"message":message,"branch_width":branch_width,"max_rounds":max_rounds}),
        )
        .await;
        let result = self
            .research_inner(
                message,
                &identity.session_id,
                branch_width,
                max_rounds,
                start.handle.as_ref(),
            )
            .await;
        debug_record(
            start.handle.as_ref(),
            "model",
            "research",
            "research_completed",
            None,
            || match &result {
                Ok(value) => json!({"status":"ok","response":value}),
                Err(error) => json!({"status":"error","error":error}),
            },
        )
        .await;
        self.finish_standalone_trace(start, result, "research_finished")
            .await
    }

    async fn finish_standalone_trace(
        &self,
        start: DebugTraceStart,
        result: Result<Value, String>,
        reason: &str,
    ) -> Result<Value, String> {
        let mut capture = start.capture;
        if let Some(handle) = start.handle.as_ref() {
            capture = Some(
                match handle
                    .finish(
                        None,
                        Some(match &result {
                            Ok(value) => value.clone(),
                            Err(error) => json!({"status":"error","error":error}),
                        }),
                        true,
                        reason,
                    )
                    .await
                {
                    Ok(_) => DebugCapture {
                        mode: handle.mode(),
                        status: "complete".into(),
                        error: None,
                    },
                    Err(error) => DebugCapture {
                        mode: handle.mode(),
                        status: "incomplete".into(),
                        error: Some(error),
                    },
                },
            );
        }
        match result {
            Ok(mut value) => {
                attach_debug_capture(&mut value, start.trace_id.as_deref(), capture.as_ref());
                Ok(value)
            }
            Err(error) => {
                if let Some(trace_id) = start.trace_id {
                    let suffix = debug_error_markers(capture.as_ref());
                    Err(format!("{error}; debug_trace_id={trace_id}{suffix}"))
                } else {
                    Err(error)
                }
            }
        }
    }

    pub async fn shutdown(&self) -> Result<(), String> {
        for cancellation in self.active_tasks.lock().await.values() {
            cancellation.cancel();
        }
        let deadline = Instant::now()
            .checked_add(self.config.shutdown_grace)
            .unwrap_or_else(Instant::now);
        let mut failures = Vec::new();
        loop {
            let notified = self.task_completion.notified();
            let active = self
                .active_tasks
                .lock()
                .await
                .keys()
                .cloned()
                .collect::<Vec<_>>();
            if active.is_empty() {
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() || tokio::time::timeout(remaining, notified).await.is_err() {
                failures.push(format!(
                    "shutdown deadline exceeded while tasks remained active: {}",
                    active.join(",")
                ));
                break;
            }
        }
        let records = {
            let mut cache = self.chat_states.lock().await;
            cache
                .values
                .drain()
                .map(|(_, value)| value)
                .collect::<Vec<_>>()
        };
        for record in &records {
            if let Some(error) = &record.blocked_error {
                failures.push(format!(
                    "Session {} has an unresolved State lifecycle: {error}",
                    record.session_id
                ));
            }
            if let Err(error) = self.release_chat_record(record).await {
                failures.push(error);
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn direct_chat(
        &self,
        task_spec: &TaskSpec,
        message: &str,
        session_id: &str,
        history: Vec<Exchange>,
        context: String,
        gate: GateDecision,
        started: Instant,
        events: Option<&mpsc::Sender<Value>>,
        cancellation: &CancellationToken,
        debug_trace: Option<&DebugTraceHandle>,
        stage_id: Option<&str>,
    ) -> Result<Value, String> {
        let history_len = history.len();
        let (cached, stale) = self.chat_states.lock().await.take(session_id, history_len);
        if let Some(stale) = &stale {
            self.release_chat_record(stale).await?;
        }
        let reused = cached.is_some();
        // Durable State owner identity must remain stable across turns. The
        // generic owner_id() intentionally adds a nonce for short-lived task
        // States and therefore cannot identify a persistent chat Session.
        let owner_id = chat_owner_id(session_id);
        let prefix = prompt::direct_prefix(&context);
        let estimated_input_tokens = ((prefix.chars().count() + 3) / 4)
            .try_into()
            .unwrap_or(u64::MAX);
        let mut placement_trace = json!({
            "contract_version":"statepool-execution-plan.v1",
            "mode":"state_affinity",
            "reason_code":"state_affinity",
        });

        let mut state = if let Some(mut cached) = cached {
            if let Some(error) = cached.blocked_error.clone() {
                let error = format!(
                    "Session State requires reconciliation before another execution: {error}"
                );
                return Err(self.preserve_chat_record(cached, &error).await);
            }
            if cached.hot_state.is_none()
                && let Some(lease) = cached.active_lease.take()
                && let Err(error) = self.cloud_plugin.release_lease(&lease).await
            {
                cached.active_lease = Some(lease);
                let error = format!("previous committed State Lease is not released: {error}");
                return Err(self.preserve_chat_record(cached, &error).await);
            }

            if let Some(hot) = cached.hot_state.clone() {
                placement_trace["worker_endpoint"] = json!(hot.home_url);
                placement_trace["state_version"] = json!(cached.state_version);
                if self.cloud_plugin.state_lifecycle_ready().await {
                    let holder = format!("rwkv-agent/hot/{session_id}/{}", cached.state_version);
                    match self
                        .cloud_plugin
                        .acquire_lease(session_id, &owner_id, &holder, cached.state_version)
                        .await
                    {
                        Ok(lease) => cached.active_lease = Some(lease),
                        Err(error)
                            if self.config.cloud_plugin.fallback == CloudPluginFallback::Local
                                && cached.state_ref.is_none() =>
                        {
                            placement_trace["lifecycle_status"] = json!("degraded_hot");
                            placement_trace["lifecycle_error"] = json!(error);
                        }
                        Err(error) => {
                            return Err(self.preserve_chat_record(cached, &error).await);
                        }
                    }
                }
                debug_record(
                    debug_trace,
                    "state",
                    "state_cache",
                    "state_reused",
                    stage_id,
                    || json!({"state_id":hot.state_id,"seen_tokens":hot.seen_tokens,"residency":"hot"}),
                )
                .await;
                cached
            } else if let Some(state_ref) = cached.state_ref.clone() {
                let plan = match self
                    .cloud_plugin
                    .plan_with_state(
                        session_id,
                        &owner_id,
                        estimated_input_tokens,
                        self.config.direct_chat_max_tokens.into(),
                        Some(state_ref.clone()),
                    )
                    .await
                {
                    Ok(plan) => plan,
                    Err(error) => return Err(self.preserve_chat_record(cached, &error).await),
                };
                placement_trace = serde_json::to_value(&plan)
                    .unwrap_or_else(|_| json!({"status":"serialization_error"}));
                let endpoint = match plan.mode.as_str() {
                    "remote" => match plan.remote_endpoint() {
                        Some(endpoint) => endpoint,
                        None => {
                            return Err(self
                                .preserve_chat_record(
                                    cached,
                                    "cloud plugin remote restore plan has no endpoint",
                                )
                                .await);
                        }
                    },
                    "defer" => {
                        return Err(self
                            .preserve_chat_record(
                                cached,
                                "cloud plugin deferred committed State restore",
                            )
                            .await);
                    }
                    "reject" => {
                        return Err(self
                            .preserve_chat_record(
                                cached,
                                "cloud plugin rejected committed State restore",
                            )
                            .await);
                    }
                    mode => {
                        return Err(self
                            .preserve_chat_record(
                                cached,
                                &format!(
                                    "committed State restore requires a remote Worker, got {mode}"
                                ),
                            )
                            .await);
                    }
                };
                let worker_id = match plan.worker_id.as_deref() {
                    Some(worker_id) => worker_id,
                    None => {
                        return Err(self
                            .preserve_chat_record(
                                cached,
                                "cloud plugin restore plan has no Worker identity",
                            )
                            .await);
                    }
                };
                let holder = format!("rwkv-agent/{}/restore", plan.decision_id);
                let lease = match self
                    .cloud_plugin
                    .acquire_lease(session_id, &owner_id, &holder, state_ref.version)
                    .await
                {
                    Ok(lease) => lease,
                    Err(error) => return Err(self.preserve_chat_record(cached, &error).await),
                };
                let restored = match self
                    .cloud_plugin
                    .read_state(state_ref.clone(), worker_id, &lease)
                    .await
                {
                    Ok(restored) => restored,
                    Err(error) => {
                        let release = self.cloud_plugin.release_lease(&lease).await;
                        let error = append_cleanup_error(error, release.err());
                        return Err(self.preserve_chat_record(cached, &error).await);
                    }
                };
                let model_ref = match self.cloud_plugin.state_model_ref() {
                    Ok(model_ref) => model_ref,
                    Err(error) => {
                        let release = self.cloud_plugin.release_lease(&lease).await;
                        let error = append_cleanup_error(error, release.err());
                        return Err(self.preserve_chat_record(cached, &error).await);
                    }
                };
                let hot = match self
                    .sidecar
                    .restore_at(
                        endpoint,
                        &owner_id,
                        &model_ref,
                        &state_ref.checksum,
                        &restored.payload_base64,
                    )
                    .await
                {
                    Ok(state) => state,
                    Err(error) => {
                        let release = self.cloud_plugin.release_lease(&lease).await;
                        let error = append_cleanup_error(error, release.err());
                        return Err(self.preserve_chat_record(cached, &error).await);
                    }
                };
                placement_trace["lifecycle_status"] = json!("restored");
                placement_trace["state_version"] = json!(state_ref.version);
                debug_record(
                    debug_trace,
                    "state",
                    "statepool_cloud_plugin",
                    "state_restored",
                    stage_id,
                    || json!({"state_id":state_ref.state_id,"worker_state_id":hot.state_id,"version":state_ref.version,"checksum":state_ref.checksum}),
                )
                .await;
                cached.hot_state = Some(hot);
                cached.active_lease = Some(lease);
                cached
            } else {
                return Err("chat State cache invariant violated: no hot or durable State".into());
            }
        } else {
            // Planning receives bounded metadata only. Prompt text and raw
            // recurrent-State bytes are never sent to the placement endpoint.
            let plan = self
                .cloud_plugin
                .plan(
                    session_id,
                    &owner_id,
                    estimated_input_tokens,
                    self.config.direct_chat_max_tokens.into(),
                )
                .await?;
            placement_trace = serde_json::to_value(&plan)
                .unwrap_or_else(|_| json!({"status":"serialization_error"}));
            debug_record(
                debug_trace,
                "routing",
                "statepool_cloud_plugin",
                "execution_planned",
                stage_id,
                || placement_trace.clone(),
            )
            .await;

            let mut selected_endpoint = match plan.mode.as_str() {
                "local" => None,
                "remote" => Some(
                    plan.remote_endpoint()
                        .ok_or_else(|| "cloud plugin remote plan has no endpoint".to_string())?
                        .to_string(),
                ),
                "defer" => return Err("cloud plugin deferred execution".into()),
                "reject" => return Err("cloud plugin rejected execution".into()),
                mode => return Err(format!("cloud plugin returned unknown mode: {mode}")),
            };
            let mut active_lease = None;
            if self.cloud_plugin.state_lifecycle_ready().await {
                let holder = format!("rwkv-agent/{}/open", plan.decision_id);
                match self
                    .cloud_plugin
                    .acquire_lease(session_id, &owner_id, &holder, 0)
                    .await
                {
                    Ok(lease) => active_lease = Some(lease),
                    Err(error)
                        if self.config.cloud_plugin.fallback == CloudPluginFallback::Local =>
                    {
                        selected_endpoint = None;
                        placement_trace["lifecycle_status"] = json!("fallback_before_execution");
                        placement_trace["lifecycle_error"] = json!(error);
                    }
                    Err(error) => return Err(error),
                }
            }
            debug_record(
                debug_trace,
                "model",
                "provider",
                "prefill_requested",
                stage_id,
                || json!({"owner_id":owner_id,"prompt":prefix}),
            )
            .await;
            let hot = match selected_endpoint {
                Some(endpoint) => self.sidecar.prefill_at(&endpoint, &owner_id, &prefix).await,
                None => self.sidecar.prefill(&owner_id, &prefix).await,
            };
            let hot = match hot {
                Ok(state) => state,
                Err(error) => {
                    let release = if let Some(lease) = &active_lease {
                        self.cloud_plugin.release_lease(lease).await.err()
                    } else {
                        None
                    };
                    return Err(append_cleanup_error(error, release));
                }
            };
            debug_record(
                debug_trace,
                "state",
                "provider",
                "state_opened",
                stage_id,
                || json!({"state":hot}),
            )
            .await;
            CachedChatState {
                session_id: session_id.to_string(),
                hot_state: Some(hot),
                state_ref: None,
                state_version: 0,
                active_lease,
                blocked_error: None,
                history_len,
                stop_reason: String::new(),
                last_used: 0,
            }
        };

        if cancellation.is_cancelled() {
            return Err(self
                .release_chat_after_error(&state, "task cancelled", debug_trace, stage_id)
                .await);
        }
        let input = prompt::direct_turn(message, reused, &state.stop_reason);
        let stops = prompt::CHAT_STOPS
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>();
        if let Some(events) = events {
            debug_record(
                debug_trace,
                "stream",
                "runtime_stream",
                "phase",
                stage_id,
                || json!({"type":"phase","phase":"decoding"}),
            )
            .await;
            let _ = events
                .send(json!({"type":"phase","phase":"decoding"}))
                .await;
        }
        let hot = state.hot_state()?.clone();
        debug_record(
            debug_trace,
            "model",
            "provider",
            "continue_requested",
            stage_id,
            || json!({"state_id":hot.state_id,"input":input,"stops":stops,"max_tokens":self.config.direct_chat_max_tokens}),
        )
        .await;
        let continuation = tokio::time::timeout(self.config.max_run_elapsed, async {
            match events {
                Some(events) if debug_trace.is_some() => {
                    self.sidecar
                        .continue_one_stream_captured(
                            &hot,
                            &input,
                            &stops,
                            self.config.direct_chat_max_tokens,
                            events.clone(),
                        )
                        .await
                }
                Some(events) => self
                    .sidecar
                    .continue_one_stream(
                        &hot,
                        &input,
                        &stops,
                        self.config.direct_chat_max_tokens,
                        events.clone(),
                    )
                    .await
                    .map(|row| (row, Vec::new())),
                None => self
                    .sidecar
                    .continue_one(&hot, &input, &stops, self.config.direct_chat_max_tokens)
                    .await
                    .map(|row| (row, Vec::new())),
            }
        })
        .await
        .unwrap_or_else(|_| {
            Err("run deadline exceeded during direct State continuation".to_string())
        });
        let (row, captured_stream_events) = match continuation {
            Ok(row) => row,
            Err(error) => {
                return Err(self
                    .release_chat_after_error(&state, &error, debug_trace, stage_id)
                    .await);
            }
        };
        for event in captured_stream_events {
            let event_type = event
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("provider_stream_event")
                .to_string();
            debug_record(
                debug_trace,
                "stream",
                "provider_stream",
                &event_type,
                stage_id,
                || event,
            )
            .await;
        }
        debug_record(
            debug_trace,
            "model",
            "provider",
            "continue_completed",
            stage_id,
            || json!({"state_id":row.state_id,"raw_output":row.text,"stop_reason":row.stop_reason,"token_ids":row.token_ids,"seen_tokens":row.seen_tokens,"elapsed_ms":row.elapsed_ms}),
        )
        .await;
        if cancellation.is_cancelled() {
            return Err(self
                .release_chat_after_error(&state, "task cancelled", debug_trace, stage_id)
                .await);
        }
        if row.state_id != hot.state_id {
            return Err(self
                .release_chat_after_error(
                    &state,
                    "direct chat continuation changed state_id",
                    debug_trace,
                    stage_id,
                )
                .await);
        }
        let (mut answer, reasoning_stripped) = strip_leading_think(&row.text);
        let generated_usable = !answer.is_empty();
        if answer.is_empty() {
            answer = generation_failure(message);
        }
        if let Err(error) = self.sessions.append(session_id, message, &answer).await {
            return Err(self
                .release_chat_after_error(&state, &error, debug_trace, stage_id)
                .await);
        }
        state.history_len = history_len + 1;
        state.stop_reason = row.stop_reason.clone();
        if let Some(hot) = &mut state.hot_state {
            hot.seen_tokens = row.seen_tokens;
        }
        // The recurrent State is the authoritative transcript. A complete
        // leading <think> block can be hidden from the user without discarding
        // the already-computed State; only an unusable generation or a System
        // boundary makes continuation unsafe.
        let safe_to_cache = chat_state_cache_safe(generated_usable, &row.stop_reason);
        let mut persistence_status = "released";
        let mut persistence_error = String::new();
        let mut released = Vec::new();

        if safe_to_cache {
            persistence_status = "hot";
            if let Some(mut lease) = state.active_lease.clone() {
                let mut lifecycle_error = None;
                match self.cloud_plugin.renew_lease(&lease).await {
                    Ok(renewed) => {
                        lease = renewed;
                        state.active_lease = Some(lease.clone());
                    }
                    Err(error) => lifecycle_error = Some(format!("Lease renewal failed: {error}")),
                }
                let mut snapshot = None;
                if lifecycle_error.is_none() {
                    match (self.cloud_plugin.state_model_ref(), state.hot_state()) {
                        (Ok(model_ref), Ok(hot)) => {
                            match self.sidecar.snapshot(hot, &model_ref).await {
                                Ok(value) => snapshot = Some(value),
                                Err(error) => {
                                    lifecycle_error =
                                        Some(format!("State snapshot failed: {error}"))
                                }
                            }
                        }
                        (Err(error), _) | (_, Err(error)) => lifecycle_error = Some(error),
                    }
                }
                let mut committed = None;
                if let Some(snapshot) = snapshot {
                    match self
                        .cloud_plugin
                        .commit_snapshot(&lease, snapshot.payload_base64, snapshot.checksum)
                        .await
                    {
                        Ok(state_ref) => committed = Some(state_ref),
                        Err(error) => {
                            lifecycle_error = Some(format!("State commit failed: {error}"))
                        }
                    }
                }
                if let Some(state_ref) = committed {
                    state.state_version = state_ref.version;
                    state.state_ref = Some(state_ref.clone());
                    let hot = state.hot_state()?.clone();
                    match self
                        .sidecar
                        .release_many(
                            &hot.home_url,
                            &hot.owner_id,
                            std::slice::from_ref(&hot.state_id),
                        )
                        .await
                    {
                        Ok(_) => {
                            state.hot_state = None;
                            state.active_lease = None;
                            persistence_status = "durable";
                            if let Err(error) = self.cloud_plugin.release_lease(&lease).await {
                                state.active_lease = Some(lease.clone());
                                persistence_error = format!("Lease release pending: {error}");
                                persistence_status = "durable_lease_pending";
                            }
                            debug_record(
                                debug_trace,
                                "state",
                                "statepool_cloud_plugin",
                                "state_committed",
                                stage_id,
                                || json!({"state_id":state_ref.state_id,"version":state_ref.version,"placement":state_ref.placement,"checksum":state_ref.checksum}),
                            )
                            .await;
                        }
                        Err(error) => {
                            lifecycle_error = Some(format!(
                                "State committed but source Worker release failed: {error}"
                            ));
                        }
                    }
                }
                if let Some(error) = lifecycle_error {
                    persistence_status = "blocked_hot";
                    persistence_error = error.clone();
                    state.blocked_error = Some(error);
                    state.active_lease = Some(lease);
                }
            }
            released = self
                .chat_states
                .lock()
                .await
                .put(state.clone(), self.config.chat_state_capacity);
            let state_id = state.state_identity();
            debug_record(
                debug_trace,
                "state",
                "state_cache",
                "state_cached",
                stage_id,
                || json!({"state_id":state_id,"seen_tokens":row.seen_tokens,"residency":persistence_status,"success":true,"error":persistence_error}),
            )
            .await;
        } else {
            self.release_chat_record(&state).await?;
            let state_id = state.state_identity();
            debug_record(
                debug_trace,
                "state",
                "state_cache",
                "state_released",
                stage_id,
                || json!({"state_id":state_id,"success":true,"reason":"unsafe_cache_boundary"}),
            )
            .await;
        }
        for evicted in &released {
            self.release_chat_record(evicted).await?;
            let state_id = evicted.state_identity();
            debug_record(
                debug_trace,
                "state",
                "state_cache",
                "state_released",
                stage_id,
                || json!({"state_id":state_id,"success":true,"reason":"cache_eviction"}),
            )
            .await;
        }
        Ok(json!({
            "status":"ok","session_id":session_id,"message":message,
            "route":{"mode":"direct","tool":null},"tool_result":null,"answer":answer,
            "trace":{
                "task_spec":task_spec,"gate":gate,"context":{"history_messages":history_len,"mode":"recurrent_session_state","session_state":{
                    "used":true,"reused":reused,"cached":safe_to_cache,"residency":persistence_status,
                    "state_version":state.state_version,"persistence_error":persistence_error,
                    "cache_reject_reason":if !generated_usable{"empty_or_incomplete_generation"}else if !safe_to_cache{"unsafe_stop_boundary"}else{""},"seen_tokens":row.seen_tokens}},
                "answer_completion":{"stop":row.stop_reason,"output_tokens":row.token_ids.len(),"model_elapsed_ms":row.elapsed_ms,"reasoning_stripped":reasoning_stripped},
                "placement":placement_trace,
                "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"control_plane":"rust",
            }
        }))
    }

    async fn tool_run(
        &self,
        message: &str,
        session_id: &str,
        options: ToolRunOptions<'_>,
    ) -> Result<Value, String> {
        let ToolRunOptions {
            task_spec,
            context,
            has_text,
            gate,
            started,
            command,
            workspace_label,
            cancellation,
            debug_trace,
            stage_id,
        } = options;
        let command_only = workspace_label.is_some();
        let workspace_inventory = if command_only {
            command.workspace_inventory().await?
        } else {
            Vec::new()
        };
        let (root_prompt, initial_input) = match workspace_label {
            Some(label) => prompt::workspace_agent_state_prompts_for_task(
                task_spec,
                context,
                label,
                self.config.max_tool_steps,
                &workspace_inventory,
            ),
            None => (
                prompt::agent_prompt(
                    &prompt::task_spec_message(task_spec),
                    has_text,
                    context,
                    self.config.command.enabled,
                ),
                format!("\n\nAssistant: {TOOL_CALL_JSON_PREFIX}"),
            ),
        };
        debug_record(
            debug_trace,
            "model",
            "provider",
            "agent_prefill_requested",
            stage_id,
            || json!({"root_prompt":root_prompt,"initial_input":initial_input,"workspace_inventory":workspace_inventory}),
        )
        .await;
        let mut model = self.sidecar.clone();
        let mut tools = RuntimeTools {
            data: self.data.for_turn(session_id, message),
            command,
            workspace: workspace_label.map(|label| {
                WorkspaceExecution::from_task_spec_with_inventory(
                    task_spec,
                    label,
                    &workspace_inventory,
                )
            }),
        };
        let registry = self.registry(command_only)?;
        let mut events = VecEventSink::default();
        let limits = RunLimits {
            max_tool_steps: self.config.max_tool_steps,
            max_protocol_retries: 2,
            max_answer_retries: 3,
            observation_reminder: String::new(),
            max_tokens_per_turn: self.config.max_model_tokens_per_turn,
            max_elapsed: self.config.max_run_elapsed,
            // Ordinary knowledge/Web/long-text tools commit after one
            // observation. Only an explicitly scoped workspace task enters
            // the multi-step command state machine.
            answer_after_tool: !command_only,
            fork_from_root: command_only,
            capture_model_output: debug_trace.is_some(),
        };
        let run_result = {
            let mut agent = AgentLoop::new(
                &mut model,
                &mut tools,
                &registry,
                &mut events,
                limits,
                cancellation,
            )
            .map_err(|e| e.to_string())?;
            agent
                .run(AgentRunRequest {
                    owner_id: owner_id("turn", session_id),
                    root_prompt,
                    initial_input,
                })
                .await
        };
        record_agent_events(debug_trace, stage_id, &events.events).await;
        let report = match run_result {
            Ok(report) => report,
            Err(error) => {
                let tool_steps = events
                    .events
                    .iter()
                    .filter_map(|event| match event {
                        AgentEvent::ToolCompleted {
                            step,
                            name,
                            arguments,
                            result,
                            ..
                        } => Some(
                            json!({"step":step,"name":name,"arguments":arguments,"result":result}),
                        ),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let last_tool = tool_steps.last();
                return Ok(json!({
                    "status":"error","error_code":error.code(),"error":error.to_string(),
                    "session_id":session_id,"message":message,"answer":"",
                    "route":{"mode":"tool_loop","tool":last_tool.and_then(|step|step.get("name")),
                        "strict":error.code() != "protocol_error","steps":tool_steps.len()},
                    "tool_result":last_tool.and_then(|step|step.get("result")),
                    "trace":{"task_spec":task_spec,"gate":gate,"agent":{"model_turns":events.events.iter().filter(|event|matches!(event, rwkv_agent_core::AgentEvent::ModelCompleted { .. })).count(),
                        "tool_steps":tool_steps,"events":events.events},"answer_protocol":Value::Null,
                        "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"control_plane":"rust"}
                }));
            }
        };
        let mut answer = report.answer.trim().to_string();
        let mut status = "ok";
        let evidence = report
            .tools
            .iter()
            .filter(|step| step.call.name == "web_search")
            .flat_map(|step| {
                step.result
                    .get("evidence")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .cloned()
            })
            .collect::<Vec<_>>();
        let answer_protocol = if evidence.is_empty() {
            Value::Null
        } else {
            let validation = self
                .data
                .validate_answer(message, &answer, &evidence)
                .await?;
            if validation
                .get("valid")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                answer = validation
                    .get("answer")
                    .and_then(Value::as_str)
                    .unwrap_or(&answer)
                    .to_string();
            } else {
                status = "insufficient_evidence";
                answer = insufficient(message);
            }
            validation
        };
        self.sessions.append(session_id, message, &answer).await?;
        let tool_steps = report
            .tools
            .iter()
            .map(|step| json!({"step":step.step,"name":step.call.name,"arguments":step.call.arguments,"result":step.result}))
            .collect::<Vec<_>>();
        let last_tool = report.tools.last();
        Ok(json!({
            "status":status,"session_id":session_id,"message":message,"answer":answer,
            "route":{"mode":"tool_loop","tool":last_tool.map(|step|step.call.name.as_str()),"strict":true,"steps":report.tool_steps},
            "tool_result":last_tool.map(|step|step.result.clone()),
            "trace":{"task_spec":task_spec,"gate":gate,"agent":{"model_turns":report.model_turns,"tool_steps":tool_steps,"events":events.events},"answer_protocol":answer_protocol,
                "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"control_plane":"rust"}
        }))
    }

    fn registry(&self, command_only: bool) -> Result<ToolRegistry, String> {
        let mut registry = ToolRegistry::default();
        if !command_only {
            for (name, description, argument) in [
                (
                    "web_search",
                    "Search live public Internet information",
                    "query",
                ),
                (
                    "knowledge_search",
                    "Search the local knowledge index",
                    "query",
                ),
                (
                    "long_text_qa",
                    "Answer from active pasted session text",
                    "question",
                ),
            ] {
                registry
                    .register(ToolDefinition::one_string(name, description, argument))
                    .map_err(|e| e.to_string())?;
            }
        }
        if self.config.command.enabled {
            registry
                .register(ToolDefinition {
                    name: "run_command".into(),
                    description: "Run one command in the isolated workspace".into(),
                    arguments: vec![
                        ArgumentSpec::required_string("command"),
                        // Tool-trained models commonly emit a short human
                        // label for command cards. It has no execution effect,
                        // remains type-checked, and all other extra keys stay
                        // rejected.
                        ArgumentSpec::optional_string("description"),
                    ],
                    allow_extra_arguments: false,
                })
                .map_err(|e| e.to_string())?;
            if command_only {
                for definition in workspace_file_tool_definitions() {
                    registry
                        .register(definition)
                        .map_err(|error| error.to_string())?;
                }
            }
        }
        Ok(registry)
    }

    fn tool_names(&self) -> Vec<&'static str> {
        let mut names = vec!["web_search", "knowledge_search", "long_text_qa"];
        if self.config.command.enabled {
            names.push("run_command");
            names.extend(["read_file", "write_file", "edit_file"]);
        }
        names
    }

    async fn invalidate_chat_state(&self, session_id: &str) -> Result<(), String> {
        if let Some(record) = self.chat_states.lock().await.pop(session_id) {
            if let Some(error) = &record.blocked_error {
                let message =
                    format!("Session State requires reconciliation before invalidation: {error}");
                return Err(self.preserve_chat_record(record, &message).await);
            }
            self.release_chat_record(&record).await?;
        }
        Ok(())
    }

    async fn release_chat_record(&self, record: &CachedChatState) -> Result<(), String> {
        let mut failures = Vec::new();
        if let Some(state) = &record.hot_state
            && let Err(error) = self
                .sidecar
                .release_many(
                    &state.home_url,
                    &state.owner_id,
                    std::slice::from_ref(&state.state_id),
                )
                .await
        {
            failures.push(format!("State release failed: {error}"));
        }
        if let Some(lease) = &record.active_lease
            && let Err(error) = self.cloud_plugin.release_lease(lease).await
        {
            failures.push(format!("Lease release failed: {error}"));
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }

    async fn preserve_chat_record(&self, record: CachedChatState, error: &str) -> String {
        let evicted = self
            .chat_states
            .lock()
            .await
            .put(record, self.config.chat_state_capacity);
        let mut failures = Vec::new();
        for record in &evicted {
            if let Err(release) = self.release_chat_record(record).await {
                failures.push(release);
            }
        }
        if failures.is_empty() {
            error.to_string()
        } else {
            format!(
                "{error}; cache eviction cleanup failed: {}",
                failures.join("; ")
            )
        }
    }

    async fn release_chat_after_error(
        &self,
        record: &CachedChatState,
        error: &str,
        debug_trace: Option<&DebugTraceHandle>,
        stage_id: Option<&str>,
    ) -> String {
        if record.active_lease.is_some() || record.state_ref.is_some() {
            let mut blocked = record.clone();
            blocked.blocked_error = Some(error.to_string());
            let preserved = self.preserve_chat_record(blocked, error).await;
            let state_id = record.state_identity();
            debug_record(
                debug_trace,
                "state",
                "statepool_cloud_plugin",
                "state_reconciliation_required",
                stage_id,
                || json!({"state_id":state_id,"success":false,"reason":error}),
            )
            .await;
            return format!("{preserved}; automatic re-execution is blocked");
        }
        let release = self.release_chat_record(record).await;
        let state_id = record.state_identity();
        debug_record(
            debug_trace,
            "state",
            "provider",
            "state_released",
            stage_id,
            || match &release {
                Ok(()) => json!({"state_id":state_id,"success":true,"reason":error}),
                Err(release) => {
                    json!({"state_id":state_id,"success":false,"reason":error,"error":release})
                }
            },
        )
        .await;
        match release {
            Ok(()) => error.to_string(),
            Err(release) => format!("{error}; State release also failed: {release}"),
        }
    }
}

fn append_cleanup_error(error: String, cleanup: Option<String>) -> String {
    match cleanup {
        Some(cleanup) => format!("{error}; cleanup also failed: {cleanup}"),
        None => error,
    }
}

fn state_capacity_status(model: &[Value], model_ready: bool) -> (Value, bool) {
    if !model_ready {
        return (
            json!({"status":"unavailable","providers":[],"error":"model Sidecar unavailable"}),
            false,
        );
    }
    let providers = model
        .iter()
        .map(|health| {
            let capacity = health
                .pointer("/persistent_states/capacity")
                .or_else(|| health.pointer("/persistent_state/capacity"))
                .and_then(Value::as_u64);
            let allocated = health
                .pointer("/persistent_states/allocated")
                .or_else(|| health.pointer("/persistent_state/allocated"))
                .and_then(Value::as_u64);
            let free = health
                .pointer("/persistent_states/free")
                .or_else(|| health.pointer("/persistent_state/free"))
                .and_then(Value::as_u64)
                .or_else(|| {
                    capacity
                        .zip(allocated)
                        .map(|(capacity, used)| capacity.saturating_sub(used))
                });
            json!({
                "model":health.get("model").cloned().unwrap_or(Value::Null),
                "capacity":capacity,
                "allocated":allocated,
                "free":free,
            })
        })
        .collect::<Vec<_>>();
    let fully_reported = !providers.is_empty()
        && providers
            .iter()
            .all(|provider| provider.get("capacity").is_some_and(Value::is_u64));
    let every_provider_has_free = providers.iter().all(|provider| {
        provider
            .get("free")
            .and_then(Value::as_u64)
            .is_some_and(|free| free > 0)
    });
    let (status, error, ready) = if !fully_reported {
        (
            "unsupported",
            "Sidecar health does not report persistent State capacity",
            false,
        )
    } else if !every_provider_has_free {
        (
            "unavailable",
            "persistent State capacity is exhausted",
            false,
        )
    } else {
        ("ready", "", true)
    };
    (
        json!({"status":status,"providers":providers,"error":error}),
        ready,
    )
}

fn owner_id(prefix: &str, session_id: &str) -> String {
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    session_id.hash(&mut hash);
    format!(
        "{prefix}-{:016x}-{:016x}",
        hash.finish(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

fn chat_owner_id(session_id: &str) -> String {
    let digest = ring::digest::digest(&ring::digest::SHA256, session_id.as_bytes());
    let stable = digest
        .as_ref()
        .iter()
        .take(16)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("chat-session-{stable}")
}

fn legacy_request_id(task_id: Option<&str>) -> String {
    task_id.map_or_else(
        || owner_id("request", "legacy"),
        |task_id| format!("legacy-{task_id}"),
    )
}

fn required_string<'a>(
    arguments: &'a serde_json::Map<String, Value>,
    tool: &str,
    name: &str,
) -> Result<&'a str, String> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{tool}.{name} must be a string"))
}

fn required_string_value<'a>(
    arguments: &'a Value,
    tool: &str,
    name: &str,
) -> Result<&'a str, String> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{tool}.{name} must be a string"))
}

fn workspace_file_tool_definitions() -> [ToolDefinition; 3] {
    [
        ToolDefinition::one_string(
            "read_file",
            "Read one UTF-8 text file inside the workspace",
            "path",
        ),
        ToolDefinition {
            name: "write_file".into(),
            description: "Atomically create or replace one UTF-8 text file inside the workspace"
                .into(),
            arguments: vec![
                ArgumentSpec::required_string("path"),
                ArgumentSpec::required_string("content"),
            ],
            allow_extra_arguments: false,
        },
        ToolDefinition {
            name: "edit_file".into(),
            description: "Replace old_text exactly once with new_text in one UTF-8 workspace file"
                .into(),
            arguments: vec![
                ArgumentSpec::required_string("path"),
                ArgumentSpec::required_string("old_text"),
                ArgumentSpec::required_string("new_text"),
            ],
            allow_extra_arguments: false,
        },
    ]
}

fn attach_task_ledger(response: &mut Value, record: &TaskRecord) {
    let Some(object) = response.as_object_mut() else {
        return;
    };
    object.insert("task_id".into(), json!(record.task_id));
    if let Some(trace_id) = &record.trace_id {
        object.insert("trace_id".into(), json!(trace_id));
    }
    if let Some(capture) = &record.debug_capture {
        object.insert("debug_capture".into(), capture.clone());
    }
    let trace = object
        .entry("trace")
        .or_insert_with(|| json!({}))
        .as_object_mut();
    if let Some(trace) = trace {
        trace.insert(
            "task_ledger".into(),
            json!({
                "task_id":record.task_id,
                "status":record.status,
                "revision":record.revision,
                "recovery_count":record.recovery_count,
                "current_stage":record.current_stage,
                "stages":record.stages.iter().map(|stage|json!({
                    "id":stage.spec.id,
                    "status":stage.status,
                    "attempts":stage.attempts,
                    "started_unix_ms":stage.started_unix_ms,
                    "completed_unix_ms":stage.completed_unix_ms,
                    "error":stage.error,
                })).collect::<Vec<_>>(),
            }),
        );
    }
}

fn attach_debug_capture(
    response: &mut Value,
    trace_id: Option<&str>,
    capture: Option<&DebugCapture>,
) {
    let Some(object) = response.as_object_mut() else {
        return;
    };
    if let Some(trace_id) = trace_id {
        object.insert("trace_id".into(), json!(trace_id));
    }
    if let Some(capture) = capture
        && let Ok(value) = serde_json::to_value(capture)
    {
        object.insert("debug_capture".into(), value);
    }
}

fn debug_error_markers(capture: Option<&DebugCapture>) -> String {
    let Some(capture) = capture else {
        return String::new();
    };
    let mut markers = format!(
        "; debug_capture_status={}; debug_capture_mode={}",
        capture.status, capture.mode
    );
    if let Some(error) = &capture.error {
        markers.push_str("; debug_trace_incomplete=");
        markers.push_str(error);
    }
    markers
}

async fn debug_record<F>(
    trace: Option<&DebugTraceHandle>,
    category: &'static str,
    component: &str,
    event_type: &str,
    stage_id: Option<&str>,
    payload: F,
) where
    F: FnOnce() -> Value,
{
    if let Some(trace) = trace {
        let _ = trace
            .record(category, component, event_type, stage_id, payload())
            .await;
    }
}

async fn record_agent_events(
    trace: Option<&DebugTraceHandle>,
    stage_id: Option<&str>,
    events: &[AgentEvent],
) {
    let Some(trace) = trace else {
        return;
    };
    for event in events {
        let (category, component, event_type, payload) = match event {
            AgentEvent::RunStarted { owner_id } => (
                "service",
                "agent_loop",
                "run_started",
                json!({"agent_owner_id":owner_id}),
            ),
            AgentEvent::StateOpened { state_id } => (
                "state",
                "agent_loop",
                "state_opened",
                json!({"state_id":state_id}),
            ),
            AgentEvent::ModelCompleted {
                turn,
                state_id,
                action,
                provider_input,
                raw_output,
                stop_reason,
                max_tokens,
            } => (
                "model",
                "provider",
                "model_completed",
                json!({
                    "turn":turn,
                    "state_id":state_id,
                    "action":action,
                    "provider_request":{
                        "input":provider_input,
                        "stops":["</tool_call>","</answer>"],
                        "max_tokens":max_tokens
                    },
                    "provider_response":{
                        "raw_output":raw_output,
                        "stop_reason":stop_reason
                    }
                }),
            ),
            AgentEvent::ProtocolRejected {
                turn,
                retry,
                message,
                output_preview,
                provider_input,
                raw_output,
                stop_reason,
                max_tokens,
            } => (
                "model",
                "strict_protocol",
                "protocol_rejected",
                json!({
                    "turn":turn,
                    "retry":retry,
                    "message":message,
                    "output_preview":output_preview,
                    "provider_request":{
                        "input":provider_input,
                        "stops":["</tool_call>","</answer>"],
                        "max_tokens":max_tokens
                    },
                    "provider_response":{
                        "raw_output":raw_output,
                        "stop_reason":stop_reason
                    }
                }),
            ),
            AgentEvent::ControllerToolScheduled { step, name } => (
                "tools",
                "controller",
                "controller_tool_scheduled",
                json!({"step":step,"name":name}),
            ),
            AgentEvent::ToolStarted { step, name } => (
                "tools",
                "tool_registry",
                "tool_started",
                json!({"step":step,"name":name}),
            ),
            AgentEvent::ToolCompleted {
                step,
                name,
                status,
                arguments,
                result,
            } => (
                "tools",
                "sandbox",
                "tool_completed",
                json!({"step":step,"name":name,"status":status,"arguments":arguments,"result":result}),
            ),
            AgentEvent::AnswerCompleted { answer } => (
                "model",
                "strict_protocol",
                "answer_completed",
                json!({"answer":answer}),
            ),
            AgentEvent::AnswerRejected {
                retry,
                require_tool,
                feedback,
            } => (
                "model",
                "strict_protocol",
                "answer_rejected",
                json!({"retry":retry,"require_tool":require_tool,"feedback":feedback}),
            ),
            AgentEvent::RunFailed { code, message } => (
                "service",
                "agent_loop",
                "run_failed",
                json!({"error_code":code,"message":message}),
            ),
            AgentEvent::StateReleased {
                state_id,
                success,
                error,
            } => (
                "state",
                "agent_loop",
                "state_released",
                json!({"state_id":state_id,"success":success,"error":error}),
            ),
        };
        debug_record(
            Some(trace),
            category,
            component,
            event_type,
            stage_id,
            || payload,
        )
        .await;
    }
}

fn replay_task_record(record: &TaskRecord) -> Result<Value, String> {
    if record.status != TaskStatus::Succeeded {
        return Err(format!(
            "request_id `{}` already exists with task status {:?}; use the task control API",
            record.request_id, record.status
        ));
    }
    let mut response = record
        .final_response
        .clone()
        .ok_or_else(|| format!("succeeded task `{}` has no final response", record.task_id))?;
    attach_task_ledger(&mut response, record);
    Ok(response)
}

fn task_requires_mutation(task: &str) -> bool {
    let value = task.to_lowercase();
    [
        "create",
        "write",
        "fix",
        "repair",
        "edit",
        "modify",
        "update",
        "replace",
        "implement",
        "add",
        "remove",
        "delete",
        "rename",
        "创建",
        "写入",
        "修复",
        "修改",
        "更新",
        "替换",
        "实现",
        "新增",
        "删除",
        "重命名",
    ]
    .into_iter()
    .any(|marker| value.contains(marker))
}

fn command_looks_like_verification(command: &str) -> bool {
    let value = command.to_lowercase();
    // A write command must never satisfy the independent verification phase
    // merely because it happens to use `cat`, `python3 -c`, or another token
    // that can also appear in a read-only check.  In particular, treating a
    // heredoc such as `cat > artifact <<EOF` as verification allowed a
    // workspace task to commit immediately after the write without observing
    // a real validator result.
    if command_looks_like_mutation(&value) {
        return false;
    }
    [
        " test",
        "test_",
        "pytest",
        "cargo test",
        "go test",
        "npm test",
        "pnpm test",
        "yarn test",
        "bun test",
        "ctest",
        " cat ",
        "cat ",
        "grep ",
        "diff ",
        "cmp ",
        "sha256sum",
        "md5sum",
        "wc ",
        "head ",
        "tail ",
        "read_file(",
        "python3 -c",
        "python -c",
    ]
    .into_iter()
    .any(|marker| value.contains(marker))
}

fn command_looks_like_mutation(value: &str) -> bool {
    let padded = format!(" {value} ");
    [
        " > ",
        ">>",
        "<<",
        " tee ",
        " touch ",
        " mkdir ",
        " rm ",
        " mv ",
        " cp ",
        " install ",
        " truncate ",
        " sed -i",
        " perl -i",
        " write_file(",
        " edit_file(",
    ]
    .into_iter()
    .any(|marker| padded.contains(marker))
}

fn command_looks_like_inspection(command: &str) -> bool {
    let value = format!(" {} ", command.to_lowercase());
    [
        " cat ",
        " sed -n ",
        " head ",
        " tail ",
        " grep ",
        " rg ",
        " read_file(",
    ]
    .into_iter()
    .any(|marker| value.contains(marker))
}

fn strip_leading_think(raw: &str) -> (String, bool) {
    let mut remaining = raw.trim_start();
    let mut stripped = false;
    while remaining.starts_with("<think>") {
        let Some(end) = remaining.find("</think>") else {
            return (String::new(), true);
        };
        remaining = remaining[end + "</think>".len()..].trim_start();
        stripped = true;
    }
    (remaining.trim().to_string(), stripped)
}

fn chat_state_cache_safe(generated_usable: bool, stop_reason: &str) -> bool {
    generated_usable && stop_reason != "\nSystem:"
}

fn contains_chinese(value: &str) -> bool {
    value
        .chars()
        .any(|character| ('\u{3400}'..='\u{9fff}').contains(&character))
}

fn insufficient(question: &str) -> String {
    if contains_chinese(question) {
        "现有证据不足以可靠回答。".into()
    } else {
        "The available evidence is insufficient for a reliable answer.".into()
    }
}

fn generation_failure(question: &str) -> String {
    if contains_chinese(question) {
        "这次没有生成可用回答，请重试。".into()
    } else {
        "No usable answer was generated. Please try again.".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            name: name.into(),
            arguments: arguments.as_object().unwrap().clone(),
        }
    }

    #[test]
    fn strips_only_complete_leading_reasoning() {
        assert_eq!(
            strip_leading_think("<think>x</think> answer"),
            ("answer".into(), true)
        );
        assert_eq!(strip_leading_think("answer"), ("answer".into(), false));
        assert_eq!(
            strip_leading_think("<think>unfinished"),
            (String::new(), true)
        );
    }

    #[test]
    fn complete_hidden_reasoning_does_not_disable_state_reuse() {
        let (answer, reasoning_stripped) = strip_leading_think("<think>private</think> visible");
        assert!(reasoning_stripped);
        assert!(chat_state_cache_safe(!answer.is_empty(), "\n\nUser:"));
        assert!(!chat_state_cache_safe(false, "\n\nUser:"));
        assert!(!chat_state_cache_safe(true, "\nSystem:"));
    }

    #[test]
    fn registry_is_minimal_and_command_is_opt_in() {
        let mut config = RuntimeConfig::default();
        config.command.enabled = false;
        assert!(!config.command.enabled);
    }

    #[test]
    fn readiness_requires_explicit_free_persistent_state_capacity() {
        let (available, ready) = state_capacity_status(
            &[json!({
                "model":"rwkv-test",
                "persistent_states":{"capacity":4,"allocated":3,"free":1}
            })],
            true,
        );
        assert!(ready);
        assert_eq!(available["status"], "ready");
        assert_eq!(available["providers"][0]["free"], 1);

        let (exhausted, ready) = state_capacity_status(
            &[json!({"persistent_states":{"capacity":4,"allocated":4,"free":0}})],
            true,
        );
        assert!(!ready);
        assert_eq!(exhausted["status"], "unavailable");

        let (unknown, ready) = state_capacity_status(&[json!({"status":"ready"})], true);
        assert!(!ready);
        assert_eq!(unknown["status"], "unsupported");
    }

    #[test]
    fn workspace_file_tool_schemas_are_strict() {
        let mut registry = ToolRegistry::default();
        for definition in workspace_file_tool_definitions() {
            registry.register(definition).unwrap();
        }
        let valid = ToolCall {
            name: "edit_file".into(),
            arguments: json!({
                "path":"calc.py",
                "old_text":"return a - b",
                "new_text":"return a + b",
            })
            .as_object()
            .unwrap()
            .clone(),
        };
        assert!(registry.validate(&valid).is_ok());
        let mut invalid = valid;
        invalid.arguments.insert("extra".into(), json!(true));
        assert!(registry.validate(&invalid).is_err());
    }

    #[test]
    fn workspace_execution_requires_real_change_then_verification() {
        let mut state =
            WorkspaceExecution::new("Run python3 test_calc.py, fix calc.py, and rerun the test.");
        state.record(
            "python3 test_calc.py",
            &json!({"status":"error","stderr":"AssertionError"}),
            false,
        );
        assert!(state.next_instruction().unwrap().contains("failed"));

        state.record("cat calc.py", &json!({"status":"ok","stdout":"bad"}), false);
        assert!(state.next_instruction().unwrap().contains("phase=MUTATE"));

        state.record("sed -i s/bad/good/ calc.py", &json!({"status":"ok"}), true);
        assert!(state.next_instruction().unwrap().contains("verification"));

        state.record(
            "python3 test_calc.py",
            &json!({"status":"ok","stdout":"tests passed"}),
            false,
        );
        assert!(state.next_instruction().is_none());
        assert!(state.committed_answer().unwrap().contains("tests passed"));
    }

    #[test]
    fn failed_command_after_inspection_refreshes_root_worker() {
        let mut state = WorkspaceExecution::new("Inspect, fix, and test calc.py.");
        state.record(
            "cat calc.py",
            &json!({"status":"ok","stdout":"return a - b"}),
            false,
        );
        state.record(
            "cat test_calc.py",
            &json!({"status":"ok","stdout":"assert add(2, 3) == 5"}),
            false,
        );
        state.record(
            "python3 test_calc.py",
            &json!({"status":"error","stderr":"AssertionError"}),
            false,
        );
        assert!(state.should_refresh_worker());
    }

    #[test]
    fn file_tool_policy_error_is_preserved_as_recovery_evidence() {
        let mut state = WorkspaceExecution::new("Inspect inputs and create report.md.");
        state.record(
            "read_file(report.md)",
            &json!({"status":"error","message":"workspace file does not exist"}),
            false,
        );
        let instruction = state.next_instruction().unwrap();
        assert!(instruction.contains("workspace file does not exist"));
        assert!(instruction.contains("Do not repeat it unchanged"));
    }

    #[test]
    fn workspace_phase_commits_mutation_then_verification_tool_names() {
        let mut state =
            WorkspaceExecution::new("Inspect inputs, implement output.py, and test it.");
        state.record(
            "read_file(SPEC.md)",
            &json!({"status":"ok","stdout":"spec"}),
            false,
        );
        state.record(
            "read_file(input.json)",
            &json!({"status":"ok","stdout":"{}"}),
            false,
        );
        state.record(
            "read_file(tests.py)",
            &json!({"status":"ok","stdout":"assert True"}),
            false,
        );
        assert_eq!(state.tool_call_prefix(), "<answer>");
        let write = state
            .controller_tool_from_answer("print('ready')\n")
            .unwrap();
        assert_eq!(write.name, "write_file");
        assert_eq!(write.arguments["path"], "output.py");
        assert_eq!(write.arguments["content"], "print('ready')");
        state.record("write_file(output.py)", &json!({"status":"ok"}), true);
        assert_eq!(
            state.tool_call_prefix(),
            "<tool_call>{\"name\":\"run_command\",\"arguments\":{\"command\":"
        );
    }

    #[test]
    fn controller_extracts_one_complete_fenced_replacement_without_relaxing_protocol() {
        let mut state = WorkspaceExecution::new("Inspect and create output.py.");
        state.record(
            "read_file(SPEC.md)",
            &json!({"status":"ok","stdout":"spec"}),
            false,
        );
        state.record(
            "read_file(input.txt)",
            &json!({"status":"ok","stdout":"input"}),
            false,
        );
        state.record(
            "read_file(tests.py)",
            &json!({"status":"ok","stdout":"assert True"}),
            false,
        );
        let instruction = state.next_instruction().unwrap();
        assert!(instruction.contains("spec"));
        assert!(instruction.contains("input"));
        assert!(instruction.contains("assert True"));
        let write = state
            .controller_tool_from_answer("Here is the file:\n```python\nprint('ok')\n```\nignored")
            .unwrap();
        assert_eq!(write.name, "write_file");
        assert_eq!(write.arguments["content"], "print('ok')");
    }

    #[test]
    fn controller_schedules_distinct_nonempty_inventory_reads() {
        let task = TaskSpec::new("Repair the package and run its tests.");
        let mut state = WorkspaceExecution::from_task_spec_with_inventory(
            &task,
            "",
            &[
                "inventory.py (50 bytes)".into(),
                "tests/__init__.py (0 bytes)".into(),
                "tests/test_parse.py (50 bytes)".into(),
                "tests/test_total.py (50 bytes)".into(),
            ],
        );
        state.record(
            "read_file(inventory.py)",
            &json!({"status":"ok","stdout":"broken"}),
            false,
        );
        assert_eq!(
            state.controller_followup_call().unwrap().arguments["path"],
            "tests/test_parse.py"
        );
        state.record(
            "read_file(tests/test_parse.py)",
            &json!({"status":"ok","stdout":"parse test"}),
            false,
        );
        assert_eq!(
            state.controller_followup_call().unwrap().arguments["path"],
            "tests/test_total.py"
        );
    }

    #[test]
    fn repair_phase_refreshes_worker_and_prefers_exact_edit() {
        let mut state = WorkspaceExecution::new("Inspect and repair calc.py, then run tests.");
        state.inspection_target_count = 2;
        state.record(
            "read_file(calc.py)",
            &json!({"status":"ok","stdout":"return a - b"}),
            false,
        );
        state.record(
            "read_file(tests/test_calc.py)",
            &json!({"status":"ok","stdout":"assert add(2, 3) == 5"}),
            false,
        );
        assert!(state.should_refresh_worker());
        assert!(state.inspection_evidence.contains("calc.py"));
        assert!(state.inspection_evidence.contains("tests/test_calc.py"));
        assert_eq!(state.tool_call_prefix(), "<answer>");
        let mutation_instruction = state.next_instruction().unwrap();
        assert!(mutation_instruction.contains("Exact current target evidence:\nreturn a - b"));
        assert!(mutation_instruction.contains("Grounded read-only task evidence"));
        assert!(mutation_instruction.contains("assert add(2, 3) == 5"));
        assert!(
            state
                .controller_tool_from_answer("return a - b\n")
                .is_none()
        );
        assert!(
            state
                .answer_payload_rejection("return a - b\n")
                .unwrap()
                .contains("content-identical")
        );
        let edit = state.controller_tool_from_answer("return a + b\n").unwrap();
        assert_eq!(edit.name, "edit_file");
        assert_eq!(edit.arguments["path"], "calc.py");
        assert_eq!(edit.arguments["old_text"], "return a - b");
        assert_eq!(edit.arguments["new_text"], "return a + b");
        assert!(
            state
                .phase_rejection(&call("read_file", json!({"path":"calc.py"})))
                .is_some()
        );
        assert!(
            state
                .phase_rejection(&call("run_command", json!({"command":"cat calc.py"})))
                .is_some()
        );
        assert!(
            state
                .phase_rejection(&call(
                    "edit_file",
                    json!({"path":"calc.py","old_text":"a","new_text":"b"})
                ))
                .is_none()
        );
        state.record(
            "edit_file(missing.py)",
            &json!({"status":"error","message":"missing"}),
            false,
        );
        assert_eq!(state.tool_call_prefix(), "<answer>");
    }

    #[test]
    fn successful_mutation_requires_verification_tool_before_more_edits() {
        let mut task = TaskSpec::new("Create output.py and test it.");
        task.requires_mutation = Some(true);
        task.verification_commands = vec!["python3 tests.py".into()];
        let mut state = WorkspaceExecution::from_task_spec(&task, "");
        state.record("write_file(output.py)", &json!({"status":"ok"}), true);
        assert!(
            state
                .phase_rejection(&call(
                    "write_file",
                    json!({"path":"output.py","content":"again"})
                ))
                .is_some()
        );
        assert!(
            state
                .phase_rejection(&call("read_file", json!({"path":"output.py"})))
                .is_some()
        );
        let verifier = call("run_command", json!({"command":"python3 tests.py"}));
        assert!(state.phase_rejection(&verifier).is_none());
        let scheduled = state.controller_followup_call().unwrap();
        assert_eq!(scheduled.name, "run_command");
        assert_eq!(scheduled.arguments["command"], "python3 tests.py");
        state.record(
            "python3 tests.py",
            &json!({"status":"error","stderr":"failed"}),
            false,
        );
        let read = state.controller_followup_call().unwrap();
        assert_eq!(read.name, "read_file");
        assert_eq!(read.arguments["path"], "output.py");
        state.record(
            "read_file(output.py)",
            &json!({"status":"ok","stdout":"broken"}),
            false,
        );
        assert!(state.should_refresh_worker());
        assert!(state.next_instruction().unwrap().contains("phase=MUTATE"));
        assert_eq!(state.tool_call_prefix(), "<answer>");
    }

    #[test]
    fn task_spec_constraints_ground_a_unique_new_target_without_keyword_guessing() {
        let mut task =
            TaskSpec::new("阅读 SPEC.md 和 validator.py，编写 migrate.py，然后检查 service.json。");
        task.requires_mutation = Some(true);
        task.constraints = vec!["不得修改 validator.py 或现有 JSON。".into()];
        task.verification_commands = vec!["python3 migrate.py --check service.json".into()];
        let state = WorkspaceExecution::from_task_spec_with_inventory(
            &task,
            "",
            &[
                "SPEC.md (10 bytes)".into(),
                "service.json (20 bytes)".into(),
                "validator.py (30 bytes)".into(),
            ],
        );

        assert_eq!(
            state.mutation_plan(),
            Some(MutationPlan {
                tool: "write_file",
                path: "migrate.py".into(),
            })
        );
        assert!(state.path_is_protected("validator.py"));
        let rejected = state
            .phase_rejection(&call(
                "edit_file",
                json!({"path":"validator.py","old_text":"a","new_text":"b"}),
            ))
            .unwrap();
        assert_eq!(rejected["code"], "protected_path");
    }

    #[test]
    fn controller_runs_baseline_verifier_after_named_inspection_before_mutation() {
        let mut task = TaskSpec::new(
            "Inspect inventory.py and tests/test_parse.py, then repair inventory.py.",
        );
        task.requires_mutation = Some(true);
        task.constraints = vec!["Do not modify files under tests/.".into()];
        task.verification_commands = vec!["python3 -m unittest tests.test_parse -v".into()];
        let mut state = WorkspaceExecution::from_task_spec_with_inventory(
            &task,
            "",
            &[
                "inventory.py (50 bytes)".into(),
                "tests/test_parse.py (50 bytes)".into(),
                "tests/test_total.py (50 bytes)".into(),
            ],
        );
        assert_eq!(state.inspection_target_count, 2);
        state.record(
            "read_file(inventory.py)",
            &json!({"status":"ok","stdout":"broken"}),
            false,
        );
        let next = state.controller_followup_call().unwrap();
        assert_eq!(next.name, "read_file");
        assert_eq!(next.arguments["path"], "tests/test_parse.py");
        state.record(
            "read_file(tests/test_parse.py)",
            &json!({"status":"ok","stdout":"assert parse('a=1')"}),
            false,
        );
        let verifier = state.controller_followup_call().unwrap();
        assert_eq!(verifier.name, "run_command");
        assert_eq!(
            verifier.arguments["command"],
            "python3 -m unittest tests.test_parse -v"
        );
    }

    #[test]
    fn latest_verifier_failure_replaces_stale_repair_evidence() {
        let mut task = TaskSpec::new("Repair summarize.py and run tests.py.");
        task.requires_mutation = Some(true);
        task.verification_commands = vec!["python3 tests.py".into()];
        let mut state = WorkspaceExecution::from_task_spec_with_inventory(
            &task,
            "",
            &[
                "summarize.py (10 bytes)".into(),
                "tests.py (10 bytes)".into(),
            ],
        );
        state.record(
            "python3 tests.py",
            &json!({"status":"error","stderr":"duplicate id"}),
            false,
        );
        state.record(
            "read_file(summarize.py)",
            &json!({"status":"ok","stdout":"broken"}),
            false,
        );
        state.record("edit_file(summarize.py)", &json!({"status":"ok"}), true);
        state.record(
            "python3 tests.py",
            &json!({"status":"error","stderr":"NameError: json is not defined"}),
            false,
        );

        assert!(state.failure_evidence.contains("NameError"));
        assert!(!state.failure_evidence.contains("duplicate id"));
    }

    #[test]
    fn read_only_workspace_task_can_finish_with_inspection() {
        let mut state = WorkspaceExecution::new("Read note.txt and report its content.");
        state.record(
            "cat note.txt",
            &json!({"status":"ok","stdout":"hello"}),
            false,
        );
        assert!(state.verification_passed);
        assert!(state.committed_answer().unwrap().contains("hello"));
    }

    #[test]
    fn artifact_write_is_not_mistaken_for_verification() {
        let mut state = WorkspaceExecution::new("Create content.json and validate it.");
        state.record(
            "cat > content.json <<'EOF'\n{\"title\":\"Nova\"}\nEOF",
            &json!({"status":"ok"}),
            true,
        );
        assert!(!state.verification_passed);
        assert!(state.next_instruction().unwrap().contains("verification"));

        state.record(
            "python3 -c \"import json; json.load(open('content.json')); print('VALID')\"",
            &json!({"status":"ok","stdout":"VALID"}),
            false,
        );
        assert!(state.verification_passed);
        assert!(state.committed_answer().unwrap().contains("VALID"));
    }

    #[test]
    fn atomic_artifact_write_and_explicit_validator_can_commit() {
        let mut state = WorkspaceExecution::new("Create content.json and validate it.");
        state.record(
            "cat > content.json <<'EOF'\n{\"title\":\"Nova\"}\nEOF\npython3 -c \"import json; json.load(open('content.json')); print('VALID')\"",
            &json!({"status":"ok","stdout":"VALID\n"}),
            true,
        );
        assert!(state.verification_passed);
        assert!(state.committed_answer().unwrap().contains("VALID"));
    }

    #[test]
    fn task_spec_requires_the_declared_verifier_exactly() {
        let mut task = TaskSpec::legacy("Fix calc.py", Some("/repo".into()));
        task.requires_mutation = Some(true);
        task.acceptance_criteria = vec!["test_calc.py passes".into()];
        task.verification_commands = vec!["python3 /repo/test_calc.py".into()];
        let mut state = WorkspaceExecution::from_task_spec(&task, "/repo");

        state.record("sed -i s/bad/good/ calc.py", &json!({"status":"ok"}), true);
        state.record(
            "cat calc.py",
            &json!({"status":"ok","stdout":"good"}),
            false,
        );
        assert!(!state.verification_passed);
        assert!(
            state
                .next_instruction()
                .unwrap()
                .contains("python3 /workspace/test_calc.py")
        );

        state.record(
            "python3 /workspace/test_calc.py",
            &json!({"status":"ok","stdout":"PASS"}),
            false,
        );
        assert!(state.verification_passed);
    }

    #[test]
    fn task_spec_can_override_mutation_heuristic_for_read_only_work() {
        let mut task = TaskSpec::legacy("Check and report update notes", Some("/repo".into()));
        task.requires_mutation = Some(false);
        task.verification_commands = vec!["cat CHANGELOG.md".into()];
        let mut state = WorkspaceExecution::from_task_spec(&task, "/repo");
        state.record(
            "cat CHANGELOG.md",
            &json!({"status":"ok","stdout":"v1.2"}),
            false,
        );
        assert!(state.verification_passed);
    }
}
