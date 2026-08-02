use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rwkv_agent_core::{
    AgentLoop, AgentRunRequest, CancellationToken, RunContext, RunLimits, ToolCall, ToolDefinition,
    ToolExecutor, ToolRegistry, VecEventSink,
};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::command::{CommandPolicy, SandboxedCommand};
use crate::data_client::DataPlaneClient;
use crate::prompt;
use crate::research::ResearchRunner;
use crate::session::{Exchange, SessionStore};
use crate::sidecar::{GateDecision, SidecarClient, SidecarState};

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub model_urls: Vec<String>,
    pub data_plane_url: String,
    pub session_dir: PathBuf,
    pub tool_gate_threshold: f64,
    pub pasted_text_gate_threshold: f64,
    pub long_text_capture_chars: usize,
    pub chat_state_capacity: usize,
    pub max_tool_steps: usize,
    pub command: CommandPolicy,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            model_urls: vec!["http://127.0.0.1:8417".into()],
            data_plane_url: "http://127.0.0.1:8121".into(),
            session_dir: PathBuf::from("var/rust-agent-sessions"),
            tool_gate_threshold: -3.2,
            pasted_text_gate_threshold: -5.5,
            long_text_capture_chars: 4000,
            chat_state_capacity: 3,
            max_tool_steps: 6,
            command: CommandPolicy::default(),
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
}

#[derive(Clone, Debug)]
struct CachedChatState {
    session_id: String,
    state: SidecarState,
    history_len: usize,
    stop_reason: String,
    last_used: u64,
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
}

impl ToolExecutor for RuntimeTools {
    async fn execute(&mut self, call: ToolCall, context: RunContext) -> Result<Value, String> {
        context.check().map_err(|e| e.to_string())?;
        if call.name == "run_command" {
            let command = call
                .arguments
                .get("command")
                .and_then(Value::as_str)
                .ok_or_else(|| "run_command.command must be a string".to_string())?;
            self.command.execute(command).await
        } else {
            self.data.execute(call, context).await
        }
    }
}

impl AgentService {
    pub async fn new(config: RuntimeConfig) -> Result<Self, String> {
        if !(-20.0..=20.0).contains(&config.tool_gate_threshold)
            || !(-20.0..=20.0).contains(&config.pasted_text_gate_threshold)
        {
            return Err("tool Gate thresholds must be between -20 and 20".into());
        }
        if config.chat_state_capacity == 0 || config.max_tool_steps == 0 {
            return Err("State and tool capacities must be positive".into());
        }
        let sidecar = SidecarClient::new(config.model_urls.clone())?;
        let data = DataPlaneClient::new(config.data_plane_url.clone())?;
        let sessions = SessionStore::new(config.session_dir.clone()).await?;
        let command = SandboxedCommand::new(config.command.clone());
        Ok(Self {
            config: Arc::new(config),
            sidecar,
            data,
            sessions,
            chat_states: Arc::new(Mutex::new(ChatStateCache::default())),
            command,
        })
    }

    pub async fn health(&self) -> Result<Value, String> {
        let model = self.sidecar.health().await?;
        let data = self.data.health().await?;
        let states = self.chat_states.lock().await;
        Ok(json!({
            "status":"ready",
            "control_plane":"rust",
            "tools":self.tool_names(),
            "model":model,
            "data_plane":data,
            "context":{
                "mode":"recurrent_session_state_with_transcript_fallback",
                "history_messages":12,
                "long_term_memory":false,
                "session_state":{
                    "enabled":true,"mode":"gpu_recurrent_lru","capacity":self.config.chat_state_capacity,
                    "allocated":states.values.len(),"metrics":states.metrics,
                }
            },
            "tool_gate":{"mode":"semantic_single_token","threshold":self.config.tool_gate_threshold,
                "pasted_text_threshold":self.config.pasted_text_gate_threshold},
            "state_parallel_search":{"enabled":true,"endpoint":"/v1/agent/run_stateful","max_branches":4,"max_rounds":3},
            "command":{"enabled":self.config.command.enabled,"available":self.command.available(),"sandbox":"bubblewrap_no_network_no_unsafe_fallback"},
        }))
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
        if name == "run_command" {
            let command = arguments
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or_default();
            return self.command.execute(command).await;
        }
        self.data.call_tool(name, arguments, session_id, "").await
    }

    pub async fn run(&self, message: &str, session_id: &str) -> Result<Value, String> {
        let message = message.trim();
        if message.is_empty() {
            return Ok(json!({"status":"invalid","message":"message must not be empty"}));
        }
        let session_id = SessionStore::normalize_session_id(session_id)?;
        let _turn = self.sessions.lock(&session_id).await?;
        if message.chars().count() >= self.config.long_text_capture_chars {
            self.invalidate_chat_state(&session_id).await;
            let tool_result = self.data.capture_text(&session_id, message).await?;
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
                "trace":{"model_called":false,"context":{"mode":"transient_session_text"},"control_plane":"rust"},
            }));
        }
        let started = Instant::now();
        let history = self.sessions.history(&session_id, 12).await?;
        let context = prompt::bounded_context(&history, 8000);
        let routing_context = prompt::bounded_context(&history, 2000);
        let text_status = self.data.text_status(&session_id).await?;
        let has_text = text_status
            .get("active")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let gate = self
            .sidecar
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
            .await?;
        if !gate.use_tool {
            return self
                .direct_chat(message, &session_id, history, context, gate, started)
                .await;
        }
        self.invalidate_chat_state(&session_id).await;
        self.tool_run(message, &session_id, &context, has_text, gate, started)
            .await
    }

    pub async fn research(
        &self,
        message: &str,
        session_id: &str,
        branch_width: usize,
        max_rounds: usize,
    ) -> Result<Value, String> {
        let message = message.trim();
        if message.is_empty() {
            return Err("message must not be empty".into());
        }
        let session_id = SessionStore::normalize_session_id(session_id)?;
        let _turn = self.sessions.lock(&session_id).await?;
        self.invalidate_chat_state(&session_id).await;
        let runner = ResearchRunner::new(self.sidecar.clone(), self.data.clone());
        let result = runner
            .run(message, &session_id, branch_width, max_rounds)
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

    pub async fn shutdown(&self) {
        let records = {
            let mut cache = self.chat_states.lock().await;
            cache
                .values
                .drain()
                .map(|(_, value)| value)
                .collect::<Vec<_>>()
        };
        for record in &records {
            let _ = self.release_chat_record(record).await;
        }
    }

    async fn direct_chat(
        &self,
        message: &str,
        session_id: &str,
        history: Vec<Exchange>,
        context: String,
        gate: GateDecision,
        started: Instant,
    ) -> Result<Value, String> {
        let history_len = history.len();
        let (cached, stale) = self.chat_states.lock().await.take(session_id, history_len);
        if let Some(stale) = &stale {
            let _ = self.release_chat_record(stale).await;
        }
        let reused = cached.is_some();
        let mut state = if let Some(cached) = cached {
            cached
        } else {
            let owner_id = owner_id("chat", session_id);
            let state = self
                .sidecar
                .prefill(&owner_id, &prompt::direct_prefix(&context))
                .await?;
            CachedChatState {
                session_id: session_id.to_string(),
                state,
                history_len,
                stop_reason: String::new(),
                last_used: 0,
            }
        };
        let input = prompt::direct_turn(message, reused, &state.stop_reason);
        let stops = prompt::CHAT_STOPS
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>();
        let row = match self
            .sidecar
            .continue_one(&state.state, &input, &stops, 256)
            .await
        {
            Ok(row) => row,
            Err(error) => {
                let _ = self.release_chat_record(&state).await;
                return Err(error);
            }
        };
        if row.state_id != state.state.state_id {
            let _ = self.release_chat_record(&state).await;
            return Err("direct chat continuation changed state_id".into());
        }
        let (mut answer, reasoning_stripped) = strip_leading_think(&row.text);
        if answer.is_empty() {
            answer = generation_failure(message);
        }
        if let Err(error) = self.sessions.append(session_id, message, &answer).await {
            let _ = self.release_chat_record(&state).await;
            return Err(error);
        }
        state.history_len = history_len + 1;
        state.stop_reason = row.stop_reason.clone();
        state.state.seen_tokens = row.seen_tokens;
        let safe_to_cache = !reasoning_stripped && row.stop_reason != "\nSystem:";
        let mut released = Vec::new();
        if safe_to_cache {
            released = self
                .chat_states
                .lock()
                .await
                .put(state.clone(), self.config.chat_state_capacity);
        } else {
            self.release_chat_record(&state).await?;
        }
        for evicted in &released {
            let _ = self.release_chat_record(evicted).await;
        }
        Ok(json!({
            "status":"ok","session_id":session_id,"message":message,
            "route":{"mode":"direct","tool":null},"tool_result":null,"answer":answer,
            "trace":{
                "gate":gate,"context":{"history_messages":history_len,"mode":"recurrent_session_state","session_state":{
                    "used":true,"reused":reused,"cached":safe_to_cache,"cache_reject_reason":if reasoning_stripped{"hidden_reasoning_was_generated"}else if !safe_to_cache{"unsafe_stop_boundary"}else{""},"seen_tokens":row.seen_tokens}},
                "answer_completion":{"stop":row.stop_reason,"output_tokens":row.token_ids.len(),"model_elapsed_ms":row.elapsed_ms,"reasoning_stripped":reasoning_stripped},
                "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"control_plane":"rust",
            }
        }))
    }

    async fn tool_run(
        &self,
        message: &str,
        session_id: &str,
        context: &str,
        has_text: bool,
        gate: GateDecision,
        started: Instant,
    ) -> Result<Value, String> {
        let root_prompt =
            prompt::agent_prompt(message, has_text, context, self.config.command.enabled);
        let mut model = self.sidecar.clone();
        let mut tools = RuntimeTools {
            data: self.data.for_turn(session_id, message),
            command: self.command.clone(),
        };
        let registry = self.registry()?;
        let mut events = VecEventSink::default();
        let limits = RunLimits {
            max_tool_steps: self.config.max_tool_steps,
            max_tokens_per_turn: 192,
            max_elapsed: Duration::from_secs(180),
            answer_after_tool: !self.config.command.enabled,
        };
        let mut agent = AgentLoop::new(
            &mut model,
            &mut tools,
            &registry,
            &mut events,
            limits,
            CancellationToken::default(),
        )
        .map_err(|e| e.to_string())?;
        let report = agent
            .run(AgentRunRequest {
                owner_id: owner_id("turn", session_id),
                root_prompt,
                initial_input: "\n\nAssistant: <tool_call>".into(),
            })
            .await
            .map_err(|e| e.to_string())?;
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
            "trace":{"gate":gate,"agent":{"model_turns":report.model_turns,"tool_steps":tool_steps,"events":events.events},"answer_protocol":answer_protocol,
                "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"control_plane":"rust"}
        }))
    }

    fn registry(&self) -> Result<ToolRegistry, String> {
        let mut registry = ToolRegistry::default();
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
        if self.config.command.enabled {
            registry
                .register(ToolDefinition::one_string(
                    "run_command",
                    "Run one command in the isolated workspace",
                    "command",
                ))
                .map_err(|e| e.to_string())?;
        }
        Ok(registry)
    }

    fn tool_names(&self) -> Vec<&'static str> {
        let mut names = vec!["web_search", "knowledge_search", "long_text_qa"];
        if self.config.command.enabled {
            names.push("run_command");
        }
        names
    }

    async fn invalidate_chat_state(&self, session_id: &str) {
        if let Some(record) = self.chat_states.lock().await.pop(session_id) {
            let _ = self.release_chat_record(&record).await;
        }
    }

    async fn release_chat_record(&self, record: &CachedChatState) -> Result<(), String> {
        self.sidecar
            .release_many(
                &record.state.home_url,
                &record.state.owner_id,
                std::slice::from_ref(&record.state.state_id),
            )
            .await
            .map(|_| ())
    }
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
    fn registry_is_minimal_and_command_is_opt_in() {
        let mut config = RuntimeConfig::default();
        config.command.enabled = false;
        assert!(!config.command.enabled);
    }
}
