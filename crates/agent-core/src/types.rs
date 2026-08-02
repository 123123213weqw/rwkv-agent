use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Tool(ToolCall),
    Answer(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateHandle {
    pub endpoint: String,
    pub owner_id: String,
    pub state_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenStateRequest {
    pub owner_id: String,
    pub root_prompt: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateContinueRequest {
    pub state: StateHandle,
    pub input: String,
    pub stops: Vec<String>,
    pub max_tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelOutput {
    pub state_id: String,
    pub text: String,
    pub stop_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AgentRunRequest {
    pub owner_id: String,
    pub root_prompt: String,
    pub initial_input: String,
}

impl AgentRunRequest {
    pub fn validate(&self) -> Result<(), AgentError> {
        if self.owner_id.trim().is_empty() {
            return Err(AgentError::InvalidRequest(
                "owner_id must not be empty".into(),
            ));
        }
        if self.root_prompt.trim().is_empty() {
            return Err(AgentError::InvalidRequest(
                "root_prompt must not be empty".into(),
            ));
        }
        if self.initial_input.trim().is_empty() {
            return Err(AgentError::InvalidRequest(
                "initial_input must not be empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct RunLimits {
    pub max_tool_steps: usize,
    pub max_tokens_per_turn: u32,
    pub max_elapsed: Duration,
    /// Commit the answer envelope after the first observation. This is the
    /// stable ordinary-tool path; multi-step command agents leave it disabled.
    pub answer_after_tool: bool,
}

impl Default for RunLimits {
    fn default() -> Self {
        Self {
            max_tool_steps: 6,
            max_tokens_per_turn: 192,
            max_elapsed: Duration::from_secs(180),
            answer_after_tool: false,
        }
    }
}

impl RunLimits {
    pub fn validate(&self) -> Result<(), AgentError> {
        if self.max_tool_steps == 0 {
            return Err(AgentError::InvalidRequest(
                "max_tool_steps must be positive".into(),
            ));
        }
        if self.max_tokens_per_turn == 0 {
            return Err(AgentError::InvalidRequest(
                "max_tokens_per_turn must be positive".into(),
            ));
        }
        if self.max_elapsed.is_zero() {
            return Err(AgentError::InvalidRequest(
                "max_elapsed must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug)]
pub struct RunContext {
    pub deadline: Instant,
    pub cancellation: CancellationToken,
}

impl RunContext {
    pub fn check(&self) -> Result<(), AgentError> {
        if self.cancellation.is_cancelled() {
            return Err(AgentError::Cancelled);
        }
        if Instant::now() >= self.deadline {
            return Err(AgentError::DeadlineExceeded);
        }
        Ok(())
    }

    pub fn remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunReport {
    pub answer: String,
    pub state_id: String,
    pub model_turns: usize,
    pub tool_steps: usize,
    pub tools: Vec<ToolStep>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolStep {
    pub step: usize,
    pub call: ToolCall,
    pub result: Value,
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("protocol error: {0}")]
    Protocol(#[from] crate::ProtocolError),
    #[error("state changed from {expected} to {actual}")]
    StateChanged { expected: String, actual: String },
    #[error("state owner changed from {expected} to {actual}")]
    OwnerChanged { expected: String, actual: String },
    #[error("tool-step budget exceeded: {max_steps}")]
    BudgetExceeded { max_steps: usize },
    #[error("run cancelled")]
    Cancelled,
    #[error("run deadline exceeded")]
    DeadlineExceeded,
    #[error("state release failed: {0}")]
    Release(String),
    #[error("run failed ({run}) and state release also failed: {release}")]
    RunAndRelease {
        run: Box<AgentError>,
        release: String,
    },
}

impl AgentError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => "invalid_request",
            Self::Model(_) => "model_error",
            Self::Protocol(_) => "protocol_error",
            Self::StateChanged { .. } => "state_changed",
            Self::OwnerChanged { .. } => "owner_changed",
            Self::BudgetExceeded { .. } => "budget_exceeded",
            Self::Cancelled => "cancelled",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::Release(_) => "release_error",
            Self::RunAndRelease { .. } => "run_and_release_error",
        }
    }
}
