use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    Tool,
    Answer,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    RunStarted {
        owner_id: String,
    },
    StateOpened {
        state_id: String,
    },
    ModelCompleted {
        turn: usize,
        state_id: String,
        action: ActionKind,
        /// Exact provider input retained in memory only for an explicitly
        /// enabled local Debug Trace.
        #[serde(skip_serializing, default)]
        provider_input: String,
        /// Complete provider generation retained in memory only for an
        /// explicitly enabled local Debug Trace. Ordinary Agent events and
        /// durable Task Ledger serialization never contain this body.
        #[serde(skip_serializing, default)]
        raw_output: String,
        #[serde(skip_serializing, default)]
        stop_reason: Option<String>,
        #[serde(skip_serializing, default)]
        max_tokens: u32,
    },
    ProtocolRejected {
        turn: usize,
        retry: usize,
        message: String,
        /// Bounded raw model output retained for protocol debugging. This is
        /// observation only: the strict parser remains the sole authority and
        /// the preview is never fed back as executable input.
        output_preview: String,
        #[serde(skip_serializing, default)]
        provider_input: String,
        #[serde(skip_serializing, default)]
        raw_output: String,
        #[serde(skip_serializing, default)]
        stop_reason: Option<String>,
        #[serde(skip_serializing, default)]
        max_tokens: u32,
    },
    ControllerToolScheduled {
        step: usize,
        name: String,
    },
    ToolStarted {
        step: usize,
        name: String,
    },
    ToolCompleted {
        step: usize,
        name: String,
        status: String,
        #[serde(skip_serializing, default)]
        arguments: Map<String, Value>,
        #[serde(skip_serializing, default)]
        result: Value,
    },
    AnswerCompleted {
        answer: String,
    },
    AnswerRejected {
        retry: usize,
        require_tool: bool,
        feedback: String,
    },
    RunFailed {
        code: String,
        message: String,
    },
    StateReleased {
        state_id: String,
        success: bool,
        error: String,
    },
}

pub trait EventSink {
    fn emit(&mut self, event: AgentEvent);
}

#[derive(Clone, Debug, Default)]
pub struct VecEventSink {
    pub events: Vec<AgentEvent>,
}

impl EventSink for VecEventSink {
    fn emit(&mut self, event: AgentEvent) {
        self.events.push(event);
    }
}
