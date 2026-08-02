use serde::{Deserialize, Serialize};

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
    },
    ToolStarted {
        step: usize,
        name: String,
    },
    ToolCompleted {
        step: usize,
        name: String,
        status: String,
    },
    AnswerCompleted {
        answer: String,
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
