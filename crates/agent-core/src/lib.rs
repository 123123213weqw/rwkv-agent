//! Runtime-neutral control plane for the RWKV Agent.

mod event;
mod protocol;
mod registry;
mod run_loop;
mod task_spec;
mod types;

pub use event::{ActionKind, AgentEvent, EventSink, VecEventSink};
pub use protocol::{
    ProtocolError, TOOL_CALL_JSON_PREFIX, parse_action, render_answer_observation,
    render_answer_observation_with_reminder, render_observation, render_observation_with_progress,
    render_observation_with_progress_and_reminder,
    render_tool_observation_with_progress_and_reminder,
    render_tool_observation_with_progress_reminder_and_prefix,
};
pub use registry::{ArgumentSpec, JsonKind, RegistryError, ToolDefinition, ToolRegistry};
pub use run_loop::{AgentLoop, StateModel, ToolExecutor};
pub use task_spec::{TASK_SPEC_SCHEMA_VERSION, TaskSpec, TaskSpecError, TaskStageSpec};
pub use types::{
    Action, AgentError, AgentRunRequest, AnswerDecision, CancellationToken, ModelOutput,
    OpenStateRequest, RunContext, RunLimits, RunReport, StateContinueRequest, StateHandle,
    ToolCall, ToolStep,
};
