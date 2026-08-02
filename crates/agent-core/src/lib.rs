//! Runtime-neutral control plane for the RWKV Agent.

mod event;
mod protocol;
mod registry;
mod run_loop;
mod types;

pub use event::{ActionKind, AgentEvent, EventSink, VecEventSink};
pub use protocol::{ProtocolError, parse_action, render_answer_observation, render_observation};
pub use registry::{ArgumentSpec, JsonKind, RegistryError, ToolDefinition, ToolRegistry};
pub use run_loop::{AgentLoop, StateModel, ToolExecutor};
pub use types::{
    Action, AgentError, AgentRunRequest, CancellationToken, ModelOutput, OpenStateRequest,
    RunContext, RunLimits, RunReport, StateContinueRequest, StateHandle, ToolCall, ToolStep,
};
