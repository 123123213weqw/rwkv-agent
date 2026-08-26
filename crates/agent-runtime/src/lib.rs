//! Production-oriented Rust control plane for RWKV Agent.
//!
//! CUDA inference and retrieval remain behind narrow HTTP data-plane APIs;
//! this crate owns recurrent-State lifecycle, routing, sessions and tool loops.

mod cloud_plugin;
mod command;
mod data_client;
mod debug_trace;
mod prompt;
mod research;
mod service;
mod session;
mod sidecar;
mod task_ledger;

pub use cloud_plugin::{
    CloudModelRef, CloudPluginClient, CloudPluginConfig, CloudPluginFallback, ExecutionPlan,
    PrivacyClass, WorkerZone,
};
pub use command::{CommandPolicy, SandboxedCommand};
pub use data_client::DataPlaneClient;
pub use debug_trace::{
    DEBUG_TRACE_SCHEMA_VERSION, DebugCapture, DebugTraceConfig, DebugTraceEvent,
    DebugTraceFileKind, DebugTraceFilter, DebugTraceHandle, DebugTraceManifest, DebugTraceMode,
    DebugTracePage, DebugTraceReadiness, DebugTraceStart, DebugTraceStore,
};
pub use research::ResearchRunner;
pub use rwkv_agent_core::{
    RequestIdentity, ResearchRequest, SERVICE_API_VERSION, ServiceErrorCode, ServiceErrorDetail,
    ServiceStreamEvent, TASK_SPEC_SCHEMA_VERSION, TaskControlRequest, TaskRunRequest, TaskSpec,
    TaskSpecError, TaskStageSpec, ToolCallRequest,
};
pub use service::{AgentService, RuntimeConfig};
pub use session::{Exchange, SessionStore};
pub use sidecar::{BatchContinuation, GateDecision, SidecarClient, SidecarState};
pub use task_ledger::{
    LEDGER_SCHEMA_VERSION, LedgerEvent, StageStatus, TaskLedger, TaskRecord, TaskStageRecord,
    TaskStatus,
};
