//! Production-oriented Rust control plane for RWKV Agent.
//!
//! CUDA inference and retrieval remain behind narrow HTTP data-plane APIs;
//! this crate owns recurrent-State lifecycle, routing, sessions and tool loops.

mod command;
mod data_client;
mod prompt;
mod research;
mod service;
mod session;
mod sidecar;
mod task_ledger;

pub use command::{CommandPolicy, SandboxedCommand};
pub use data_client::DataPlaneClient;
pub use research::ResearchRunner;
pub use rwkv_agent_core::{TASK_SPEC_SCHEMA_VERSION, TaskSpec, TaskSpecError, TaskStageSpec};
pub use service::{AgentService, RuntimeConfig};
pub use session::{Exchange, SessionStore};
pub use sidecar::{BatchContinuation, GateDecision, SidecarClient, SidecarState};
pub use task_ledger::{
    LedgerEvent, StageStatus, TaskLedger, TaskRecord, TaskStageRecord, TaskStatus,
};
