//! Rust-only contract and benchmark runtime for long-lived inference sessions.
//!
//! Durable Agent state is the source of truth.  Provider handles and snapshots
//! are disposable accelerator checkpoints and never expose tensors to Agents.

mod contract;
mod http_rwkv;
mod in_memory;
mod provider;
mod qwen_transcript;
mod runner;
mod trace;

pub use contract::{
    CONTRACT_VERSION, CheckpointRef, ContinueRequest, ContinueResult, CreateRequest, ModelRef,
    Placement, ProviderMode, ReleaseOutcome, RestoreRequest, SessionDescription, SessionHandle,
    SnapshotRequest, StateContractError,
};
pub use http_rwkv::RwkvHttpProvider;
pub use in_memory::InMemoryConformanceProvider;
pub use provider::{ProviderFuture, StatefulInferenceProvider};
pub use qwen_transcript::QwenTranscriptProvider;
pub use runner::{
    LongLivedResult, PlacementPolicy, WorkloadConfig, WorkloadError, macro_and_worst_category,
    run_long_lived_workload,
};
pub use trace::{
    AntiOverfitManifest, DuplicateReport, EVENT_SCHEMA_VERSION, EventPayload, EventTrace,
    MANIFEST_SCHEMA_VERSION, Split, TraceGeneration, TraceManifest, build_trace, duplicate_report,
    load_trace_jsonl, sha256_file, write_trace_jsonl,
};
