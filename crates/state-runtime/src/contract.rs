use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const CONTRACT_VERSION: &str = "stateful-inference-session.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
    Gpu,
    Cpu,
    Disk,
    Dropped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderMode {
    RwkvRecurrent,
    QwenNativeKv,
    QwenTranscriptReprefill,
    ContractTest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelRef {
    pub model_id: String,
    pub revision: String,
    pub tokenizer: String,
    pub state_abi: String,
}

impl ModelRef {
    pub fn validate(&self) -> Result<(), StateContractError> {
        for (name, value) in [
            ("model_id", &self.model_id),
            ("revision", &self.revision),
            ("tokenizer", &self.tokenizer),
            ("state_abi", &self.state_abi),
        ] {
            if value.trim().is_empty() {
                return Err(StateContractError::InvalidRequest(format!(
                    "{name} must not be empty"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SessionHandle {
    pub contract_version: String,
    pub session_id: String,
    pub owner_id: String,
    pub provider_mode: ProviderMode,
    pub model_ref: ModelRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CheckpointRef {
    pub contract_version: String,
    pub checkpoint_id: String,
    pub owner_id: String,
    pub provider_mode: ProviderMode,
    pub model_ref: ModelRef,
    pub placement: Placement,
    pub checksum: String,
    pub size_bytes: u64,
    pub created_at_ms: u64,
    pub last_active_at_ms: u64,
    pub atomic: bool,
}

impl CheckpointRef {
    pub fn validate(&self) -> Result<(), StateContractError> {
        if self.contract_version != CONTRACT_VERSION {
            return Err(StateContractError::ContractVersion);
        }
        if !matches!(self.placement, Placement::Cpu | Placement::Disk) {
            return Err(StateContractError::UnsupportedPlacement(self.placement));
        }
        let checksum = self.checksum.strip_prefix("sha256:").unwrap_or_default();
        if checksum.len() != 64
            || !checksum
                .bytes()
                .all(|value| value.is_ascii_digit() || (b'a'..=b'f').contains(&value))
        {
            return Err(StateContractError::InvalidRequest(
                "checksum must be sha256:<64 lowercase hex>".into(),
            ));
        }
        if !self.atomic {
            return Err(StateContractError::NonAtomicCheckpoint);
        }
        self.model_ref.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CreateRequest {
    pub owner_id: String,
    pub durable_session_ref: String,
    pub model_ref: ModelRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ContinueRequest {
    pub owner_id: String,
    pub session_handle: SessionHandle,
    pub input: String,
    pub token_budget: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContinueResult {
    pub session_handle: SessionHandle,
    pub text: String,
    pub output_tokens: u32,
    pub seen_tokens: u64,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SnapshotRequest {
    pub owner_id: String,
    pub session_handle: SessionHandle,
    pub target_tier: Placement,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RestoreRequest {
    pub owner_id: String,
    pub checkpoint_ref: CheckpointRef,
    pub expected_model_ref: ModelRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SessionDescription {
    pub session_handle: SessionHandle,
    pub durable_session_ref: String,
    pub placement: Placement,
    pub created_at_ms: u64,
    pub last_active_at_ms: u64,
    pub state_bytes: Option<u64>,
    pub seen_tokens: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReleaseOutcome {
    pub released: bool,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum StateContractError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("unsupported contract version")]
    ContractVersion,
    #[error("session owner mismatch")]
    OwnerMismatch,
    #[error("unknown or released session handle")]
    StaleHandle,
    #[error("unknown checkpoint")]
    StaleCheckpoint,
    #[error("checkpoint model reference mismatch")]
    ModelMismatch,
    #[error("checkpoint provider mismatch")]
    ProviderMismatch,
    #[error("checkpoint checksum or size mismatch")]
    ChecksumFailure,
    #[error("snapshot placement is unsupported: {0:?}")]
    UnsupportedPlacement(Placement),
    #[error("non-atomic checkpoint cannot be restored")]
    NonAtomicCheckpoint,
    #[error("provider operation is unsupported: {0}")]
    Unsupported(String),
    #[error("provider failure: {0}")]
    Provider(String),
    #[error("operation cancelled")]
    Cancelled,
}

impl StateContractError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => "invalid_request",
            Self::ContractVersion => "contract_version",
            Self::OwnerMismatch => "owner_mismatch",
            Self::StaleHandle => "stale_handle",
            Self::StaleCheckpoint => "stale_checkpoint",
            Self::ModelMismatch => "model_mismatch",
            Self::ProviderMismatch => "provider_mismatch",
            Self::ChecksumFailure => "checksum_failure",
            Self::UnsupportedPlacement(_) => "unsupported_placement",
            Self::NonAtomicCheckpoint => "non_atomic_checkpoint",
            Self::Unsupported(_) => "unsupported",
            Self::Provider(_) => "provider_failure",
            Self::Cancelled => "cancelled",
        }
    }
}
