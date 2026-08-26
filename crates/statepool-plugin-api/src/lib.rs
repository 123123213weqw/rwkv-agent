//! Versioned wire contracts for the optional out-of-process StatePool plugin.
//!
//! This crate contains data only. It does not open sockets, contact cloud
//! services or change the normal `rwkv-agent` execution path.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const PLUGIN_CONTRACT_VERSION: &str = "statepool-plugin.v1";
pub const PLAN_REQUEST_CONTRACT_VERSION: &str = "statepool-plan-request.v1";
pub const EXECUTION_PLAN_CONTRACT_VERSION: &str = "statepool-execution-plan.v1";
pub const WORKER_CAPABILITY_CONTRACT_VERSION: &str = "statepool-worker-capability.v1";
pub const STATE_REFERENCE_CONTRACT_VERSION: &str = "statepool-state-reference.v1";
pub const USAGE_RECORD_CONTRACT_VERSION: &str = "statepool-usage-record.v1";
pub const ACQUIRE_LEASE_REQUEST_CONTRACT_VERSION: &str = "statepool-acquire-lease-request.v1";
pub const RENEW_LEASE_REQUEST_CONTRACT_VERSION: &str = "statepool-renew-lease-request.v1";
pub const RELEASE_LEASE_REQUEST_CONTRACT_VERSION: &str = "statepool-release-lease-request.v1";
pub const LEASE_CONTRACT_VERSION: &str = "statepool-lease.v1";
pub const SNAPSHOT_REQUEST_CONTRACT_VERSION: &str = "statepool-snapshot-request.v1";
pub const RESTORE_REQUEST_CONTRACT_VERSION: &str = "statepool-restore-request.v1";
pub const RESTORE_RESPONSE_CONTRACT_VERSION: &str = "statepool-restore-response.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyClass {
    LocalOnly,
    Hybrid,
    CloudAllowed,
}

impl std::fmt::Display for PrivacyClass {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalOnly => formatter.write_str("local_only"),
            Self::Hybrid => formatter.write_str("hybrid"),
            Self::CloudAllowed => formatter.write_str("cloud_allowed"),
        }
    }
}

impl std::str::FromStr for PrivacyClass {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "local_only" | "local-only" => Ok(Self::LocalOnly),
            "hybrid" => Ok(Self::Hybrid),
            "cloud_allowed" | "cloud-allowed" => Ok(Self::CloudAllowed),
            _ => Err("cloud plugin privacy must be local_only, hybrid or cloud_allowed".into()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelRef {
    pub model_id: String,
    pub revision: String,
    pub tokenizer: String,
    pub state_abi: String,
}

impl ModelRef {
    pub fn validate(&self) -> Result<(), ContractError> {
        for (name, value) in [
            ("model_id", &self.model_id),
            ("revision", &self.revision),
            ("tokenizer", &self.tokenizer),
            ("state_abi", &self.state_abi),
        ] {
            if value.trim().is_empty() {
                return Err(ContractError::Invalid(format!(
                    "model_ref.{name} must not be empty"
                )));
            }
        }
        Ok(())
    }

    /// Raw recurrent State is portable only across exact identities.
    pub fn exact_compatible(&self, other: &Self) -> bool {
        self == other
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Money {
    pub currency: String,
    pub amount: f64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HandshakeRequest {
    pub contract_version: String,
    pub host: String,
    pub host_version: String,
    pub required_capabilities: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub contract_version: String,
    pub plugin: String,
    pub plugin_version: String,
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerZone {
    Local,
    Edge,
    Cloud,
}

impl std::fmt::Display for WorkerZone {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => formatter.write_str("local"),
            Self::Edge => formatter.write_str("edge"),
            Self::Cloud => formatter.write_str("cloud"),
        }
    }
}

impl std::str::FromStr for WorkerZone {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "local" => Ok(Self::Local),
            "edge" => Ok(Self::Edge),
            "cloud" => Ok(Self::Cloud),
            _ => Err("Worker zone must be local, edge or cloud".into()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerLifecycle {
    Starting,
    Ready,
    Draining,
    Offline,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkerDevice {
    pub vendor: String,
    pub model: String,
    #[serde(default)]
    pub runtime: String,
    pub memory_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkerCapacity {
    pub state_slots: u32,
    pub free_state_slots: u32,
    pub max_batch: u32,
    pub queue_depth: u32,
    pub running_requests: u32,
    /// `None` means the Worker has not proved persistence readiness and must
    /// not be declared safe to stop by the control plane.
    #[serde(default)]
    pub unpersisted_state_slots: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkerPrice {
    pub currency: String,
    pub per_gpu_hour: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkerCapability {
    pub contract_version: String,
    pub worker_id: String,
    pub zone: WorkerZone,
    pub endpoint: String,
    pub lifecycle: WorkerLifecycle,
    pub models: Vec<ModelRef>,
    pub device: WorkerDevice,
    pub capacity: WorkerCapacity,
    #[serde(default)]
    pub price: Option<WorkerPrice>,
    #[serde(default)]
    pub labels: std::collections::BTreeMap<String, String>,
    pub reported_at_ms: u64,
}

impl WorkerCapability {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.contract_version != WORKER_CAPABILITY_CONTRACT_VERSION {
            return Err(ContractError::Version);
        }
        if self.worker_id.trim().is_empty() || self.endpoint.trim().is_empty() {
            return Err(ContractError::Invalid(
                "worker_id and endpoint must not be empty".into(),
            ));
        }
        if self.models.is_empty()
            || self.capacity.state_slots == 0
            || self.capacity.max_batch == 0
            || self.capacity.free_state_slots > self.capacity.state_slots
            || self
                .capacity
                .unpersisted_state_slots
                .is_some_and(|value| value > self.capacity.state_slots)
        {
            return Err(ContractError::Invalid(
                "Worker model and capacity values are invalid".into(),
            ));
        }
        for model in &self.models {
            model.validate()?;
        }
        Ok(())
    }

    pub fn supports(&self, model: &ModelRef) -> bool {
        self.lifecycle == WorkerLifecycle::Ready
            && self.capacity.free_state_slots > 0
            && self
                .models
                .iter()
                .any(|candidate| candidate.exact_compatible(model))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatePlacement {
    Hot,
    Warm,
    Cold,
    Dropped,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StateReference {
    pub contract_version: String,
    pub state_id: String,
    pub session_id: String,
    pub owner_id: String,
    pub version: u64,
    #[serde(default)]
    pub fencing_token: Option<u64>,
    pub provider_mode: String,
    pub model_ref: ModelRef,
    pub placement: StatePlacement,
    #[serde(default)]
    pub worker_id: Option<String>,
    #[serde(default)]
    pub object_uri: Option<String>,
    pub checksum: String,
    pub size_bytes: u64,
    pub atomic: bool,
    pub created_at_ms: u64,
    pub last_active_at_ms: u64,
    #[serde(default)]
    pub encryption: Option<Value>,
}

impl StateReference {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.contract_version != STATE_REFERENCE_CONTRACT_VERSION {
            return Err(ContractError::Version);
        }
        if self.state_id.trim().is_empty()
            || self.session_id.trim().is_empty()
            || self.owner_id.trim().is_empty()
            || self.provider_mode.trim().is_empty()
            || self.checksum.trim().is_empty()
            || !self.atomic
        {
            return Err(ContractError::Invalid(
                "State identity, checksum and atomic marker are required".into(),
            ));
        }
        if !self.checksum.starts_with("sha256:") || self.checksum.len() != 71 {
            return Err(ContractError::Invalid(
                "State checksum must use sha256:<64 lowercase hex characters>".into(),
            ));
        }
        self.model_ref.validate()?;
        Ok(())
    }

    pub fn exact_restore_compatible(&self, expected: &ModelRef) -> bool {
        self.atomic && self.model_ref.exact_compatible(expected)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AcquireLeaseRequest {
    pub contract_version: String,
    pub session_id: String,
    pub owner_id: String,
    pub holder_id: String,
    pub expected_state_version: u64,
    pub ttl_ms: u64,
}

impl AcquireLeaseRequest {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.contract_version != ACQUIRE_LEASE_REQUEST_CONTRACT_VERSION {
            return Err(ContractError::Version);
        }
        if self.session_id.trim().is_empty()
            || self.owner_id.trim().is_empty()
            || self.holder_id.trim().is_empty()
            || self.ttl_ms < 1_000
        {
            return Err(ContractError::Invalid(
                "Lease identities and a ttl_ms of at least 1000 are required".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Lease {
    pub contract_version: String,
    pub lease_id: String,
    pub session_id: String,
    pub owner_id: String,
    pub holder_id: String,
    pub fencing_token: u64,
    pub expected_state_version: u64,
    pub expires_at_ms: u64,
}

impl Lease {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.contract_version != LEASE_CONTRACT_VERSION {
            return Err(ContractError::Version);
        }
        if self.lease_id.trim().is_empty()
            || self.session_id.trim().is_empty()
            || self.owner_id.trim().is_empty()
            || self.holder_id.trim().is_empty()
            || self.fencing_token == 0
        {
            return Err(ContractError::Invalid(
                "Lease identity and positive fencing_token are required".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RenewLeaseRequest {
    pub contract_version: String,
    pub lease: Lease,
    pub ttl_ms: u64,
}

impl RenewLeaseRequest {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.contract_version != RENEW_LEASE_REQUEST_CONTRACT_VERSION || self.ttl_ms < 1_000 {
            return Err(ContractError::Invalid(
                "Renew contract and a ttl_ms of at least 1000 are required".into(),
            ));
        }
        self.lease.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReleaseLeaseRequest {
    pub contract_version: String,
    pub lease: Lease,
}

impl ReleaseLeaseRequest {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.contract_version != RELEASE_LEASE_REQUEST_CONTRACT_VERSION {
            return Err(ContractError::Version);
        }
        self.lease.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SnapshotStateRequest {
    pub contract_version: String,
    pub provider_mode: String,
    pub model_ref: ModelRef,
    pub target_tier: StatePlacement,
    pub lease: Lease,
    pub expected_state_version: u64,
    pub payload_base64: String,
    #[serde(default)]
    pub expected_checksum: Option<String>,
}

impl SnapshotStateRequest {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.contract_version != SNAPSHOT_REQUEST_CONTRACT_VERSION {
            return Err(ContractError::Version);
        }
        if self.provider_mode.trim().is_empty() || self.payload_base64.is_empty() {
            return Err(ContractError::Invalid(
                "provider_mode and state payload are required".into(),
            ));
        }
        if !matches!(
            self.target_tier,
            StatePlacement::Warm | StatePlacement::Cold
        ) {
            return Err(ContractError::Invalid(
                "Snapshot target_tier must be warm or cold".into(),
            ));
        }
        self.model_ref.validate()?;
        self.lease.validate()?;
        if self.expected_state_version != self.lease.expected_state_version {
            return Err(ContractError::Invalid(
                "Snapshot and Lease expected State versions differ".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RestoreStateRequest {
    pub contract_version: String,
    pub state_ref: StateReference,
    pub expected_model_ref: ModelRef,
    pub target_worker_id: String,
    pub lease: Lease,
}

impl RestoreStateRequest {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.contract_version != RESTORE_REQUEST_CONTRACT_VERSION {
            return Err(ContractError::Version);
        }
        if self.target_worker_id.trim().is_empty() {
            return Err(ContractError::Invalid(
                "Restore target_worker_id is required".into(),
            ));
        }
        self.state_ref.validate()?;
        self.expected_model_ref.validate()?;
        self.lease.validate()?;
        if !self
            .state_ref
            .exact_restore_compatible(&self.expected_model_ref)
        {
            return Err(ContractError::Invalid(
                "Raw State restore requires exact model identity and an atomic State".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RestoreStateResponse {
    pub contract_version: String,
    pub state_ref: StateReference,
    pub payload_base64: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanRequest {
    pub contract_version: String,
    pub request_id: String,
    pub session_id: String,
    pub owner_id: String,
    pub model_ref: ModelRef,
    pub privacy: PrivacyClass,
    pub latency_slo_ms: u64,
    #[serde(default)]
    pub max_cost: Option<Money>,
    #[serde(default)]
    pub preferred_zone: Option<WorkerZone>,
    #[serde(default)]
    pub state_ref: Option<StateReference>,
    pub estimated_input_tokens: u64,
    pub estimated_output_tokens: u64,
}

impl PlanRequest {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.contract_version != PLAN_REQUEST_CONTRACT_VERSION {
            return Err(ContractError::Version);
        }
        if self.request_id.trim().is_empty()
            || self.session_id.trim().is_empty()
            || self.owner_id.trim().is_empty()
            || self.latency_slo_ms == 0
            || self.estimated_output_tokens == 0
        {
            return Err(ContractError::Invalid(
                "planning identity, SLO and output budget must be positive".into(),
            ));
        }
        self.model_ref.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub contract_version: String,
    pub decision_id: String,
    pub request_id: String,
    pub mode: String,
    #[serde(default)]
    pub worker_id: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    pub state_action: String,
    pub reason_code: String,
    pub lease_required: bool,
    #[serde(default)]
    pub estimated_queue_ms: Option<f64>,
    #[serde(default)]
    pub estimated_restore_ms: Option<f64>,
    #[serde(default)]
    pub estimated_cost: Option<Money>,
    pub fallback: String,
}

impl ExecutionPlan {
    pub fn local(request_id: String, reason_code: &str) -> Self {
        Self {
            contract_version: EXECUTION_PLAN_CONTRACT_VERSION.into(),
            decision_id: format!("local-{request_id}"),
            request_id,
            mode: "local".into(),
            worker_id: None,
            endpoint: None,
            state_action: "none".into(),
            reason_code: reason_code.into(),
            lease_required: false,
            estimated_queue_ms: None,
            estimated_restore_ms: None,
            estimated_cost: None,
            fallback: "local".into(),
        }
    }

    pub fn remote_endpoint(&self) -> Option<&str> {
        (self.mode == "remote")
            .then_some(self.endpoint.as_deref())
            .flatten()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UsageMetrics {
    pub elapsed_ms: f64,
    #[serde(default)]
    pub queue_ms: Option<f64>,
    #[serde(default)]
    pub restore_ms: Option<f64>,
    #[serde(default)]
    pub snapshot_ms: Option<f64>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub prefill_tokens_avoided: u64,
    pub gpu_seconds: f64,
    pub state_bytes_read: u64,
    pub state_bytes_written: u64,
    #[serde(default)]
    pub estimated_cost: Option<Money>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UsageRecord {
    pub contract_version: String,
    pub record_id: String,
    pub request_id: String,
    pub session_id: String,
    pub owner_id: String,
    pub worker_id: String,
    pub zone: WorkerZone,
    pub operation: String,
    pub outcome: String,
    #[serde(default)]
    pub state_tier_before: Option<StatePlacement>,
    #[serde(default)]
    pub state_tier_after: Option<StatePlacement>,
    pub started_at_ms: u64,
    pub finished_at_ms: u64,
    pub metrics: UsageMetrics,
    #[serde(default)]
    pub error_code: Option<String>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ContractError {
    #[error("contract version mismatch")]
    Version,
    #[error("invalid contract: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(revision: &str) -> ModelRef {
        ModelRef {
            model_id: "rwkv7".into(),
            revision: revision.into(),
            tokenizer: "world".into(),
            state_abi: "rwkv7-state-v1".into(),
        }
    }

    #[test]
    fn exact_restore_requires_the_complete_model_identity() {
        assert!(model("a").exact_compatible(&model("a")));
        assert!(!model("a").exact_compatible(&model("b")));
    }

    #[test]
    fn ready_worker_requires_capacity_and_exact_model() {
        let worker = WorkerCapability {
            contract_version: WORKER_CAPABILITY_CONTRACT_VERSION.into(),
            worker_id: "worker".into(),
            zone: WorkerZone::Cloud,
            endpoint: "http://worker".into(),
            lifecycle: WorkerLifecycle::Ready,
            models: vec![model("a")],
            device: WorkerDevice {
                vendor: "nvidia".into(),
                model: "v100".into(),
                runtime: "cuda".into(),
                memory_bytes: 32,
            },
            capacity: WorkerCapacity {
                state_slots: 8,
                free_state_slots: 8,
                max_batch: 8,
                queue_depth: 0,
                running_requests: 0,
                unpersisted_state_slots: Some(0),
            },
            price: None,
            labels: Default::default(),
            reported_at_ms: 1,
        };
        assert!(worker.supports(&model("a")));
        assert!(!worker.supports(&model("b")));
    }

    #[test]
    fn checked_in_json_schema_examples_deserialize_into_wire_types() {
        let handshake_request: HandshakeRequest = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/handshake-request.json"
        ))
        .unwrap();
        let handshake_response: HandshakeResponse = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/handshake-response.json"
        ))
        .unwrap();
        let worker: WorkerCapability = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/worker.json"
        ))
        .unwrap();
        let state: StateReference = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/state-reference.json"
        ))
        .unwrap();
        let request: PlanRequest = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/plan-request.json"
        ))
        .unwrap();
        let plan: ExecutionPlan = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/execution-plan.json"
        ))
        .unwrap();
        let usage: UsageRecord = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/usage-record.json"
        ))
        .unwrap();
        let acquire: AcquireLeaseRequest = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/acquire-lease-request.json"
        ))
        .unwrap();
        let lease: Lease = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/lease.json"
        ))
        .unwrap();
        let renew: RenewLeaseRequest = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/renew-lease-request.json"
        ))
        .unwrap();
        let release: ReleaseLeaseRequest = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/release-lease-request.json"
        ))
        .unwrap();
        let snapshot: SnapshotStateRequest = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/snapshot-request.json"
        ))
        .unwrap();
        let restore: RestoreStateRequest = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/restore-request.json"
        ))
        .unwrap();
        let restore_response: RestoreStateResponse = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/restore-response.json"
        ))
        .unwrap();

        assert_eq!(handshake_request.contract_version, PLUGIN_CONTRACT_VERSION);
        assert_eq!(handshake_response.plugin, "statepool-cloud");
        worker.validate().unwrap();
        assert!(state.exact_restore_compatible(&request.model_ref));
        request.validate().unwrap();
        assert_eq!(plan.contract_version, EXECUTION_PLAN_CONTRACT_VERSION);
        assert_eq!(usage.contract_version, USAGE_RECORD_CONTRACT_VERSION);
        acquire.validate().unwrap();
        lease.validate().unwrap();
        renew.validate().unwrap();
        release.validate().unwrap();
        snapshot.validate().unwrap();
        restore.validate().unwrap();
        assert_eq!(
            restore_response.contract_version,
            RESTORE_RESPONSE_CONTRACT_VERSION
        );
    }
}
