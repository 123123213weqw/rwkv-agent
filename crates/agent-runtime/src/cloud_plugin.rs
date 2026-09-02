use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use tokio::sync::RwLock;

use rwkv_statepool_plugin_api::{
    ACQUIRE_LEASE_REQUEST_CONTRACT_VERSION, AcquireLeaseRequest, EXECUTION_PLAN_CONTRACT_VERSION,
    HandshakeRequest, HandshakeResponse, LEASE_CONTRACT_VERSION, PLAN_REQUEST_CONTRACT_VERSION,
    PLUGIN_CONTRACT_VERSION, PlanRequest, RELEASE_LEASE_REQUEST_CONTRACT_VERSION,
    RENEW_LEASE_REQUEST_CONTRACT_VERSION, RESTORE_REQUEST_CONTRACT_VERSION,
    RESTORE_RESPONSE_CONTRACT_VERSION, ReleaseLeaseRequest, RenewLeaseRequest, RestoreStateRequest,
    SNAPSHOT_REQUEST_CONTRACT_VERSION, SnapshotStateRequest,
};
pub use rwkv_statepool_plugin_api::{
    ExecutionPlan, Lease as CloudLease, ModelRef as CloudModelRef, Money as CloudMoney,
    PrivacyClass, RestoreStateResponse as CloudRestoreStateResponse,
    StatePlacement as CloudStatePlacement, StateReference as CloudStateReference,
    USAGE_RECORD_CONTRACT_VERSION as CLOUD_USAGE_RECORD_CONTRACT_VERSION,
    UsageMetrics as CloudUsageMetrics, UsageRecord as CloudUsageRecord, WorkerZone,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloudPluginFallback {
    Local,
    FailClosed,
}

impl std::fmt::Display for CloudPluginFallback {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => formatter.write_str("local"),
            Self::FailClosed => formatter.write_str("fail_closed"),
        }
    }
}

impl std::str::FromStr for CloudPluginFallback {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "local" => Ok(Self::Local),
            "fail_closed" | "fail-closed" => Ok(Self::FailClosed),
            _ => Err("cloud plugin fallback must be local or fail_closed".into()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CloudPluginConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub fallback: CloudPluginFallback,
    pub default_privacy: PrivacyClass,
    pub latency_slo: Duration,
    pub preferred_zone: Option<WorkerZone>,
    pub model_ref: Option<CloudModelRef>,
    pub required_capabilities: Vec<String>,
    pub state_lifecycle: bool,
    pub state_target_tier: CloudStatePlacement,
    pub lease_ttl: Duration,
    pub lifecycle_timeout: Duration,
}

impl Default for CloudPluginConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: "http://127.0.0.1:8130".into(),
            connect_timeout: Duration::from_millis(500),
            request_timeout: Duration::from_secs(2),
            fallback: CloudPluginFallback::Local,
            default_privacy: PrivacyClass::LocalOnly,
            latency_slo: Duration::from_secs(5),
            preferred_zone: None,
            model_ref: None,
            required_capabilities: vec!["placement".into()],
            state_lifecycle: false,
            state_target_tier: CloudStatePlacement::Cold,
            lease_ttl: Duration::from_secs(120),
            lifecycle_timeout: Duration::from_secs(180),
        }
    }
}

#[derive(Clone, Debug)]
struct PluginState {
    status: &'static str,
    error: String,
    plugin_version: String,
    capabilities: Vec<String>,
}

impl PluginState {
    fn disabled() -> Self {
        Self {
            status: "disabled",
            error: String::new(),
            plugin_version: String::new(),
            capabilities: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct CloudPluginClient {
    config: Arc<CloudPluginConfig>,
    http: Option<Client>,
    state: Arc<RwLock<PluginState>>,
    request_counter: Arc<AtomicU64>,
}

impl CloudPluginClient {
    pub fn new(config: CloudPluginConfig) -> Result<Self, String> {
        let endpoint = config.endpoint.trim().trim_end_matches('/').to_string();
        if config.enabled {
            if endpoint.is_empty() {
                return Err("cloud plugin endpoint must not be empty when enabled".into());
            }
            config
                .model_ref
                .as_ref()
                .ok_or_else(|| "cloud plugin model_ref is required when enabled".to_string())?
                .validate()
                .map_err(|error| error.to_string())?;
            if config.latency_slo.is_zero() {
                return Err("cloud plugin latency_slo must be positive".into());
            }
            if config.required_capabilities.is_empty()
                || config
                    .required_capabilities
                    .iter()
                    .any(|value| value.trim().is_empty())
            {
                return Err("cloud plugin required_capabilities must not be empty".into());
            }
            if config.state_lifecycle
                && (config.lease_ttl < Duration::from_secs(1)
                    || config.lifecycle_timeout.is_zero()
                    || !matches!(
                        config.state_target_tier,
                        CloudStatePlacement::Warm | CloudStatePlacement::Cold
                    ))
            {
                return Err(
                    "cloud State lifecycle requires a positive Lease TTL, lifecycle timeout and warm/cold target".into(),
                );
            }
        }
        let http = if config.enabled {
            Some(
                Client::builder()
                    .connect_timeout(config.connect_timeout)
                    .timeout(config.request_timeout)
                    .build()
                    .map_err(|error| format!("build cloud plugin client: {error}"))?,
            )
        } else {
            None
        };
        let mut config = config;
        config.endpoint = endpoint;
        Ok(Self {
            config: Arc::new(config),
            http,
            state: Arc::new(RwLock::new(PluginState::disabled())),
            request_counter: Arc::new(AtomicU64::new(1)),
        })
    }

    pub async fn initialize(&self, host_version: &str) -> Result<(), String> {
        if !self.config.enabled {
            return Ok(());
        }
        let result = self.handshake(host_version).await;
        if let Err(error) = &result {
            let mut state = self.state.write().await;
            state.status = if self.config.fallback == CloudPluginFallback::Local {
                "degraded"
            } else {
                "unavailable"
            };
            state.error = error.clone();
        }
        match (result, self.config.fallback) {
            (Ok(()), _) => Ok(()),
            (Err(_), CloudPluginFallback::Local) => Ok(()),
            (Err(error), CloudPluginFallback::FailClosed) => Err(error),
        }
    }

    async fn handshake(&self, host_version: &str) -> Result<(), String> {
        let required_capabilities = self.effective_required_capabilities();
        let response = self
            .http
            .as_ref()
            .ok_or_else(|| "cloud plugin is disabled".to_string())?
            .post(format!("{}/plugin/v1/handshake", self.config.endpoint))
            .json(&HandshakeRequest {
                contract_version: PLUGIN_CONTRACT_VERSION.into(),
                host: "rwkv-agent".into(),
                host_version: host_version.into(),
                required_capabilities: required_capabilities.clone(),
            })
            .send()
            .await
            .map_err(|error| format!("cloud plugin handshake unavailable: {error}"))?;
        let status = response.status();
        let body = response
            .json::<HandshakeResponse>()
            .await
            .map_err(|error| format!("cloud plugin handshake invalid JSON: {error}"))?;
        if !status.is_success() {
            return Err(format!("cloud plugin handshake HTTP {status}"));
        }
        if body.contract_version != PLUGIN_CONTRACT_VERSION || body.plugin != "statepool-cloud" {
            return Err("cloud plugin handshake contract or identity mismatch".into());
        }
        let capabilities = body.capabilities.iter().cloned().collect::<BTreeSet<_>>();
        let missing = required_capabilities
            .iter()
            .filter(|value| !capabilities.contains(*value))
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(format!(
                "cloud plugin is missing required capabilities: {}",
                missing.join(",")
            ));
        }
        *self.state.write().await = PluginState {
            status: "ready",
            error: String::new(),
            plugin_version: body.plugin_version,
            capabilities: body.capabilities,
        };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn plan(
        &self,
        session_id: &str,
        owner_id: &str,
        estimated_input_tokens: u64,
        estimated_output_tokens: u64,
    ) -> Result<ExecutionPlan, String> {
        self.plan_with_state(
            session_id,
            owner_id,
            estimated_input_tokens,
            estimated_output_tokens,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn plan_with_state(
        &self,
        session_id: &str,
        owner_id: &str,
        estimated_input_tokens: u64,
        estimated_output_tokens: u64,
        state_ref: Option<CloudStateReference>,
    ) -> Result<ExecutionPlan, String> {
        let has_committed_state = state_ref.is_some();
        let request_id = format!(
            "plugin-{}-{:016x}",
            sanitize_id(session_id),
            self.request_counter.fetch_add(1, Ordering::Relaxed)
        );
        if !self.config.enabled {
            return if has_committed_state {
                Err("cloud plugin is disabled; committed remote State exists, so local fallback is forbidden".into())
            } else {
                Ok(ExecutionPlan::local(request_id, "plugin_disabled"))
            };
        }
        if self.state.read().await.status != "ready" {
            return self.fallback_or_error_after_plan(
                request_id,
                "cloud plugin is not ready; no remote operation was started",
                has_committed_state,
            );
        }
        let model_ref = self
            .config
            .model_ref
            .clone()
            .ok_or_else(|| "cloud plugin model_ref is unavailable".to_string())?;
        let request = PlanRequest {
            contract_version: PLAN_REQUEST_CONTRACT_VERSION.into(),
            request_id: request_id.clone(),
            session_id: session_id.into(),
            owner_id: owner_id.into(),
            model_ref,
            privacy: self.config.default_privacy,
            latency_slo_ms: self
                .config
                .latency_slo
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
            max_cost: None,
            preferred_zone: self.config.preferred_zone.clone(),
            affinity_worker_id: None,
            state_ref,
            estimated_input_tokens,
            estimated_output_tokens,
        };
        if let Err(error) = request.validate() {
            return self.fallback_or_error_after_plan(
                request_id,
                &format!("cloud plugin planning request is invalid: {error}"),
                has_committed_state,
            );
        }
        let response = match self
            .http
            .as_ref()
            .expect("enabled cloud plugin has an HTTP client")
            .post(format!("{}/plugin/v1/plan", self.config.endpoint))
            .json(&request)
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                return self.fallback_or_error_after_plan(
                    request_id,
                    &format!("cloud plugin planning unavailable: {error}"),
                    has_committed_state,
                );
            }
        };
        let status = response.status();
        let plan = match response.json::<ExecutionPlan>().await {
            Ok(plan) => plan,
            Err(error) => {
                return self.fallback_or_error_after_plan(
                    request_id,
                    &format!("cloud plugin planning invalid JSON: {error}"),
                    has_committed_state,
                );
            }
        };
        if !status.is_success() {
            return self.fallback_or_error_after_plan(
                request_id,
                &format!("cloud plugin planning HTTP {status}"),
                has_committed_state,
            );
        }
        if let Err(error) = self.validate_plan(&plan, &request_id) {
            return self.fallback_or_error_after_plan(request_id, &error, has_committed_state);
        }
        if matches!(plan.mode.as_str(), "defer" | "reject")
            && self.config.fallback == CloudPluginFallback::Local
            && !has_committed_state
        {
            return Ok(ExecutionPlan::local(request_id, &plan.reason_code));
        }
        Ok(plan)
    }

    pub fn state_lifecycle_enabled(&self) -> bool {
        self.config.enabled && self.config.state_lifecycle
    }

    pub async fn state_lifecycle_ready(&self) -> bool {
        self.state_lifecycle_enabled() && self.state.read().await.status == "ready"
    }

    pub async fn plugin_ready(&self) -> bool {
        self.config.enabled && self.state.read().await.status == "ready"
    }

    pub async fn finops_ready(&self) -> bool {
        self.config.enabled && {
            let state = self.state.read().await;
            state.status == "ready"
                && state
                    .capabilities
                    .iter()
                    .any(|capability| capability == "finops")
        }
    }

    pub fn state_target_tier(&self) -> CloudStatePlacement {
        self.config.state_target_tier.clone()
    }

    pub fn state_model_ref(&self) -> Result<CloudModelRef, String> {
        self.lifecycle_model_ref()
    }

    pub async fn acquire_lease(
        &self,
        session_id: &str,
        owner_id: &str,
        holder_id: &str,
        expected_state_version: u64,
    ) -> Result<CloudLease, String> {
        let ttl_ms = self
            .config
            .lease_ttl
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        let request = AcquireLeaseRequest {
            contract_version: ACQUIRE_LEASE_REQUEST_CONTRACT_VERSION.into(),
            session_id: session_id.into(),
            owner_id: owner_id.into(),
            holder_id: holder_id.into(),
            expected_state_version,
            ttl_ms,
        };
        request.validate().map_err(|error| error.to_string())?;
        let lease = self
            .lifecycle_post::<_, CloudLease>("/plugin/v1/leases/acquire", &request)
            .await?;
        lease.validate().map_err(|error| error.to_string())?;
        if lease.contract_version != LEASE_CONTRACT_VERSION
            || lease.session_id != session_id
            || lease.owner_id != owner_id
            || lease.holder_id != holder_id
            || lease.expected_state_version != expected_state_version
        {
            return Err("cloud plugin returned a mismatched Lease".into());
        }
        Ok(lease)
    }

    pub async fn renew_lease(&self, lease: &CloudLease) -> Result<CloudLease, String> {
        let ttl_ms = self
            .config
            .lease_ttl
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        let request = RenewLeaseRequest {
            contract_version: RENEW_LEASE_REQUEST_CONTRACT_VERSION.into(),
            lease: lease.clone(),
            ttl_ms,
        };
        request.validate().map_err(|error| error.to_string())?;
        let renewed = self
            .lifecycle_post::<_, CloudLease>("/plugin/v1/leases/renew", &request)
            .await?;
        renewed.validate().map_err(|error| error.to_string())?;
        if renewed.lease_id != lease.lease_id
            || renewed.fencing_token != lease.fencing_token
            || renewed.expected_state_version != lease.expected_state_version
            || renewed.session_id != lease.session_id
            || renewed.owner_id != lease.owner_id
            || renewed.holder_id != lease.holder_id
        {
            return Err("cloud plugin renewed a different Lease".into());
        }
        Ok(renewed)
    }

    pub async fn release_lease(&self, lease: &CloudLease) -> Result<(), String> {
        self.require_lifecycle_ready().await?;
        let request = ReleaseLeaseRequest {
            contract_version: RELEASE_LEASE_REQUEST_CONTRACT_VERSION.into(),
            lease: lease.clone(),
        };
        request.validate().map_err(|error| error.to_string())?;
        let response = self
            .http
            .as_ref()
            .expect("enabled cloud plugin has an HTTP client")
            .post(format!("{}/plugin/v1/leases/release", self.config.endpoint))
            .json(&request)
            .timeout(self.config.lifecycle_timeout)
            .send()
            .await
            .map_err(|error| format!("cloud plugin Lease release unavailable: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "cloud plugin Lease release HTTP {status}: {}",
                body.chars().take(500).collect::<String>()
            ));
        }
        Ok(())
    }

    pub async fn commit_snapshot(
        &self,
        lease: &CloudLease,
        payload_base64: String,
        expected_checksum: String,
    ) -> Result<CloudStateReference, String> {
        let model_ref = self.lifecycle_model_ref()?;
        let target_tier = self.config.state_target_tier.clone();
        let request = SnapshotStateRequest {
            contract_version: SNAPSHOT_REQUEST_CONTRACT_VERSION.into(),
            provider_mode: "rwkv_recurrent".into(),
            model_ref: model_ref.clone(),
            target_tier: target_tier.clone(),
            lease: lease.clone(),
            expected_state_version: lease.expected_state_version,
            payload_base64,
            expected_checksum: Some(expected_checksum.clone()),
        };
        request.validate().map_err(|error| error.to_string())?;
        let state_ref = self
            .lifecycle_post::<_, CloudStateReference>("/plugin/v1/states/snapshot", &request)
            .await?;
        state_ref.validate().map_err(|error| error.to_string())?;
        if state_ref.session_id != lease.session_id
            || state_ref.owner_id != lease.owner_id
            || state_ref.version != lease.expected_state_version.saturating_add(1)
            || state_ref.fencing_token != Some(lease.fencing_token)
            || state_ref.provider_mode != "rwkv_recurrent"
            || state_ref.model_ref != model_ref
            || state_ref.placement != target_tier
            || state_ref.checksum != expected_checksum
        {
            return Err("cloud plugin committed a mismatched State reference".into());
        }
        Ok(state_ref)
    }

    pub async fn read_state(
        &self,
        state_ref: CloudStateReference,
        worker_id: &str,
        lease: &CloudLease,
    ) -> Result<CloudRestoreStateResponse, String> {
        state_ref.validate().map_err(|error| error.to_string())?;
        lease.validate().map_err(|error| error.to_string())?;
        if lease.session_id != state_ref.session_id
            || lease.owner_id != state_ref.owner_id
            || lease.expected_state_version != state_ref.version
        {
            return Err("cloud State restore Lease does not fence the requested State".into());
        }
        let request = RestoreStateRequest {
            contract_version: RESTORE_REQUEST_CONTRACT_VERSION.into(),
            state_ref: state_ref.clone(),
            expected_model_ref: self.lifecycle_model_ref()?,
            target_worker_id: worker_id.into(),
            lease: lease.clone(),
        };
        request.validate().map_err(|error| error.to_string())?;
        let response = self
            .lifecycle_post::<_, CloudRestoreStateResponse>("/plugin/v1/states/restore", &request)
            .await?;
        response.validate().map_err(|error| error.to_string())?;
        if response.contract_version != RESTORE_RESPONSE_CONTRACT_VERSION
            || response.state_ref != state_ref
        {
            return Err("cloud plugin returned a mismatched State restore".into());
        }
        Ok(response)
    }

    pub async fn record_usage(&self, record: &CloudUsageRecord) -> Result<(), String> {
        record.validate().map_err(|error| error.to_string())?;
        if !self.finops_ready().await {
            return Err("cloud plugin is not ready or does not advertise finops accounting".into());
        }
        let response = self
            .http
            .as_ref()
            .expect("enabled cloud plugin has an HTTP client")
            .post(format!("{}/plugin/v1/usage", self.config.endpoint))
            .json(record)
            .send()
            .await
            .map_err(|error| format!("cloud plugin usage accounting unavailable: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "cloud plugin usage accounting HTTP {status}: {}",
                body.chars().take(500).collect::<String>()
            ));
        }
        Ok(())
    }

    async fn lifecycle_post<T: Serialize + ?Sized, R: DeserializeOwned>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<R, String> {
        self.require_lifecycle_ready().await?;
        let response = self
            .http
            .as_ref()
            .expect("enabled cloud plugin has an HTTP client")
            .post(format!("{}{path}", self.config.endpoint))
            .json(body)
            .timeout(self.config.lifecycle_timeout)
            .send()
            .await
            .map_err(|error| format!("cloud State lifecycle unavailable: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            let value = response.text().await.unwrap_or_default();
            return Err(format!(
                "cloud State lifecycle HTTP {status}: {}",
                value.chars().take(500).collect::<String>()
            ));
        }
        response
            .json::<R>()
            .await
            .map_err(|error| format!("cloud State lifecycle invalid JSON: {error}"))
    }

    async fn require_lifecycle_ready(&self) -> Result<(), String> {
        if !self.state_lifecycle_enabled() {
            return Err("cloud State lifecycle is disabled".into());
        }
        if self.state.read().await.status != "ready" {
            return Err("cloud State lifecycle is not ready".into());
        }
        Ok(())
    }

    fn lifecycle_model_ref(&self) -> Result<CloudModelRef, String> {
        self.config
            .model_ref
            .clone()
            .ok_or_else(|| "cloud plugin model_ref is unavailable".into())
    }

    fn effective_required_capabilities(&self) -> Vec<String> {
        let mut capabilities = self.config.required_capabilities.clone();
        if self.config.state_lifecycle {
            for capability in ["leases", "state_lifecycle"] {
                if !capabilities.iter().any(|value| value == capability) {
                    capabilities.push(capability.into());
                }
            }
        }
        capabilities
    }

    fn validate_plan(&self, plan: &ExecutionPlan, request_id: &str) -> Result<(), String> {
        if plan.contract_version != EXECUTION_PLAN_CONTRACT_VERSION
            || plan.request_id != request_id
            || plan.decision_id.trim().is_empty()
        {
            return Err("cloud plugin returned a mismatched execution plan".into());
        }
        if !matches!(plan.mode.as_str(), "local" | "remote" | "defer" | "reject") {
            return Err("cloud plugin returned an unknown execution mode".into());
        }
        if plan.mode == "remote"
            && (plan.worker_id.as_deref().is_none_or(str::is_empty)
                || plan.endpoint.as_deref().is_none_or(str::is_empty))
        {
            return Err("cloud plugin remote plan has no Worker or endpoint".into());
        }
        if plan.mode == "local" && (plan.worker_id.is_some() != plan.endpoint.is_some()) {
            return Err(
                "cloud plugin local plan must provide both Worker and endpoint or neither".into(),
            );
        }
        if self.config.default_privacy == PrivacyClass::LocalOnly && plan.mode == "remote" {
            return Err("cloud plugin violated local_only privacy policy".into());
        }
        Ok(())
    }

    fn fallback_or_error(&self, request_id: String, error: &str) -> Result<ExecutionPlan, String> {
        match self.config.fallback {
            CloudPluginFallback::Local => {
                Ok(ExecutionPlan::local(request_id, "provider_unavailable"))
            }
            CloudPluginFallback::FailClosed => Err(error.into()),
        }
    }

    fn fallback_or_error_after_plan(
        &self,
        request_id: String,
        error: &str,
        has_committed_state: bool,
    ) -> Result<ExecutionPlan, String> {
        if has_committed_state {
            Err(format!(
                "{error}; committed remote State exists, so local fallback is forbidden"
            ))
        } else {
            self.fallback_or_error(request_id, error)
        }
    }

    pub async fn readiness(&self) -> Value {
        let state = self.state.read().await;
        json!({
            "status":state.status,
            "enabled":self.config.enabled,
            "fallback":self.config.fallback,
            "endpoint":if self.config.enabled {Some(self.config.endpoint.as_str())} else {None},
            "plugin_version":state.plugin_version,
            "capabilities":state.capabilities,
            "state_lifecycle":self.config.state_lifecycle,
            "state_target_tier":self.config.state_target_tier,
            "lease_ttl_ms":self.config.lease_ttl.as_millis(),
            "lifecycle_timeout_ms":self.config.lifecycle_timeout.as_millis(),
            "error":state.error,
        })
    }

    pub async fn blocks_readiness(&self) -> bool {
        self.config.enabled
            && self.config.fallback == CloudPluginFallback::FailClosed
            && self.state.read().await.status != "ready"
    }
}

fn sanitize_id(value: &str) -> String {
    let value = value
        .chars()
        .take(32)
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    if value.is_empty() {
        "session".into()
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use axum::{Json, Router, extract::State as AxumState, http::StatusCode, routing::post};

    fn enabled_config() -> CloudPluginConfig {
        CloudPluginConfig {
            enabled: true,
            model_ref: Some(CloudModelRef {
                model_id: "rwkv7".into(),
                revision: "revision".into(),
                tokenizer: "tokenizer".into(),
                state_abi: "rwkv7-state-v1".into(),
            }),
            default_privacy: PrivacyClass::CloudAllowed,
            ..CloudPluginConfig::default()
        }
    }

    fn state_reference(version: u64, fencing_token: u64) -> CloudStateReference {
        CloudStateReference {
            contract_version: rwkv_statepool_plugin_api::STATE_REFERENCE_CONTRACT_VERSION.into(),
            state_id: format!("state-{version}"),
            session_id: "session".into(),
            owner_id: "owner".into(),
            version,
            fencing_token: Some(fencing_token),
            provider_mode: "rwkv_recurrent".into(),
            model_ref: enabled_config().model_ref.unwrap(),
            placement: CloudStatePlacement::Cold,
            worker_id: None,
            object_uri: Some(format!("s3://statepool/state-{version}")),
            checksum: format!("sha256:{}", "a".repeat(64)),
            size_bytes: 5,
            atomic: true,
            created_at_ms: 1,
            last_active_at_ms: 1,
            encryption: None,
        }
    }

    #[derive(Clone, Default)]
    struct LifecycleMock {
        events: Arc<Mutex<Vec<String>>>,
    }

    async fn mock_lifecycle_plugin() -> (String, LifecycleMock) {
        async fn handshake(Json(request): Json<HandshakeRequest>) -> Json<Value> {
            assert!(request.required_capabilities.contains(&"placement".into()));
            assert!(request.required_capabilities.contains(&"leases".into()));
            assert!(
                request
                    .required_capabilities
                    .contains(&"state_lifecycle".into())
            );
            Json(json!({
                "contract_version":PLUGIN_CONTRACT_VERSION,
                "plugin":"statepool-cloud",
                "plugin_version":"test-lifecycle",
                "capabilities":["placement","leases","state_lifecycle","finops"]
            }))
        }

        async fn plan(Json(request): Json<PlanRequest>) -> Json<Value> {
            request.validate().unwrap();
            Json(json!({
                "contract_version":EXECUTION_PLAN_CONTRACT_VERSION,
                "decision_id":"decision-lifecycle",
                "request_id":request.request_id,
                "mode":"remote",
                "worker_id":"worker-test",
                "worker_zone":"cloud",
                "endpoint":"http://worker.test",
                "state_action":if request.state_ref.is_some() {"restore"} else {"none"},
                "reason_code":"state_affinity",
                "lease_required":true,
                "estimated_queue_ms":1.0,
                "estimated_restore_ms":1.0,
                "estimated_cost":null,
                "fallback":"fail_closed"
            }))
        }

        async fn acquire(
            AxumState(mock): AxumState<LifecycleMock>,
            Json(request): Json<AcquireLeaseRequest>,
        ) -> Json<CloudLease> {
            request.validate().unwrap();
            mock.events
                .lock()
                .unwrap()
                .push(format!("acquire:{}", request.expected_state_version));
            Json(CloudLease {
                contract_version: LEASE_CONTRACT_VERSION.into(),
                lease_id: format!("lease-{}", request.expected_state_version),
                session_id: request.session_id,
                owner_id: request.owner_id,
                holder_id: request.holder_id,
                fencing_token: request.expected_state_version + 1,
                expected_state_version: request.expected_state_version,
                expires_at_ms: 9_999_999_999_999,
            })
        }

        async fn renew(
            AxumState(mock): AxumState<LifecycleMock>,
            Json(request): Json<RenewLeaseRequest>,
        ) -> Json<CloudLease> {
            request.validate().unwrap();
            mock.events.lock().unwrap().push("renew".into());
            let mut lease = request.lease;
            lease.expires_at_ms += request.ttl_ms;
            Json(lease)
        }

        async fn release(
            AxumState(mock): AxumState<LifecycleMock>,
            Json(request): Json<ReleaseLeaseRequest>,
        ) -> StatusCode {
            request.validate().unwrap();
            mock.events
                .lock()
                .unwrap()
                .push(format!("release:{}", request.lease.expected_state_version));
            StatusCode::NO_CONTENT
        }

        async fn snapshot(
            AxumState(mock): AxumState<LifecycleMock>,
            Json(request): Json<SnapshotStateRequest>,
        ) -> Json<CloudStateReference> {
            request.validate().unwrap();
            mock.events.lock().unwrap().push("snapshot".into());
            Json(CloudStateReference {
                contract_version: rwkv_statepool_plugin_api::STATE_REFERENCE_CONTRACT_VERSION
                    .into(),
                state_id: "state-1".into(),
                session_id: request.lease.session_id,
                owner_id: request.lease.owner_id,
                version: request.expected_state_version + 1,
                fencing_token: Some(request.lease.fencing_token),
                provider_mode: request.provider_mode,
                model_ref: request.model_ref,
                placement: request.target_tier,
                worker_id: None,
                object_uri: Some("s3://statepool/state-1".into()),
                checksum: request.expected_checksum.unwrap(),
                size_bytes: 5,
                atomic: true,
                created_at_ms: 1,
                last_active_at_ms: 1,
                encryption: None,
            })
        }

        async fn restore(
            AxumState(mock): AxumState<LifecycleMock>,
            Json(request): Json<RestoreStateRequest>,
        ) -> Json<CloudRestoreStateResponse> {
            request.validate().unwrap();
            mock.events.lock().unwrap().push("restore".into());
            Json(CloudRestoreStateResponse {
                contract_version: RESTORE_RESPONSE_CONTRACT_VERSION.into(),
                state_ref: request.state_ref,
                payload_base64: "c3RhdGU=".into(),
            })
        }

        async fn usage(
            AxumState(mock): AxumState<LifecycleMock>,
            Json(record): Json<CloudUsageRecord>,
        ) -> StatusCode {
            record.validate().unwrap();
            mock.events.lock().unwrap().push("usage".into());
            StatusCode::ACCEPTED
        }

        let mock = LifecycleMock::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/plugin/v1/handshake", post(handshake))
            .route("/plugin/v1/plan", post(plan))
            .route("/plugin/v1/leases/acquire", post(acquire))
            .route("/plugin/v1/leases/renew", post(renew))
            .route("/plugin/v1/leases/release", post(release))
            .route("/plugin/v1/states/snapshot", post(snapshot))
            .route("/plugin/v1/states/restore", post(restore))
            .route("/plugin/v1/usage", post(usage))
            .with_state(mock.clone());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{address}"), mock)
    }

    async fn mock_plugin() -> String {
        async fn handshake() -> Json<Value> {
            Json(json!({
                "contract_version":PLUGIN_CONTRACT_VERSION,
                "plugin":"statepool-cloud",
                "plugin_version":"test",
                "capabilities":["placement"]
            }))
        }

        async fn plan(Json(request): Json<Value>) -> Json<Value> {
            Json(json!({
                "contract_version":EXECUTION_PLAN_CONTRACT_VERSION,
                "decision_id":"decision-test",
                "request_id":request["request_id"],
                "mode":"remote",
                "worker_id":"worker-test",
                "endpoint":"http://worker.test",
                "state_action":"none",
                "reason_code":"cloud_capacity",
                "lease_required":false,
                "estimated_queue_ms":1.0,
                "estimated_restore_ms":0.0,
                "estimated_cost":null,
                "fallback":"local"
            }))
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new()
                    .route("/plugin/v1/handshake", post(handshake))
                    .route("/plugin/v1/plan", post(plan)),
            )
            .await
            .unwrap();
        });
        format!("http://{address}")
    }

    #[tokio::test]
    async fn disabled_plugin_never_builds_http_and_returns_original_local_path() {
        let plugin = CloudPluginClient::new(CloudPluginConfig::default()).unwrap();
        plugin.initialize("test").await.unwrap();
        assert!(plugin.http.is_none());
        let plan = plugin.plan("session", "owner", 10, 20).await.unwrap();
        assert_eq!(plan.mode, "local");
        assert_eq!(plan.reason_code, "plugin_disabled");
        assert_eq!(plugin.readiness().await["status"], "disabled");
    }

    #[tokio::test]
    async fn enabled_plugin_requires_complete_model_identity() {
        let error = match CloudPluginClient::new(CloudPluginConfig {
            enabled: true,
            ..CloudPluginConfig::default()
        }) {
            Ok(_) => panic!("enabled plugin without a model identity must fail"),
            Err(error) => error,
        };
        assert!(error.contains("model_ref"));
    }

    #[tokio::test]
    async fn unavailable_plugin_degrades_to_local_only_before_remote_execution() {
        let plugin = CloudPluginClient::new(CloudPluginConfig {
            endpoint: "http://127.0.0.1:1".into(),
            connect_timeout: Duration::from_millis(5),
            request_timeout: Duration::from_millis(10),
            fallback: CloudPluginFallback::Local,
            ..enabled_config()
        })
        .unwrap();
        plugin.initialize("test").await.unwrap();
        let plan = plugin.plan("session", "owner", 10, 20).await.unwrap();
        assert_eq!(plan.mode, "local");
        assert_eq!(plan.reason_code, "provider_unavailable");
        assert_eq!(plugin.readiness().await["status"], "degraded");
    }

    #[tokio::test]
    async fn compatible_plugin_handshake_and_remote_plan_round_trip() {
        let plugin = CloudPluginClient::new(CloudPluginConfig {
            endpoint: mock_plugin().await,
            ..enabled_config()
        })
        .unwrap();
        plugin.initialize("test-host").await.unwrap();
        assert_eq!(plugin.readiness().await["status"], "ready");
        let plan = plugin.plan("session", "owner", 10, 20).await.unwrap();
        assert_eq!(plan.mode, "remote");
        assert_eq!(plan.worker_id.as_deref(), Some("worker-test"));
        assert_eq!(plan.remote_endpoint(), Some("http://worker.test"));
    }

    #[tokio::test]
    async fn lifecycle_capabilities_and_fenced_operations_round_trip() {
        let (endpoint, mock) = mock_lifecycle_plugin().await;
        let plugin = CloudPluginClient::new(CloudPluginConfig {
            endpoint,
            state_lifecycle: true,
            ..enabled_config()
        })
        .unwrap();
        plugin.initialize("test-host").await.unwrap();
        assert_eq!(plugin.readiness().await["status"], "ready");

        let lease0 = plugin
            .acquire_lease("session", "owner", "controller", 0)
            .await
            .unwrap();
        let lease0 = plugin.renew_lease(&lease0).await.unwrap();
        let state_ref = plugin
            .commit_snapshot(
                &lease0,
                "c3RhdGU=".into(),
                format!("sha256:{}", "a".repeat(64)),
            )
            .await
            .unwrap();
        plugin.release_lease(&lease0).await.unwrap();

        let plan = plugin
            .plan_with_state("session", "owner", 10, 20, Some(state_ref.clone()))
            .await
            .unwrap();
        assert_eq!(plan.state_action, "restore");
        let lease1 = plugin
            .acquire_lease("session", "owner", "controller", state_ref.version)
            .await
            .unwrap();
        let restored = plugin
            .read_state(state_ref.clone(), "worker-test", &lease1)
            .await
            .unwrap();
        assert_eq!(restored.state_ref, state_ref);
        assert_eq!(restored.payload_base64, "c3RhdGU=");
        plugin.release_lease(&lease1).await.unwrap();

        let usage: CloudUsageRecord = serde_json::from_str(include_str!(
            "../../../contracts/examples/statepool-plugin-v1/usage-record.json"
        ))
        .unwrap();
        assert!(plugin.finops_ready().await);
        plugin.record_usage(&usage).await.unwrap();

        assert_eq!(
            *mock.events.lock().unwrap(),
            [
                "acquire:0",
                "renew",
                "snapshot",
                "release:0",
                "acquire:1",
                "restore",
                "release:1",
                "usage"
            ]
        );
    }

    #[tokio::test]
    async fn lifecycle_handshake_requires_lease_and_state_capabilities() {
        let plugin = CloudPluginClient::new(CloudPluginConfig {
            endpoint: mock_plugin().await,
            state_lifecycle: true,
            ..enabled_config()
        })
        .unwrap();
        plugin.initialize("test-host").await.unwrap();
        let readiness = plugin.readiness().await;
        assert_eq!(readiness["status"], "degraded");
        assert!(readiness["error"].as_str().unwrap().contains("leases"));
    }

    #[tokio::test]
    async fn committed_state_never_falls_back_to_local_when_plugin_is_unavailable() {
        let plugin = CloudPluginClient::new(CloudPluginConfig {
            endpoint: "http://127.0.0.1:1".into(),
            connect_timeout: Duration::from_millis(5),
            request_timeout: Duration::from_millis(10),
            fallback: CloudPluginFallback::Local,
            state_lifecycle: true,
            ..enabled_config()
        })
        .unwrap();
        plugin.initialize("test-host").await.unwrap();
        let error = plugin
            .plan_with_state("session", "owner", 10, 20, Some(state_reference(1, 1)))
            .await
            .unwrap_err();
        assert!(error.contains("local fallback is forbidden"));
    }

    #[test]
    fn local_only_policy_rejects_remote_plan() {
        let plugin = CloudPluginClient::new(CloudPluginConfig {
            default_privacy: PrivacyClass::LocalOnly,
            ..enabled_config()
        })
        .unwrap();
        let plan = ExecutionPlan {
            contract_version: EXECUTION_PLAN_CONTRACT_VERSION.into(),
            decision_id: "decision".into(),
            request_id: "request".into(),
            mode: "remote".into(),
            worker_id: Some("worker".into()),
            worker_zone: Some(WorkerZone::Cloud),
            endpoint: Some("http://worker".into()),
            state_action: "none".into(),
            reason_code: "cloud_capacity".into(),
            lease_required: false,
            estimated_queue_ms: None,
            estimated_restore_ms: None,
            estimated_cost: None,
            fallback: "local".into(),
        };
        assert!(plugin.validate_plan(&plan, "request").is_err());
    }

    #[test]
    fn selected_local_endpoint_requires_a_worker_identity() {
        let plugin = CloudPluginClient::new(enabled_config()).unwrap();
        let mut plan = ExecutionPlan::local("request".into(), "local_within_slo");
        plan.endpoint = Some("http://local-worker".into());
        assert!(plugin.validate_plan(&plan, "request").is_err());
        plan.worker_id = Some("local-worker".into());
        assert!(plugin.validate_plan(&plan, "request").is_ok());
    }
}
