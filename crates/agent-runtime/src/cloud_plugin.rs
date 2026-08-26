use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::RwLock;

use rwkv_statepool_plugin_api::{
    EXECUTION_PLAN_CONTRACT_VERSION, HandshakeRequest, HandshakeResponse,
    PLAN_REQUEST_CONTRACT_VERSION, PLUGIN_CONTRACT_VERSION, PlanRequest,
};
pub use rwkv_statepool_plugin_api::{
    ExecutionPlan, ModelRef as CloudModelRef, PrivacyClass, WorkerZone,
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
        let response = self
            .http
            .as_ref()
            .ok_or_else(|| "cloud plugin is disabled".to_string())?
            .post(format!("{}/plugin/v1/handshake", self.config.endpoint))
            .json(&HandshakeRequest {
                contract_version: PLUGIN_CONTRACT_VERSION.into(),
                host: "rwkv-agent".into(),
                host_version: host_version.into(),
                required_capabilities: self.config.required_capabilities.clone(),
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
        let missing = self
            .config
            .required_capabilities
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
        let request_id = format!(
            "plugin-{}-{:016x}",
            sanitize_id(session_id),
            self.request_counter.fetch_add(1, Ordering::Relaxed)
        );
        if !self.config.enabled {
            return Ok(ExecutionPlan::local(request_id, "plugin_disabled"));
        }
        if self.state.read().await.status != "ready" {
            return self.fallback_or_error(
                request_id,
                "cloud plugin is not ready; no remote operation was started",
            );
        }
        let model_ref = self
            .config
            .model_ref
            .clone()
            .ok_or_else(|| "cloud plugin model_ref is unavailable".to_string())?;
        let response = match self
            .http
            .as_ref()
            .expect("enabled cloud plugin has an HTTP client")
            .post(format!("{}/plugin/v1/plan", self.config.endpoint))
            .json(&PlanRequest {
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
                state_ref: None,
                estimated_input_tokens,
                estimated_output_tokens,
            })
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                return self.fallback_or_error(
                    request_id,
                    &format!("cloud plugin planning unavailable: {error}"),
                );
            }
        };
        let status = response.status();
        let plan = match response.json::<ExecutionPlan>().await {
            Ok(plan) => plan,
            Err(error) => {
                return self.fallback_or_error(
                    request_id,
                    &format!("cloud plugin planning invalid JSON: {error}"),
                );
            }
        };
        if !status.is_success() {
            return self
                .fallback_or_error(request_id, &format!("cloud plugin planning HTTP {status}"));
        }
        if let Err(error) = self.validate_plan(&plan, &request_id) {
            return self.fallback_or_error(request_id, &error);
        }
        if matches!(plan.mode.as_str(), "defer" | "reject")
            && self.config.fallback == CloudPluginFallback::Local
        {
            return Ok(ExecutionPlan::local(request_id, &plan.reason_code));
        }
        Ok(plan)
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

    pub async fn readiness(&self) -> Value {
        let state = self.state.read().await;
        json!({
            "status":state.status,
            "enabled":self.config.enabled,
            "fallback":self.config.fallback,
            "endpoint":if self.config.enabled {Some(self.config.endpoint.as_str())} else {None},
            "plugin_version":state.plugin_version,
            "capabilities":state.capabilities,
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
    use axum::{Json, Router, routing::post};

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
}
