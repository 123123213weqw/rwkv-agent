use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use reqwest::Client;
use rwkv_agent_core::{
    ModelOutput, OpenStateRequest, RunContext, StateContinueRequest, StateHandle, StateModel,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SidecarState {
    #[serde(default)]
    pub owner_id: String,
    pub state_id: String,
    #[serde(default)]
    pub home_url: String,
    #[serde(default)]
    pub branch: String,
    #[serde(default)]
    pub seen_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchContinuation {
    pub state_id: String,
    #[serde(default)]
    pub branch: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub stop_reason: String,
    #[serde(default)]
    pub token_ids: Vec<u32>,
    #[serde(default)]
    pub seen_tokens: u64,
    #[serde(default)]
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateDecision {
    #[serde(default)]
    pub use_tool: bool,
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(flatten)]
    pub trace: serde_json::Map<String, Value>,
}

#[derive(Clone)]
pub struct SidecarClient {
    endpoints: Arc<Vec<String>>,
    next: Arc<AtomicUsize>,
    http: Client,
}

impl SidecarClient {
    pub fn new(endpoints: Vec<String>) -> Result<Self, String> {
        let endpoints = endpoints
            .into_iter()
            .map(|value| value.trim_end_matches('/').trim().to_string())
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>();
        if endpoints.is_empty() {
            return Err("at least one sidecar endpoint is required".into());
        }
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(190))
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self {
            endpoints: Arc::new(endpoints),
            next: Arc::new(AtomicUsize::new(0)),
            http,
        })
    }

    fn choose(&self) -> String {
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        self.endpoints[index % self.endpoints.len()].clone()
    }

    async fn get_json(&self, url: String) -> Result<Value, String> {
        let response = self.http.get(url).send().await.map_err(|e| e.to_string())?;
        let status = response.status();
        let value = response.json::<Value>().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!("sidecar HTTP {status}: {value}"));
        }
        Ok(value)
    }

    async fn post_json(&self, url: String, body: Value) -> Result<Value, String> {
        let response = self
            .http
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = response.status();
        let value = response.json::<Value>().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!("sidecar HTTP {status}: {value}"));
        }
        Ok(value)
    }

    pub async fn health(&self) -> Result<Vec<Value>, String> {
        let mut values = Vec::with_capacity(self.endpoints.len());
        for endpoint in self.endpoints.iter() {
            values.push(self.get_json(format!("{endpoint}/health")).await?);
        }
        Ok(values)
    }

    pub async fn gate(
        &self,
        message: &str,
        threshold: f64,
        context: &str,
        has_pasted_text: bool,
    ) -> Result<GateDecision, String> {
        let endpoint = self.choose();
        let value = self
            .post_json(
                format!("{endpoint}/v1/gate/tool"),
                json!({
                    "message": message,
                    "threshold": threshold,
                    "context": context,
                    "has_pasted_text": has_pasted_text,
                }),
            )
            .await?;
        serde_json::from_value(value).map_err(|e| e.to_string())
    }

    pub async fn prefill(&self, owner_id: &str, prompt: &str) -> Result<SidecarState, String> {
        let endpoint = self.choose();
        let value = self
            .post_json(
                format!("{endpoint}/v1/states/prefill"),
                json!({"owner_id": owner_id, "prompt": prompt, "branch": "root"}),
            )
            .await?;
        let state = value
            .get("state")
            .cloned()
            .ok_or_else(|| "sidecar prefill response has no state".to_string())?;
        let mut parsed: SidecarState = serde_json::from_value(state).map_err(|e| e.to_string())?;
        parsed.owner_id = owner_id.to_string();
        parsed.home_url = endpoint;
        Ok(parsed)
    }

    pub async fn fork(
        &self,
        root: &SidecarState,
        branches: &[String],
    ) -> Result<Vec<SidecarState>, String> {
        let value = self
            .post_json(
                format!("{}/v1/states/{}/fork", root.home_url, root.state_id),
                json!({"owner_id": root.owner_id, "branches": branches}),
            )
            .await?;
        value
            .get("states")
            .and_then(Value::as_array)
            .ok_or_else(|| "sidecar fork response has no states".to_string())?
            .iter()
            .map(|item| {
                let mut state: SidecarState =
                    serde_json::from_value(item.clone()).map_err(|e| e.to_string())?;
                state.owner_id = root.owner_id.clone();
                state.home_url = root.home_url.clone();
                Ok(state)
            })
            .collect()
    }

    pub async fn batch_continue(
        &self,
        home_url: &str,
        owner_id: &str,
        items: Vec<Value>,
        stops: &[String],
        max_tokens: u32,
    ) -> Result<Vec<BatchContinuation>, String> {
        let value = self
            .post_json(
                format!(
                    "{}/v1/states/batch_continue",
                    home_url.trim_end_matches('/')
                ),
                json!({
                    "owner_id": owner_id,
                    "items": items,
                    "stop": stops,
                    "max_tokens": max_tokens,
                }),
            )
            .await?;
        serde_json::from_value(
            value
                .get("results")
                .cloned()
                .ok_or_else(|| "sidecar continuation response has no results".to_string())?,
        )
        .map_err(|e| e.to_string())
    }

    pub async fn continue_one(
        &self,
        state: &SidecarState,
        input: &str,
        stops: &[String],
        max_tokens: u32,
    ) -> Result<BatchContinuation, String> {
        let mut rows = self
            .batch_continue(
                &state.home_url,
                &state.owner_id,
                vec![json!({"state_id": state.state_id, "input": input})],
                stops,
                max_tokens,
            )
            .await?;
        if rows.len() != 1 {
            return Err(format!("expected one continuation row, got {}", rows.len()));
        }
        Ok(rows.remove(0))
    }

    pub async fn release_many(
        &self,
        home_url: &str,
        owner_id: &str,
        state_ids: &[String],
    ) -> Result<Value, String> {
        self.post_json(
            format!("{}/v1/states/release", home_url.trim_end_matches('/')),
            json!({"owner_id": owner_id, "state_ids": state_ids}),
        )
        .await
    }

    pub async fn complete(
        &self,
        prompt: &str,
        stops: &[String],
        max_tokens: u32,
    ) -> Result<BatchContinuation, String> {
        let endpoint = self.choose();
        let value = self
            .post_json(
                format!("{endpoint}/v1/completions"),
                json!({"prompt": prompt, "stop": stops, "max_tokens": max_tokens}),
            )
            .await?;
        let g1i = value
            .get("g1i")
            .cloned()
            .ok_or_else(|| "completion response has no g1i".to_string())?;
        serde_json::from_value(g1i).map_err(|e| e.to_string())
    }

    fn reconstruct_envelope(input: &str, row: &BatchContinuation) -> String {
        let generated = row.text.trim_start();
        match row.stop_reason.as_str() {
            "</tool_call>" => {
                let prefix = if generated.starts_with("<tool_call>") {
                    ""
                } else {
                    "<tool_call>"
                };
                let suffix = if generated.ends_with("</tool_call>") {
                    ""
                } else {
                    "</tool_call>"
                };
                format!("{prefix}{generated}{suffix}")
            }
            "</answer>" => {
                let prefix = if generated.starts_with("<answer>") {
                    ""
                } else {
                    "<answer>"
                };
                let suffix = if generated.ends_with("</answer>") {
                    ""
                } else {
                    "</answer>"
                };
                format!("{prefix}{generated}{suffix}")
            }
            _ if input.trim_end().ends_with("<tool_call>") => {
                format!("<tool_call>{generated}")
            }
            _ if input.trim_end().ends_with("<answer>") => format!("<answer>{generated}"),
            _ => row.text.clone(),
        }
    }
}

impl StateModel for SidecarClient {
    async fn open(
        &mut self,
        request: OpenStateRequest,
        context: RunContext,
    ) -> Result<StateHandle, String> {
        context.check().map_err(|e| e.to_string())?;
        let state = tokio::time::timeout(
            context.remaining(),
            self.prefill(&request.owner_id, &request.root_prompt),
        )
        .await
        .map_err(|_| "run deadline exceeded during State prefill".to_string())??;
        Ok(StateHandle {
            endpoint: state.home_url,
            owner_id: state.owner_id,
            state_id: state.state_id,
        })
    }

    async fn continue_state(
        &mut self,
        request: StateContinueRequest,
        context: RunContext,
    ) -> Result<ModelOutput, String> {
        context.check().map_err(|e| e.to_string())?;
        let state = SidecarState {
            owner_id: request.state.owner_id,
            state_id: request.state.state_id.clone(),
            home_url: request.state.endpoint,
            branch: String::new(),
            seen_tokens: 0,
        };
        let row = tokio::time::timeout(
            context.remaining(),
            self.continue_one(&state, &request.input, &request.stops, request.max_tokens),
        )
        .await
        .map_err(|_| "run deadline exceeded during State continuation".to_string())??;
        let text = Self::reconstruct_envelope(&request.input, &row);
        Ok(ModelOutput {
            state_id: row.state_id,
            text,
            stop_reason: (!row.stop_reason.is_empty()).then_some(row.stop_reason),
        })
    }

    async fn release(&mut self, state: StateHandle) -> Result<(), String> {
        self.release_many(&state.endpoint, &state.owner_id, &[state.state_id])
            .await
            .map(|_| ())
    }
}
